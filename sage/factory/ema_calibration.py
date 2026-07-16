#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-training EMA finalisation (run as a SEPARATE step, never in the hot path).

The per-step weight EMA (``ema.pt``) exponentially-averages the training weights,
but for a BatchNorm-heavy net the averaged weights' *correct* BN running
statistics are NOT the exponentially-averaged BN buffers -- they are the
activation stats *under the averaged weights*. So we:

1. **Recalibrate BatchNorm** for the EMA weights: reset the BN running stats and
   forward ``bn_batches`` training-distribution batches (BN in cumulative-average
   mode, no grad), so ``running_mean/var`` match the EMA weights. Saved as a NEW
   file ``ema_calibrated.pt`` (nothing is deleted). This is exactly
   ``torch.optim.swa_utils.update_bn`` adapted to Sage's signal+noise batch build.
2. **Compare** the calibrated EMA against ``best.pt`` on the validation set, using
   the SAME seeded validation batches for both (fair), and write a NOTE
   (``ema_vs_best.{json,txt}``) recording which won. **No weights are removed** --
   the note just says which to prefer. (Plots come later, outside this step.)
"""

import os
import json
import torch
from contextlib import nullcontext
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.swa_utils import update_bn

from sage.core.pipeline import GWBatch, Grid, ProcessingState


# --------------------------------------------------------------------------- io
def _clean_sd(sd):
    """Strip a ``torch.compile`` ``_orig_mod.`` prefix so a compiled-model
    checkpoint loads into the plain (uncompiled) module used here."""
    p = "_orig_mod."
    return {(k[len(p):] if k.startswith(p) else k): v for k, v in sd.items()}


def _load_weights_into(model, path, device, key=None):
    """Load ``path`` (optionally the sub-dict ``key``) into ``model`` (strict)."""
    obj = torch.load(path, map_location=device, weights_only=False)
    sd = obj[key] if key is not None else obj
    model.load_state_dict(_clean_sd(sd), strict=True)


# --------------------------------------------------------------- batch building
def _batch_meta(signal_sampler, cfg):
    """The fixed per-batch geometry (mirrors SageVanillaTraining/Validation)."""
    return dict(
        B=cfg.batch_size,
        S=int(cfg.batch_size * cfg.class_balance),
        num_pe=len(cfg.do_point_estimate),
        initial_state=getattr(signal_sampler, "output_state",
                              ProcessingState(Grid.FD_UNIFORM)),
        selector=getattr(signal_sampler, "selector", None),
    )


def _build_batch(signal_sampler, noise_sampler, processor, meta, device,
                 want_targets):
    """One noise+injected-signal batch -> (net_input, targets or None).

    Identical construction to the training/validation loops so BN sees the same
    input distribution it will at inference, and the val loss matches training's.
    """
    sel = meta["selector"]
    freqs = sel.coarse_freqs if sel is not None else None
    coarse = sel.coarse_indices if sel is not None else None
    B, S, num_pe = meta["B"], meta["S"], meta["num_pe"]

    signal_data, signal_targets = signal_sampler()
    noise_data, noise_targets = noise_sampler()
    if sel is not None:
        noise_data = sel(noise_data)

    idx = torch.randperm(B, device=device)[:S]
    signal_pad = torch.zeros_like(noise_data)
    signal_pad[idx] = signal_data
    x = noise_data + signal_pad

    targets = None
    if want_targets:
        pad = torch.zeros(noise_targets.shape[0], num_pe,
                          device=device, dtype=noise_targets.dtype)
        noise_targets = torch.cat((pad, noise_targets), dim=1)
        target_pad = torch.zeros(B, num_pe + 1, device=device,
                                 dtype=signal_targets.dtype)
        target_pad[idx] = signal_targets
        targets = noise_targets + target_pad

    batch = GWBatch(x, state=meta["initial_state"], freqs=freqs,
                    coarse_indices=coarse)
    batch = processor(batch)
    return batch.to_network_input(), targets


# ------------------------------------------------------------ core computations
def _bn_loader(signal_sampler, noise_sampler, processor, meta, device, n_batches):
    """Yield ``n_batches`` training-distribution net_input tensors for update_bn."""
    for _ in range(int(n_batches)):
        net_input, _ = _build_batch(signal_sampler, noise_sampler, processor,
                                    meta, device, want_targets=False)
        yield net_input


def recalibrate_bn(model, signal_sampler, noise_sampler, processor, meta,
                   n_batches, device):
    """Recompute BatchNorm running stats to match ``model``'s current weights,
    using the official ``torch.optim.swa_utils.update_bn`` (it resets the stats,
    sets ``momentum=None`` -> cumulative average, and forwards each batch). Wrapped
    in ``no_grad`` so the forwards build no autograd graph. Returns the number of
    BatchNorm layers (0 = nothing to do).
    """
    n_bn = sum(1 for m in model.modules() if isinstance(m, _BatchNorm))
    if n_bn == 0:
        return 0
    loader = _bn_loader(signal_sampler, noise_sampler, processor, meta,
                        device, n_batches)
    with torch.no_grad():
        update_bn(loader, model, device=device)
    return n_bn


def mean_val_components(model, signal_sampler, noise_sampler, processor, loss_fn,
                        meta, n_iters, device, amp_dtype, autocast):
    """Mean per-component validation loss over ``n_iters`` batches (component 0
    is the primary detection loss used for the comparison)."""
    model.eval()
    acc = None
    with torch.inference_mode():
        for _ in range(int(n_iters)):
            net_input, targets = _build_batch(signal_sampler, noise_sampler,
                                              processor, meta, device,
                                              want_targets=True)
            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if autocast else nullcontext())
            with ctx:
                out = model(net_input)
                loss = loss_fn(out, targets)
            comp = loss.detach().float()
            acc = comp if acc is None else acc + comp
    return (acc / max(1, int(n_iters))).cpu()


# ------------------------------------------------------------------ orchestrator
def calibrate_and_compare(cfg, ckpt_dir, model, bn_signal, bn_noise,
                          make_val_graph, processor, loss_fn,
                          bn_batches, val_iters, amp_dtype):
    """Recalibrate BN for the EMA weights, then compare calibrated-EMA vs best.pt
    on the (same, seeded) validation set and write a note. Deletes nothing.

    ``make_val_graph()`` must return a FRESH, identically-seeded ``(sig, noise)``
    each call, so both models are evaluated on the exact same validation batches.
    Returns the note dict.
    """
    device = cfg.device
    ema_path = os.path.join(ckpt_dir, "ema.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    cal_path = os.path.join(ckpt_dir, "ema_calibrated.pt")
    if not os.path.exists(ema_path):
        raise FileNotFoundError(f"no EMA weights to calibrate: {ema_path}")

    from sage.utils.checkpoint import _atomic_torch_save, _atomic_json_dump

    # 1. Load EMA weights, 2. recalibrate BN on TRAINING-distribution batches.
    _load_weights_into(model, ema_path, device)
    bn_meta = _batch_meta(bn_signal, cfg)
    n_bn = recalibrate_bn(model, bn_signal, bn_noise, processor, bn_meta,
                          bn_batches, device)
    print(f"[EMA-cal] recalibrated {n_bn} BatchNorm layers over {bn_batches} "
          f"training batches", flush=True)

    # 3. Save the calibrated EMA as a NEW file (nothing removed).
    _atomic_torch_save(model.state_dict(), cal_path)
    print(f"[EMA-cal] wrote {cal_path}", flush=True)

    # 4. Evaluate calibrated EMA on a fresh (seeded) val graph.
    vs, vn = make_val_graph()
    va_meta = _batch_meta(vs, cfg)
    ema_comp = mean_val_components(model, vs, vn, processor, loss_fn, va_meta,
                                   val_iters, device, amp_dtype, cfg.autocast)

    # 5. Evaluate best.pt on an IDENTICALLY-seeded val graph (same batches).
    best_comp = None
    if os.path.exists(best_path):
        _load_weights_into(model, best_path, device, key="model_state_dict")
        vs2, vn2 = make_val_graph()
        best_comp = mean_val_components(model, vs2, vn2, processor, loss_fn,
                                        va_meta, val_iters, device, amp_dtype,
                                        cfg.autocast)

    # 6. Decide + write the note (lower primary-loss component wins). Keep BOTH.
    ema_loss = float(ema_comp[0])
    best_loss = float(best_comp[0]) if best_comp is not None else None
    if best_loss is None:
        winner = "ema_calibrated"          # nothing to compare against
    else:
        winner = "ema_calibrated" if ema_loss <= best_loss else "best"
    note = {
        "winner_on_validation": winner,
        "ema_calibrated_pt": {"val_loss": ema_loss,
                              "val_components": [float(x) for x in ema_comp]},
        "best_pt": (None if best_comp is None else
                    {"val_loss": best_loss,
                     "val_components": [float(x) for x in best_comp]}),
        "delta_best_minus_ema": (None if best_loss is None
                                 else best_loss - ema_loss),
        "bn_layers_recalibrated": n_bn,
        "bn_batches": int(bn_batches),
        "val_iters": int(val_iters),
        "note": ("No weights were deleted. Prefer the winner above for "
                 "evaluation/deployment; both ema_calibrated.pt and best.pt "
                 "remain on disk. Plots are a separate, later step."),
    }
    _atomic_json_dump(note, os.path.join(ckpt_dir, "ema_vs_best.json"))
    lines = [
        "EMA calibration + validation comparison",
        "=" * 44,
        f"WINNER ON VALIDATION : {winner}",
        f"ema_calibrated.pt    : val_loss = {ema_loss:.6f}",
        (f"best.pt              : val_loss = {best_loss:.6f}"
         if best_loss is not None else "best.pt              : (absent)"),
        (f"delta (best - ema)   : {best_loss - ema_loss:+.6f}"
         if best_loss is not None else ""),
        f"BN layers recalibrated: {n_bn} over {bn_batches} training batches "
        f"(val over {val_iters} iters)",
        "",
        "No weights deleted. Prefer the winner; both files remain on disk.",
    ]
    with open(os.path.join(ckpt_dir, "ema_vs_best.txt"), "w") as f:
        f.write("\n".join(l for l in lines if l != "") + "\n")
    print("[EMA-cal] " + " | ".join([
        f"winner={winner}", f"ema={ema_loss:.6f}",
        ("best=%.6f" % best_loss) if best_loss is not None else "best=absent",
    ]), flush=True)
    return note

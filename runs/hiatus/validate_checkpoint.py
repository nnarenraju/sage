#!/usr/bin/env python3
"""
validate_checkpoint.py
Run one validation pass for a given epoch checkpoint with a configurable
number of samples.  Can be used from the notebook or as a standalone script.

Usage (script):
    python3 validate_checkpoint.py --epoch 79 --num-samples 20000
    python3 validate_checkpoint.py --epoch 79 --num-samples 20000 --save val_ep79.h5

Usage (notebook / import):
    from validate_checkpoint import run_checkpoint_validation
    result = run_checkpoint_validation(epoch=79, num_samples=20000)
"""

import os
import sys
import math
import argparse
from contextlib import nullcontext

import h5py
import numpy as np
import torch
from tqdm import tqdm

# ── Ensure sage is importable ────────────────────────────────────────────────
RUN_DIR  = os.path.dirname(os.path.abspath(__file__))
SAGE_DIR = os.path.join(RUN_DIR, "..", "..")
sys.path.insert(0, RUN_DIR)
sys.path.insert(0, SAGE_DIR)

# Config must be registered before sage imports
import os as _os, importlib as _il
# Config module to run: named by SAGE_CONFIG (default "config"). Make a
# per-network config with `cp config.py config_<DETS>.py`, edit `detectors`
# + `export_dir`, and pick it at launch. No network logic lives in code.
set_configs = _il.import_module(_os.environ.get("SAGE_CONFIG", "config")).set_configs
set_configs()

from sage.core.config import get_cfg

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
from sage.data.waveform import HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.data.noise import MemmapNoiseSampler
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning
from sage.core.graph import Preprocessor
from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic
from sage.architecture.custom_losses import BCEWithPEsigmaLoss


CHECKPOINTS_DIR = os.path.join(RUN_DIR, "run_export", "CHECKPOINTS")
GWCONFIG_PATH   = os.path.join(RUN_DIR, "gwconfig.yaml")


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint(epoch):
    """Return the checkpoint path for a given epoch number or path string."""
    if isinstance(epoch, str) and os.path.isfile(epoch):
        return epoch
    # epoch_79 → epoch_79.pt  (also accepts 79 as int)
    name = f"epoch_{int(epoch)}.pt"
    path = os.path.join(CHECKPOINTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _build_validation_graph(cfg):
    """Build signal sampler, noise sampler, processor (validation seeds)."""
    param_sampler   = read_from_config(GWCONFIG_PATH, seed=170817)
    snr_sampler     = HalfNorm(scale=4.0, loc=5.0, seed=170817)
    snrscaler       = OptimalSNRRescaler(snr_sampler)
    signal_sampler  = IMRPhenomPv2(
        param_sampler,
        ConstantProjection(),
        augment=snrscaler,
    )
    noise_sampler   = MemmapNoiseSampler(postprocess_fn=None, prefetch=4, seed=170817)

    whitener        = FiducialWhitening()
    dyadic_binning  = DyadicPyramidBinning(param_sampler.bounds)
    mrsampler       = MultirateSampler(binning_method=dyadic_binning)
    processor       = Preprocessor([whitener, mrsampler])

    return signal_sampler, noise_sampler, processor, param_sampler


def _build_model(cfg):
    """Construct the model architecture (un-compiled, for inference)."""
    model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="groupnorm",   # MUST match training (train_hard.py); else weights load into wrong norm
    )
    return model


def _load_weights(model, checkpoint_path):
    """Load state dict from a checkpoint, stripping torch.compile prefix."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan"))


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def run_checkpoint_validation(epoch, num_samples, save_path=None, verbose=True):
    """
    Run one validation pass for the given checkpoint.

    Parameters
    ----------
    epoch       : int or str
        Epoch number (e.g. 79) or full path to a .pt file.
    num_samples : int
        Total number of samples (signals + noise) to process.
        Rounded up to the nearest complete batch.
    save_path   : str, optional
        If given, save results to this HDF5 file under group ``"epoch_{epoch:04d}"``.
    verbose     : bool
        Print progress bar and summary.

    Returns
    -------
    dict with keys:
        network_output  : (N, 5) float32 ndarray
                          [ranking_stat, tc_pred, mchirp_pred, tc_sigma, mchirp_sigma]
        network_target  : (N, 3) float32 ndarray
                          [tc_std, mchirp_std, label]
        signal_params   : (S_total, 25) float32 ndarray – physical params per signal
        signal_idx      : (num_iter, S) int32 ndarray – batch placement indices
        labels          : (N,) float32 – 0=noise, 1=signal
        ranking_stat    : (N,) float32 – sigmoid pre-activation logit
        pred_prob       : (N,) float32 – sigmoid(ranking_stat)
        epoch_loaded    : int
        val_loss        : float – from checkpoint metadata
    """
    from scipy.special import expit

    cfg = get_cfg()
    device = cfg.device
    dtype  = torch.float32

    ckpt_path = _resolve_checkpoint(epoch)
    if verbose:
        print(f"Checkpoint : {ckpt_path}")

    # ── Build model ─────────────────────────────────────────────────────────
    model = _build_model(cfg)
    epoch_loaded, val_loss = _load_weights(model, ckpt_path)
    model.eval()
    model.to(device=device, dtype=dtype)
    if verbose:
        print(f"Loaded epoch {epoch_loaded}  (val_loss={val_loss:.6f})")

    # ── Build validation graph ───────────────────────────────────────────────
    signal_sampler, noise_sampler, processor, param_sampler = _build_validation_graph(cfg)
    processor.eval()
    processor.to(device=device)

    # ── Derive iteration count ────────────────────────────────────────────────
    B          = cfg.batch_size                           # total samples per batch
    S          = int(B * cfg.class_balance)               # signals per batch
    num_pe     = len(cfg.do_point_estimate)               # number of PE targets
    num_targets = num_pe + 1                              # PE + label

    num_iter = math.ceil(num_samples / B)
    N_actual = num_iter * B
    if verbose:
        print(f"Batch size : {B}  ({S} signal, {B-S} noise)")
        print(f"Iterations : {num_iter}  →  {N_actual:,} total samples")

    loss_fn = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    loss_fn.to(device=device, dtype=dtype)

    # ── Validation loop ───────────────────────────────────────────────────────
    save = {k: [] for k in ("network_output", "network_target", "signal_params", "signal_idx")}
    total_loss = torch.zeros(4, device=device)  # [total, bce, reg, coupling]

    with torch.inference_mode():
        for _ in tqdm(range(num_iter), desc="Validation", disable=not verbose):

            # Generate data
            signal_data, signal_targets, theta = signal_sampler(return_theta=True)
            noise_data,  noise_targets         = noise_sampler()

            # Pad noise targets to match signal target shape: [pe..., label]
            pad = torch.zeros(
                noise_targets.shape[0], num_pe,
                device=device, dtype=noise_targets.dtype,
            )
            noise_targets = torch.cat((pad, noise_targets), dim=1)

            # Random signal placement inside the batch
            idx = torch.randperm(B, device=device)[:S]
            save["signal_idx"].append(idx.cpu())

            signal_pad = torch.zeros_like(noise_data)
            target_pad = torch.zeros(B, num_targets, device=device, dtype=signal_targets.dtype)
            signal_pad[idx] = signal_data
            target_pad[idx] = signal_targets

            x       = noise_data + signal_pad
            targets = noise_targets + target_pad

            # Preprocess and forward
            x = processor(x)
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if cfg.autocast else nullcontext()
            ):
                out  = model(x)
                loss = loss_fn(out, targets)

            total_loss += loss.detach()

            # Unstandardise PE predictions to physical units
            network_output = torch.cat([*out], dim=1)
            ranking  = network_output[:, 0:1]
            mu_std   = network_output[:, 1 : 1 + num_pe]
            log_var  = network_output[:, 1 + num_pe : 1 + 2 * num_pe]
            mu_phys  = signal_sampler.param_sampler.unstandardise_from_batch(mu_std)
            # log_var is in standardised parameter space; multiply by prior std
            # to obtain sigma in physical units (same fix as SageUncompiledValidation).
            sig_std  = torch.exp(0.5 * log_var)
            std_prior = signal_sampler.param_sampler._std_stds.to(sig_std.device)
            sig_phys = sig_std * std_prior
            network_output = torch.cat([ranking, mu_phys, sig_phys], dim=1)

            save["network_output"].append(network_output.cpu())
            save["network_target"].append(targets.cpu())
            save["signal_params"].append(theta.cpu())

    avg_losses = (total_loss / num_iter).cpu().numpy()  # [total, bce, reg, coupling]
    avg_loss = float(avg_losses[0])
    if verbose:
        print(f"Avg loss over this run : {avg_loss:.6f}  "
              f"(bce={avg_losses[1]:.6f}, reg={avg_losses[2]:.6f}, coupling={avg_losses[3]:.6f})")

    # ── Stack results ─────────────────────────────────────────────────────────
    network_output = torch.vstack(save["network_output"]).numpy()
    network_target = torch.vstack(save["network_target"]).numpy()
    signal_params  = torch.vstack(save["signal_params"]).numpy()
    signal_idx     = torch.stack(save["signal_idx"]).numpy()   # (num_iter, S)

    labels       = network_target[:, -1]
    ranking_stat = network_output[:, 0]

    # ── Optionally save to HDF5 ───────────────────────────────────────────────
    if save_path is not None:
        ep_int = epoch_loaded if isinstance(epoch_loaded, int) else int(epoch)
        group  = f"epoch_{ep_int:04d}"
        with h5py.File(save_path, "a") as f:
            if group in f:
                del f[group]
            grp = f.create_group(group)
            grp.create_dataset("network_output", data=network_output, compression="gzip")
            grp.create_dataset("network_target", data=network_target, compression="gzip")
            grp.create_dataset("signal_params",  data=signal_params,  compression="gzip")
            grp.create_dataset("signal_idx",     data=signal_idx,     compression="gzip")
        if verbose:
            print(f"Saved to {save_path}  [{group}]")

    return {
        "network_output" : network_output,
        "network_target" : network_target,
        "signal_params"  : signal_params,
        "signal_idx"     : signal_idx,
        "labels"         : labels,
        "ranking_stat"   : ranking_stat,
        "pred_prob"      : expit(ranking_stat).astype(np.float32),
        "epoch_loaded"   : epoch_loaded,
        "val_loss"       : val_loss,
        "avg_loss_run"   : avg_loss,
        "avg_losses_run" : avg_losses,  # [total, bce, reg, coupling]
        "num_samples"    : N_actual,
    }


# ────────────────────────────────────────────────────────────────────────────
# Alignment helper
# ────────────────────────────────────────────────────────────────────────────

_PARAM_NAMES = [
    "chirp_distance", "coa_phase", "dec", "distance", "inclination",
    "injection_time", "mass1", "mass2", "mchirp", "polarization", "q", "ra",
    "spin1_a", "spin1_azimuthal", "spin1_polar", "spin1x", "spin1y", "spin1z",
    "spin2_a", "spin2_azimuthal", "spin2_polar", "spin2x", "spin2y", "spin2z",
    "tc",
]


def build_source_params(result):
    """
    Align signal_params with per-sample network_output rows and return a dict
    mapping param name → full-length array (NaN for noise rows).

    Parameters
    ----------
    result : dict
        Return value of run_checkpoint_validation().

    Returns
    -------
    dict[str, np.ndarray]  shape (N,) each, NaN for noise rows.
    """
    signal_idx    = result["signal_idx"]    # (num_iter, S)
    signal_params = result["signal_params"] # (num_iter * S, 25)
    labels        = result["labels"]        # (N,)

    num_iter, S = signal_idx.shape
    signal_mask = labels == 1.0

    aligned = []
    for k in range(num_iter):
        batch_params = signal_params[k * S : (k + 1) * S]
        batch_idx    = signal_idx[k]
        aligned.append(batch_params[np.argsort(batch_idx)])
    aligned_params = np.concatenate(aligned, axis=0)  # (S_total, 25)

    N = len(labels)
    full_params = np.full((N, len(_PARAM_NAMES)), np.nan)
    full_params[signal_mask] = aligned_params

    return {name: full_params[:, i] for i, name in enumerate(_PARAM_NAMES)}


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Run one validation pass for a checkpoint.")
    p.add_argument("--epoch",       type=int, required=True, help="Epoch number to load")
    p.add_argument("--num-samples", type=int, required=True, help="Total samples (rounded up to nearest batch)")
    p.add_argument("--save",        type=str, default=None,  help="HDF5 path to save results (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_checkpoint_validation(
        epoch=args.epoch,
        num_samples=args.num_samples,
        save_path=args.save,
        verbose=True,
    )
    print(f"\nDone.  {result['num_samples']:,} samples processed.")
    sig_mask = result["labels"] == 1.0
    print(f"  Signals    : {sig_mask.sum():,}")
    print(f"  Noise      : {(~sig_mask).sum():,}")
    print(f"  Ranking stat range  : [{result['ranking_stat'].min():.3f}, {result['ranking_stat'].max():.3f}]")

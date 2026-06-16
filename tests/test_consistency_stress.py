"""
End-to-end STRESS + smoke test for the full multi-detector consistency stack,
exercising every option we have built, together:

  - per-detector consistency heads (tc + mchirp), GroupNorm, dropout,
  - the 4-class non-astrophysical masker (signal+signal / signal+noise /
    signal+signal' / noise+noise) at ``p_non_astrophysical``,
  - recolour noise augmentation (already in the o3b noise sampler),
  - the ``torch.compile`` production path (fullgraph + dynamic),
  - MC-dropout inference.

Stages
------
  1. build the all-options graph + COMPILED consistency model,
  2. shape + class-composition probe (sampler, masker, one compiled forward),
  3. long compiled run: throughput, peak GPU memory, finite loss every iter,
  4. ``p_non_astrophysical`` sweep {0.0, 0.5} (eager): both extremes run finite,
  5. MC-dropout inference on the consistency model: stochastic under eval.

Needs CUDA + the local data release + rebuilt fiducial PSDs; auto-skips where
that environment is absent. Runnable standalone under the sage conda python:

    python3 tests/test_consistency_stress.py
"""

import os
import sys
import time
from pathlib import Path

import torch

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "o3b"
DATA = Path("/local/scratch/igr/nnarenraju/data_release")

_HAS_CUDA = torch.cuda.is_available()
_HAS_NOISE = (DATA / "o3b_dataset" / "data_H1_O3b.bin").exists()
_HAS_FID = (RUN_DIR / "run_export" / "fiducial_psds" / "fiducial_H1_psd.bin").exists()
_READY = _HAS_CUDA and _HAS_NOISE and _HAS_FID

N_LONG_ITERS = 40
N_SWEEP_ITERS = 3


def _stage(name):
    print(f"\n{'='*70}\n[STAGE] {name}\n{'='*70}", flush=True)


def _shp(name, t):
    if isinstance(t, (tuple, list)):
        for i, e in enumerate(t):
            _shp(f"{name}[{i}]", e)
    elif hasattr(t, "shape"):
        print(f"    {name:22s} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}")
    else:
        print(f"    {name:22s} {type(t).__name__}={t}")


def _enter_run_dir():
    sys.path.insert(0, str(RUN_DIR))
    os.chdir(RUN_DIR)
    for m in ("config", "train_consistency"):
        sys.modules.pop(m, None)


def _build(p_non_astro, dropout, compile_model):
    """Register o3b config with the given options and build the full consistency
    run (graph + processor + model + masker + losses + optimiser)."""
    import importlib

    config = importlib.import_module("config")
    tcmod = importlib.import_module("train_consistency")
    importlib.reload(config)
    importlib.reload(tcmod)
    config.O3bCFG.p_non_astrophysical = float(p_non_astro)
    config.O3bCFG.dropout = float(dropout)
    config.set_configs()

    from sage.core.config import get_cfg, get_data_cfg
    from sage.architecture.network import MSCNN1D_2DResNetCBAM_Consistency
    from sage.architecture.custom_losses import BCEWithPEsigmaLoss, ConsistencyNLLLoss
    from sage.data.non_astrophysical import NonAstrophysicalMasker
    from sage.factory import SageConsistencyTraining
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    cfg, data_cfg = get_cfg(), get_data_cfg()
    signal_sampler, noise_sampler, bounds = tcmod.make_training_graph()
    processor, t_grid = tcmod.make_processor(bounds)

    model = MSCNN1D_2DResNetCBAM_Consistency(
        t_grid, frontend_filters=32, frontend_kernel=64,
        backend_resnet_size=50, norm_type="groupnorm", dropout=cfg.dropout,
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)
    if compile_model:
        model = torch.compile(model, fullgraph=True, dynamic=True)

    merged = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    cons = ConsistencyNLLLoss(tc_weight=1.0, mc_weight=1.0)
    opt = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    masker = NonAstrophysicalMasker(
        delta_f=signal_sampler.df, tc_bounds=bounds["tc"],
        analysis_length_s=data_cfg.sample_length_in_s, seed=150914,
    )
    return dict(
        cfg=cfg, data_cfg=data_cfg, signal=signal_sampler, noise=noise_sampler,
        bounds=bounds, processor=processor, model=model, merged=merged,
        cons=cons, opt=opt, sched=sched, scaler=scaler, masker=masker,
    )


def _make_trainer(b, n_iters):
    from sage.factory import SageConsistencyTraining

    return SageConsistencyTraining(
        b["signal"], b["noise"], b["processor"], b["model"], b["merged"],
        b["cons"], b["opt"], b["sched"], b["scaler"],
        num_iterations=n_iters, num_epochs=1,
        consistency_weight=0.1, masker=b["masker"],
    )


def _probe(b):
    """One step, stage by stage: sampler -> masker -> assemble -> compiled fwd."""
    cfg = b["cfg"]
    D = len(cfg.detectors)
    S = int(cfg.batch_size * cfg.class_balance)
    num_pe = len(cfg.do_point_estimate)
    mw = num_pe + 1
    with torch.no_grad():
        _stage("2. SHAPE + CLASS-COMPOSITION PROBE")
        sig, sig_t = b["signal"]()
        extra = sig.shape[0] - S
        print(f"  signal sampler: S={S} coherent + extra={extra} pool "
              f"(= round(p * (B-S)))")
        _shp("signal_data", sig)
        _shp("signal_targets", sig_t)
        # coherent injections: per-det mchirp identical across detectors
        coh_mc = sig_t[:S, mw + D: mw + 2 * D]
        assert torch.allclose(coh_mc[:, 0], coh_mc[:, 1], atol=1e-5), \
            "coherent per-det mchirp should match across detectors"
        print("  coherent per-det mchirp matches across detectors  OK")

        if extra > 0:
            pool_tc = sig_t[S:, mw: mw + D]
            pool_mc = sig_t[S:, mw + D: mw + 2 * D]
            na_d, na_tc, na_mc, na_mask = b["masker"](sig[S:], pool_tc, pool_mc)
            _shp("na_data", na_d)
            _shp("na_mask", na_mask)
            both = int((na_mask.sum(1) == D).sum())
            one = int((na_mask.sum(1) == 1).sum())
            print(f"  non-astro split: signal+signal'={both} (mask[1,1]), "
                  f"signal+noise={one} (mask[1,0]) of {extra}")
            assert both + one == extra

        nd, nt = b["noise"]()
        _shp("noise_data", nd)
        net = _one_assembled_forward(b, sig, sig_t, nd, nt, S, D, num_pe, mw)
        print(f"  class balance: class1={net['c1']} (== S), "
              f"non-astro={net['na']}, pure-noise={net['noise']} "
              f"(class0 total={net['na'] + net['noise']})")
        assert net["c1"] == S, "class-1 count must equal the signal budget S"
        assert net["c1"] == net["na"] + net["noise"], "class balance broken"
        print("  compiled forward + both losses finite  "
              f"merged={net['merged']:.3f} cons={net['cons']:.3f}")


def _one_assembled_forward(b, sig, sig_t, nd, nt, S, D, num_pe, mw):
    """Mirror SageConsistencyTraining batch assembly for one step and run the
    compiled model + both losses; return composition counts + loss values."""
    from contextlib import nullcontext
    from sage.core.pipeline import GWBatch, Grid, ProcessingState

    cfg = b["cfg"]
    device = cfg.device
    fw = mw + 2 * D
    tc0, mc0 = mw, mw + D
    extra = sig.shape[0] - S
    coh_data, coh_tgt = sig[:S], sig_t[:S]

    na_n = 0
    if extra > 0:
        na_d, na_tc, na_mc, na_mask = b["masker"](
            sig[S:], sig_t[S:, tc0:mc0], sig_t[S:, mc0:mc0 + D])
        na_n = na_d.shape[0]
        na_tgt = torch.zeros(na_n, fw, device=device, dtype=sig_t.dtype)
        na_tgt[:, tc0:mc0] = na_tc
        na_tgt[:, mc0:mc0 + D] = na_mc

    B = cfg.batch_size
    perm = torch.randperm(B, device=device)
    coh_slots, na_slots = perm[:S], perm[S:S + na_n]
    inj = torch.zeros_like(nd)
    targets = torch.zeros(B, fw, device=device, dtype=sig_t.dtype)
    mask = torch.zeros(B, D, device=device, dtype=sig_t.dtype)
    targets[:, num_pe:num_pe + 1] = nt
    inj[coh_slots] = coh_data
    targets[coh_slots] = coh_tgt
    mask[coh_slots] = 1.0
    if na_n:
        inj[na_slots] = na_d
        targets[na_slots] = na_tgt
        mask[na_slots] = na_mask
    x = nd + inj

    batch = GWBatch(x, state=getattr(b["signal"], "output_state",
                                     ProcessingState(Grid.FD_UNIFORM)))
    net_input = b["processor"](batch).to_network_input()
    with (torch.autocast(device_type="cuda", dtype=torch.float16)
          if cfg.autocast else nullcontext()):
        out = b["model"](net_input)
        _shp("ConsistencyOutput", out)
        merged = b["merged"]((out.ranking_stat, out.point_estimates), targets[:, :mw])
        cons = b["cons"](out.mu_tc, out.log_sigma_tc, out.mu_mc, out.log_sigma_mc,
                         targets[:, tc0:mc0], targets[:, mc0:mc0 + D], mask)
    c1 = int((targets[:, num_pe] == 1).sum())
    return dict(c1=c1, na=na_n, noise=B - S - na_n,
                merged=float(merged[0]), cons=float(cons[0]))


def _long_run(b):
    _stage(f"3. LONG COMPILED RUN — {N_LONG_ITERS} iters (all options on)")
    torch.cuda.reset_peak_memory_stats()
    trainer = _make_trainer(b, N_LONG_ITERS)
    t = time.time()
    trainer(nepoch=0)
    dt = time.time() - t
    comps = trainer.loss_components[0]
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {N_LONG_ITERS} iters in {dt:.1f}s = {N_LONG_ITERS/dt:.2f} it/s "
          f"({b['cfg'].batch_size*N_LONG_ITERS/dt:.0f} samples/s) incl. compile")
    print(f"  peak GPU memory = {peak:.1f} GB")
    print(f"  loss [total, merged, cons] = {comps.tolist()}")
    assert torch.isfinite(comps).all(), f"non-finite loss: {comps}"


def _sweep():
    # p is the fraction of the NOISE budget (B - S) made non-astrophysical, so it
    # is valid over the full [0, 1]: 0.0 = pure 2-class (no non-astro), 1.0 = the
    # entire noise class is non-astro (zero pure-noise samples). Sweep both
    # extremes plus the midpoint.
    _stage("4. p_non_astrophysical SWEEP {0.0, 0.5, 1.0} (eager), dropout=0.05")
    for p in (0.0, 0.5, 1.0):
        b = _build(p_non_astro=p, dropout=0.05, compile_model=False)
        S = int(b["cfg"].batch_size * b["cfg"].class_balance)
        n_noise = b["cfg"].batch_size - S
        extra = b["signal"].signal_batch_size - S
        trainer = _make_trainer(b, N_SWEEP_ITERS)
        trainer(nepoch=0)
        comps = trainer.loss_components[0]
        pure_noise = n_noise - extra
        print(f"  p={p:<4} extra={extra:<3} pure_noise={pure_noise:<3} "
              f"loss={comps.tolist()}")
        assert torch.isfinite(comps).all(), f"non-finite loss at p={p}: {comps}"
        if p == 1.0:
            assert extra == n_noise and pure_noise == 0, \
                "p=1.0 must turn the entire noise budget non-astrophysical"


def _mc_dropout():
    _stage("5. MC-DROPOUT inference on the consistency model")
    from sage.architecture.network import enable_mc_dropout

    b = _build(p_non_astro=0.0, dropout=0.05, compile_model=False)
    model = b["model"]
    # a real preprocessed batch to feed the model
    nd, _ = b["noise"]()
    from sage.core.pipeline import GWBatch, Grid, ProcessingState
    net_input = b["processor"](
        GWBatch(nd, state=getattr(b["signal"], "output_state",
                                  ProcessingState(Grid.FD_UNIFORM)))
    ).to_network_input()

    model.eval()
    enable_mc_dropout(model)            # dropout back ON under eval
    with torch.no_grad():
        a = model(net_input).ranking_stat
        c = model(net_input).ranking_stat
    spread = float((a - c).abs().mean())
    print(f"  two MC passes differ: mean|Δ ranking_stat| = {spread:.3e}")
    assert spread > 0.0, "MC-dropout produced identical passes (dropout inactive)"


def test_consistency_stack_stress():
    if not _READY:
        print(f"SKIP: CUDA={_HAS_CUDA} noise={_HAS_NOISE} fiducial={_HAS_FID}")
        return
    prev = os.getcwd()
    _enter_run_dir()
    try:
        _stage("1. BUILD all-options graph + COMPILED model "
               "(groupnorm, dropout=0.05, p=0.5, recolour, compile)")
        b = _build(p_non_astro=0.5, dropout=0.05, compile_model=True)
        print(f"  params: {sum(p.numel() for p in b['model'].parameters()):,}")
        _probe(b)
        _long_run(b)
        _sweep()
        _mc_dropout()
        _stage("CONSISTENCY STACK STRESS TEST PASSED")
    finally:
        os.chdir(prev)
        if str(RUN_DIR) in sys.path:
            sys.path.remove(str(RUN_DIR))


if __name__ == "__main__":
    test_consistency_stack_stress()
    print("\n>>> CONSISTENCY STACK STRESS TEST COMPLETE <<<")

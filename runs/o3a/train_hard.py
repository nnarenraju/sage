#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O3a BBH hard negative mining training.

Pipeline
--------
Signal  : IMRPhenomPv2, time-domain multirate sampling
Noise   : MemmapNoiseSampler (O3a H1 + L1)
          + RecolourPostprocess: O3a -> O3b spectral bridge
          → hard start-times are pushed in by the mining callback (below)
Preproc : FiducialWhitening → MultirateSampler (TD_MULTIRATE)
Network : MSCNN1D_2DResNetCBAM_HardMining (heteroscedastic + frontend-embed tap, torch.compiled)
Loss    : BCEWithPEsigmaLoss

Hard-negative mining
--------------------
On scheduled epochs (``mine_schedule``), CMA-MAE (pyribs) mines per-detector
start times for hard noise windows, keeping diverse hard windows (diversity
measured in the model's **frontend** embedding). Hard windows are persisted to an
HDF5 bank and the currently-above-threshold start-times are pushed into the noise
sampler via ``set_hard_bank``, which replays them with probability
``hard_bias_prob``. Mining is a callback on plain SageVanillaTraining -- see
:class:`sage.factory.HardMiningCallback`.

Usage
-----
    python train_hard.py
    # or:
    # sbatch start.sh train_hard.py
"""

import os
import sys
from pathlib import Path

import torch

torch._dynamo.config.verbose = False
torch._inductor.config.debug = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch._dynamo.reset()

RUN_DIR  = os.path.dirname(os.path.abspath(__file__))
SAGE_DIR = os.path.join(RUN_DIR, "..", "..")
sys.path.insert(0, RUN_DIR)
sys.path.insert(0, SAGE_DIR)
os.chdir(RUN_DIR)

import os as _os, importlib as _il
# Config module to run: named by SAGE_CONFIG (default "config"). Make a
# per-network config with `cp config.py config_<DETS>.py`, edit `detectors`
# + `export_dir`, and pick it at launch. No network logic lives in code.
set_configs = _il.import_module(_os.environ.get("SAGE_CONFIG", "config")).set_configs
from sage.core.config import get_cfg, get_data_cfg
from sage.utils.servers import get_server

# Signal
from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
from sage.data.waveform import HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler

# Noise
from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess

# Preprocessing
from sage.core.graph import Preprocessor
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning

# Model and loss
from sage.architecture.network import MSCNN1D_2DResNetCBAM_HardMining
from sage.architecture.custom_losses import BCEWithPEsigmaLoss
from sage.core.logger import (
    HDF5LossLogger,
    setup_logging,
    get_logger,
)

logger = get_logger("sage.run.o3a.hard")

# Optimiser and scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Training / validation / hard mining
from sage.factory.training import SageVanillaTraining
from sage.factory.validation import SageVanillaValidation
from sage.factory.callbacks import HardMiningCallback, EMACallback
from sage.utils.checkpoint import CheckpointManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

# Seeds (module-level so run_hard AND calibrate_ema share one source of truth):
# a fresh run uses BASE_SEED; a run resuming at epoch K uses BASE_SEED +
# SEED_STRIDE*K, so the sampler RNG streams never replay their pre-crash draws.
BASE_SEED   = 150914
SEED_STRIDE = 1_000_003


def make_training_graph(seed):
    # `seed` is resume-aware (derived from the resume epoch in run_hard) so the
    # noise/parameter/SNR RNGs never replay their pre-crash draws on restart.
    param_sampler  = read_from_config("./gwconfig.yaml", seed=seed)
    snrscaler      = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=seed))
    signal_sampler = IMRPhenomPv2(
        param_sampler, ConstantProjection(), augment=snrscaler,
    )

    recolour = RecolourPostprocess(
        p_recolour          = 0.37,
        recolour_dataset_dir= get_server().dataset_dir("O3b"),   # O3a -> O3b bridge
        # data-driven k*sigma(f) PSD augmenter; config-overridable (default 0.5 = on)
        dr_gain             = getattr(get_cfg(), "recolour_dr_gain", 0.5),
        seed                = seed + 7,   # resume-aware, distinct from other streams
    )
    noise_sampler = MemmapNoiseSampler(
        postprocess_fn   = recolour,
        prefetch         = 8,
        seed             = seed,
        training         = True,
        # Hard biasing is driven entirely by HardMiningCallback via the HDF5
        # bank (set_hard_bank); no .npz pre-loading and no bias until the first
        # mining round writes some hard start-times.
        hard_bias_prob   = 0.0,
    )
    return signal_sampler, noise_sampler, param_sampler.bounds


def make_validation_graph():
    param_sampler  = read_from_config("./gwconfig.yaml", seed=170817)
    snrscaler      = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=170817))
    signal_sampler = IMRPhenomPv2(
        param_sampler, ConstantProjection(), augment=snrscaler,
    )
    noise_sampler  = MemmapNoiseSampler(
        postprocess_fn = None, prefetch=8, seed=170817, training=False,
    )
    return signal_sampler, noise_sampler


def make_processor(bounds):
    whitener = FiducialWhitening()
    mrsampler = MultirateSampler(binning_method=DyadicPyramidBinning(bounds))
    return Preprocessor([whitener, mrsampler])


def calibrate_ema():
    """SEPARATE post-training step (never in the hot path): recalibrate BatchNorm
    for the averaged EMA weights (official ``torch.optim.swa_utils.update_bn``),
    then compare the calibrated EMA against best.pt on validation and write a note
    -- keeping ALL saved weights. Run after the 128-epoch training completes:
        ./submit.sh calibrate config_HL
    """
    from sage.factory.ema_calibration import calibrate_and_compare

    set_configs()
    cfg, data_cfg = get_cfg(), get_data_cfg()

    # Training-distribution graph (recoloured noise + signals; no hard bias ->
    # the base training distribution) for BN recalibration. The validation graph
    # is rebuilt per-model so ema and best see the SAME seeded batches.
    bn_sig, bn_noise, bounds = make_training_graph(seed=BASE_SEED)
    processor = make_processor(bounds)

    model = MSCNN1D_2DResNetCBAM_HardMining(
        frontend_filters=32, frontend_kernel=64,
        backend_resnet_size=50, norm_type="groupnorm", dropout=cfg.dropout,
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    loss_function = BCEWithPEsigmaLoss(
        regression_weight=0.005, coupling_weight=0.005, beta=0.5,
    )
    use_bf16  = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    calibrate_and_compare(
        cfg, os.path.join(cfg.export_dir, "CHECKPOINTS"), model,
        bn_signal=bn_sig, bn_noise=bn_noise,
        make_val_graph=make_validation_graph,
        processor=processor, loss_fn=loss_function,
        bn_batches=getattr(cfg, "bn_calib_batches", 500),
        val_iters=getattr(cfg, "calib_val_iters", cfg.validation_iterations),
        amp_dtype=amp_dtype,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_hard():

    set_configs()
    cfg, data_cfg = get_cfg(), get_data_cfg()

    # Logging: console + <export_dir>/logs/run.log. After set_configs, since
    # that is what tells us where this run writes. SAGE_LOG_LEVEL=DEBUG for the
    # verbose console format.
    log_path = setup_logging(cfg.export_dir)
    logger.info("Run: %s | detectors %s", cfg.export_dir, cfg.detectors)
    logger.info("Logging to %s", log_path)

    os.makedirs(cfg.export_dir, exist_ok=True)

    # ── Resume-aware seeding ─────────────────────────────────────────────────
    # Peek the resume epoch BEFORE building the samplers (does NOT load the full
    # checkpoint) so their RNGs are seeded resume-aware: a fresh run uses
    # BASE_SEED; a run resuming at epoch K uses a distinct, deterministic seed,
    # so the noise/parameter/SNR streams never replay the pre-crash draws. The
    # model/optimiser/scheduler + global RNG and the mining bank are restored
    # *exactly* further down -- only the sampler streams advance. BASE_SEED and
    # SEED_STRIDE are module-level constants (shared with calibrate_ema).
    start_epoch  = CheckpointManager.peek_next_epoch(cfg.export_dir)
    sampler_seed = BASE_SEED + SEED_STRIDE * start_epoch
    print(f"[train_hard] start_epoch={start_epoch}  sampler_seed={sampler_seed}",
          flush=True)

    # Seed the GLOBAL torch RNG (resume-aware) so the draws that use it -- sky
    # orientation/GMST (project.py) and the injection-slot randperm
    # (training.py) -- plus one-off model init are reproducible on a fresh run
    # like every other stream. On resume, load_latest() restores the checkpointed
    # global RNG below, so these streams continue forward and never replay.
    torch.manual_seed(sampler_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampler_seed)

    # ── Graphs ──────────────────────────────────────────────────────────────
    tr_sig, tr_noise, bounds = make_training_graph(seed=sampler_seed)
    val_sig, val_noise       = make_validation_graph()
    processor                = make_processor(bounds)

    # ── Model (compiled for max throughput) ──────────────────────────────────
    model = MSCNN1D_2DResNetCBAM_HardMining(
        frontend_filters    = 32,
        frontend_kernel     = 64,
        backend_resnet_size = 50,
        norm_type           = "groupnorm",   # joint over detectors -> preserves the
                                              # inter-detector amplitude (SNR/sky info)
                                              # InstanceNorm would erase per-window.
        dropout             = cfg.dropout,    # wired; config sets 0.0 (off)
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    base_model = model   # uncompiled handle for the weight-EMA (shares params)
    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print("Model compiled with torch.compile!")

    # ── Optimisation ─────────────────────────────────────────────────────────
    loss_function = BCEWithPEsigmaLoss(
        regression_weight=0.005, coupling_weight=0.005,
        beta=0.5,   # beta-NLL (Seitzer et al. 2022): temper the ~1/sigma^2 grad
    )
    # No weight-decay on 1-D params (BN/GN scales+shifts, biases) -- He et al.
    # "Bag of Tricks" CVPR 2019: decaying normalisation affine params just shrinks
    # the normalisation and hurts. Decay only >=2-D weights (conv/linear).
    _decay, _no_decay = [], []
    for _n, _p in base_model.named_parameters():
        if _p.requires_grad:
            (_decay if _p.ndim >= 2 else _no_decay).append(_p)
    optimiser = optim.AdamW(
        [{"params": _decay, "weight_decay": 1e-4},   # conventional mild AdamW wd (weights only)
         {"params": _no_decay, "weight_decay": 0.0}],
        lr=2e-4, fused=True,
    )
    # Single cosine decay with a short linear warmup, stepped per batch over the
    # whole run. Best-single-model schedule: "decay, not restarts" drives the
    # gains (Gotmare et al. ICLR 2019), and the run ends at eta_min so the final
    # checkpoint IS the fully-annealed model (no wasted post-restart epochs).
    # Warmup stabilises early Adam + the heteroscedastic head (Kalra&Barkeshli 2024).
    total_steps  = cfg.num_epochs * cfg.training_iterations
    warmup_steps = min(getattr(cfg, "warmup_steps", 5000), max(1, total_steps // 2))
    scheduler = SequentialLR(
        optimiser,
        schedulers=[
            LinearLR(optimiser, start_factor=1e-3, total_iters=warmup_steps),
            CosineAnnealingLR(optimiser, T_max=max(1, total_steps - warmup_steps),
                              eta_min=1e-6),
        ],
        milestones=[warmup_steps],
    )
    # AMP dtype: bf16 on GPUs that support it (H100 etc.) — full fp32 exponent
    # range, so NO GradScaler is needed and there's no fp16 underflow; fall back
    # to fp16 + GradScaler on older GPUs (Micikevicius et al. ICLR 2018).
    use_bf16      = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype     = torch.bfloat16 if use_bf16 else torch.float16
    scaler        = torch.amp.GradScaler(cfg.device,
                                         enabled=(cfg.autocast and not use_bf16))

    # ── Training (vanilla + hard-mining callback) + validation ───────────────
    # Hard mining is a callback on plain SageVanillaTraining. Drop the
    # callbacks=[...] to disable mining entirely (no pyribs, random noise only).
    # Front-load mining through the first ~60% of the run to build a large (~1M)
    # hard-noise bank by the midpoint, then stop -- the persisted bank keeps
    # biasing the second half. Bounding the number of mine events (each re-scores
    # the whole growing bank) keeps the total walltime under the 4-day cap.
    _mine_stop = int(round(0.6 * cfg.num_epochs))
    _front_load_sched = list(range(3, _mine_stop, 4))
    hard_cb = HardMiningCallback(
                # File-resident HDF5 bank lives on /work; one file per
                # (train_runs, detectors) -- see hardbank_<runs>_<dets>.h5.
                bank_dir       = getattr(cfg, "bank_dir",
                                         os.path.join(get_server().data_root, "hard_mining")),
                # One arg, two forms: int N -> mine every N epochs; or a list of
                # epoch indices. Default = the front-loaded list above (every 4
                # epochs over the first ~60%); a 2-epoch smoke overrides with [0].
                mine_schedule  = getattr(cfg, "mine_schedule", _front_load_sched),
                # Hard-noise bias is ANNEALED with the active-bank size (see
                # HardMiningCallback._bias_for): ramps 0 -> hard_bias_prob so a few
                # thousand early hard windows aren't replayed hundreds of times per
                # epoch (variance collapse). hard_bias_prob is the target/max;
                # bias_replays caps replays/window/epoch.
                hard_bias_prob = getattr(cfg, "hard_bias_prob", 0.2),
                bias_replays_per_epoch = getattr(cfg, "bias_replays_per_epoch", 4.0),
                # Keep noise scoring >= logit 0.0 (~50% signal prob): any window
                # the model leans "signal" on counts as hard, not just confident
                # false positives -- a broad bar to accumulate a LARGE hard pool.
                # Use keep_threshold_sigmoided=<p> for the same bar as a prob (raw
                # wins if both given); raise toward logit ~2 (~88%) for confident
                # FPs only.
                keep_threshold_raw = getattr(cfg, "keep_threshold_raw", 0.0),
                # Volume: candidates/event = n_warmup + mine_iters*emitter_batch
                # (= 2048 + 6000*72 ~= 434k). With ~12% keep-and-distinct rate and
                # the front-loaded ~19-event schedule, targets ~1M by epoch ~72.
                # Verify kept/event via the [HardMining] log at the smoke and
                # retune mine_iters if the real rate differs.
                mine_iters         = getattr(cfg, "mine_iters", 6000),
                emitter_batch_size = getattr(cfg, "emitter_batch_size", 72),
                # Diversity (cover the ~22 H/L glitch families + V unknowns + tail
                # without collapse). Measure space = descriptor_dim-D PCA of the
                # model FRONTEND embedding; the CVT archive holds n_cells niches
                # (one elite each). novelty_dist gates the embedding memory (>= this
                # cosine apart); novelty_weight steers search toward uncovered
                # families. These are reasonable estimates -- retune from archive
                # occupancy / kept-per-event at the smoke.
                descriptor_dim = getattr(cfg, "descriptor_dim", 8),
                n_cells        = getattr(cfg, "n_cells", 2048),
                novelty_dist   = getattr(cfg, "novelty_dist", 0.15),
                novelty_weight = getattr(cfg, "novelty_weight", 1.5),
                max_embeddings = getattr(cfg, "max_embeddings", 50_000),
                mine_seed      = 150914,
    )
    # Per-step weight EMA -> ema.pt (recommended single model for eval/deploy).
    ema_cb = EMACallback(
        base_model,
        decay        = float(getattr(cfg, "ema_decay", 0.9998)),
        save_path    = os.path.join(cfg.export_dir, "CHECKPOINTS", "ema.pt"),
        resume       = start_epoch > 0,
        map_location = cfg.device,
    )
    trainer = SageVanillaTraining(
        tr_sig, tr_noise, processor, model, loss_function,
        optimiser, scheduler, scaler,
        num_iterations = cfg.training_iterations,
        num_epochs     = cfg.num_epochs,
        callbacks      = [hard_cb, ema_cb],
        # Step the warmup+cosine schedule once per batch across all total_steps.
        scheduler_mode = "batch",
        amp_dtype      = amp_dtype,
    )

    vanilla_val = SageVanillaValidation(
        val_sig, val_noise, processor, model, loss_function,
        num_iterations = cfg.validation_iterations,
        num_epochs     = cfg.num_epochs,
        amp_dtype      = amp_dtype,
    )

    # ── Checkpoint + logger ──────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        cfg=cfg, data_cfg=data_cfg, model=model,
        optimizer=optimiser, scheduler=scheduler, scaler=scaler,
    )
    loss_logger = HDF5LossLogger(
        path           = os.path.join(cfg.export_dir, "losses.h5"),
        num_epochs     = cfg.num_epochs,
        num_components = loss_function.num_components,
    )

    # ── Restore model / optimiser / scheduler / global-RNG + miner (if resuming)
    # Exact continuation of everything except the sampler streams (seeded above).
    if start_epoch > 0:
        loaded = ckpt_mgr.load_latest(map_location=cfg.device)
        if loaded != start_epoch:
            # progress.json can lag latest.pt by one epoch if a crash landed
            # between their two writes; latest.pt is authoritative for the
            # restored state, so continue from it. (The samplers were seeded from
            # the peeked epoch, which is still a valid non-replaying seed.)
            print(f"[resume] progress.json epoch {start_epoch} != latest.pt "
                  f"{loaded}; continuing from latest.pt", flush=True)
            start_epoch = loaded
        print(f"Resuming from epoch {start_epoch}")
        # Re-bias the (freshly-built) sampler from the persisted hard bank so we
        # don't train on purely random noise until the next scheduled mine epoch.
        hard_cb.attach_for_resume(trainer)

    # ── Epoch loop (train + mine, validate every 5) ───────────────────────────
    for epoch in range(start_epoch, cfg.num_epochs):
        trainer(nepoch=epoch)
        loss_logger.log(trainer.loss_components, epoch, split="training")
        val_loss = None
        if epoch % 5 == 0 or epoch == cfg.num_epochs - 1:
            vanilla_val(nepoch=epoch)
            loss_logger.log(vanilla_val.loss_components, epoch, split="validation")
            val_loss = float(vanilla_val.loss_components[epoch][0].item())
        # best.pt tracks the lowest VALIDATION loss (persists across resume);
        # latest.pt + per-epoch epoch_{N}.pt are written every epoch (the
        # learning-evolution history). save() always writes latest.pt.
        ckpt_mgr.save(epoch=epoch, val_loss=val_loss)


if __name__ == "__main__":
    run_hard()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O3b BBH hard negative mining training.

Pipeline
--------
Signal  : IMRPhenomPv2, time-domain multirate sampling
Noise   : MemmapNoiseSampler (O3b H1 + L1) with hard_dataset_dir + hard_bias_prob
          + optional RecolourPostprocess for O3a → O3b spectral bridge
          → hard/random mixing is automatic inside the sampler
Preproc : FiducialWhitening → MultirateSampler (TD_MULTIRATE)
Network : MSCNN1D_2DResNetCBAM_Heteroscedastic (torch.compiled)
Loss    : BCEWithPEsigmaLoss

Mining schedule
---------------
Every mine_explore_every epochs: CMAMEMiner
    GPS explorer — finds new GPS time regions in O3b data.
    70% budget = random exploration, 30% = re-validate known hard GPS positions.

Every mine_refine_every epochs: CMAMEGAMiner
    Pattern refiner — re-evaluates bank templates, gradient-refines stale ones,
    then mines millions more using the updated bank as a pre-filter.

Both miners share a SharedHardNoiseBank that grows across all runs.

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

from config import set_configs
from sage.core.config import get_cfg, get_data_cfg

# Signal
from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
from sage.data.waveform import HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler

# Noise
from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess
from sage.data.noise import SharedHardNoiseBank, CMAMEMiner, CMAMEGAMiner

# Preprocessing
from sage.core.graph import Preprocessor
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning

# Model and loss
from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic
from sage.architecture.custom_losses import BCEWithPEsigmaLoss
from sage.core.logger import HDF5LossLogger

# Optimiser and scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Training / validation / hard mining
from sage.factory.training import SageVanillaTraining
from sage.factory.validation import SageVanillaValidation
from sage.factory.hard_mining import SageHardMiningTraining
from sage.factory.miner_schedules import LinearThresholdSchedule
from sage.utils.checkpoint import CheckpointManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_DIR = Path("./hard_noise_datasets/")
BANK_PATH   = DATASET_DIR / "bank.npz"


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def make_training_graph():
    param_sampler  = read_from_config("./gwconfig.yaml", seed=150914)
    snrscaler      = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
    signal_sampler = IMRPhenomPv2(
        param_sampler, ConstantProjection(), augment=snrscaler,
    )

    recolour = RecolourPostprocess(
        p_recolour          = 0.37,
        recolour_dataset_dir= "/work/nagarajan/sage/o3a",
    )
    noise_sampler = MemmapNoiseSampler(
        postprocess_fn   = recolour,
        prefetch         = 8,
        seed             = 150914,
        training         = True,
        hard_dataset_dir = str(DATASET_DIR),
        hard_bias_prob   = 0.6,
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


def make_miners(K: int = 32):
    """Build explorer and refiner miners."""

    explorer = CMAMEMiner(
        n_svd_components = K,
        n_init_batches   = 400,
        n_iterations     = 10_000,
        batch_size       = 128,
        sigma_g          = 0.5,
        explore_fraction = 0.7,
        threshold        = 3.0,         # initial; overridden each epoch by schedule
        max_samples      = 10_000_000,
        grid_size        = 32,
        autocast         = True,
    )

    refiner = CMAMEGAMiner(
        n_svd_components  = K,
        n_init_batches    = 400,
        n_iterations      = 30_000,
        scan_batch_size   = 512,
        model_batch_size  = 128,
        exploit_fraction  = 0.3,
        pure_explore_frac = 0.1,
        sigma_g_gps       = 0.5,
        svd_distance_pct  = 10.0,
        refine_lr         = 1.0,
        n_rescore         = 10_000,
        threshold         = 5.0,        # initial; overridden each epoch by schedule
        target_samples    = 5_000_000,
        max_samples       = 10_000_000,
        max_bank_size     = 500_000,
        autocast          = True,
    )

    return explorer, refiner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_hard():

    set_configs()
    cfg, data_cfg = get_cfg(), get_data_cfg()

    os.makedirs(cfg.export_dir, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # ── Graphs ──────────────────────────────────────────────────────────────
    tr_sig, tr_noise, bounds = make_training_graph()
    val_sig, val_noise       = make_validation_graph()
    processor                = make_processor(bounds)

    # ── Model (compiled for max throughput) ──────────────────────────────────
    model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
        frontend_filters    = 32,
        frontend_kernel     = 64,
        backend_resnet_size = 50,
        norm_type           = "instancenorm",
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print("Model compiled with torch.compile!")

    # ── Optimisation ─────────────────────────────────────────────────────────
    loss_function = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    optimiser     = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    scheduler     = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
    scaler        = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    # ── Training + validation loops ──────────────────────────────────────────
    vanilla_train = SageVanillaTraining(
        tr_sig, tr_noise, processor, model, loss_function,
        optimiser, scheduler, scaler,
        num_iterations = cfg.training_iterations,
        num_epochs     = cfg.num_epochs,
    )

    vanilla_val = SageVanillaValidation(
        val_sig, val_noise, processor, model, loss_function,
        num_iterations = cfg.validation_iterations,
        num_epochs     = cfg.num_epochs,
    )

    # ── Checkpoint + logger ──────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        cfg=cfg, data_cfg=data_cfg, model=model,
        optimizer=optimiser, scheduler=scheduler, scaler=scaler,
    )

    logger = HDF5LossLogger(
        path           = os.path.join(cfg.export_dir, "losses.h5"),
        num_epochs     = cfg.num_epochs,
        num_components = vanilla_train.loss_function.num_components,
    )

    # ── Shared bank + miners ─────────────────────────────────────────────────
    K = 32
    bank = (
        SharedHardNoiseBank.load(BANK_PATH)
        if BANK_PATH.exists()
        else SharedHardNoiseBank(K=K)
    )
    print(f"SharedHardNoiseBank: {len(bank):,} templates loaded")

    explorer, refiner = make_miners(K=K)

    # ── Hard negative mining ─────────────────────────────────────────────────
    warmup_epochs = 10
    hard_mining = SageHardMiningTraining(
        vanilla_training   = vanilla_train,
        vanilla_validation = vanilla_val,
        noise_sampler      = tr_noise,
        signal_sampler     = tr_sig,
        processor          = processor,
        model              = model,
        explorer           = explorer,
        refiner            = refiner,
        shared_bank        = bank,
        dataset_dir        = DATASET_DIR,
        bank_path          = BANK_PATH,
        total_epochs       = cfg.num_epochs,
        warmup_epochs      = warmup_epochs,
        mine_explore_every = 5,
        mine_refine_every  = 1,
        validate_every     = 5,
        logger             = logger,
        ckpt_mgr           = ckpt_mgr,
        threshold_schedule = LinearThresholdSchedule(cfg.num_epochs, warmup_epochs),
        accumulate         = True,
        max_total_samples  = 30_000_000,
    )

    start_epoch = 0
    if os.path.exists(ckpt_mgr.latest_path):
        start_epoch = ckpt_mgr.load_latest(map_location=cfg.device)
        print(f"Resuming from epoch {start_epoch}")

    hard_mining.run(start_epoch=start_epoch)


if __name__ == "__main__":
    run_hard()

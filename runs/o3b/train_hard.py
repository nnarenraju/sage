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

Hard-negative mining
--------------------
After each epoch's training (past ``warmup_epochs``), CMA-MAE (pyribs) mines
per-detector start times for hard noise windows, keeping diverse hard windows
(diversity measured in the model's own embedding).  Every window above
``keep_threshold`` is accumulated and replayed via the noise sampler with
probability ``hard_bias_prob``.  Mining is a callback on plain
SageVanillaTraining -- see :class:`sage.factory.HardMiningCallback`.

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
from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic
from sage.architecture.custom_losses import BCEWithPEsigmaLoss
from sage.core.logger import HDF5LossLogger

# Optimiser and scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Training / validation / hard mining
from sage.factory.training import SageVanillaTraining
from sage.factory.validation import SageVanillaValidation
from sage.factory.callbacks import HardMiningCallback
from sage.utils.checkpoint import CheckpointManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_DIR = Path("./hard_noise_datasets/")


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
        recolour_dataset_dir= get_server().dataset_dir("O3a"),
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

    # ── Training (vanilla + hard-mining callback) + validation ───────────────
    # Hard mining is a callback on plain SageVanillaTraining. Drop the
    # callbacks=[...] to disable mining entirely (no pyribs, random noise only).
    trainer = SageVanillaTraining(
        tr_sig, tr_noise, processor, model, loss_function,
        optimiser, scheduler, scaler,
        num_iterations = cfg.training_iterations,
        num_epochs     = cfg.num_epochs,
        callbacks      = [
            HardMiningCallback(
                hard_bias_prob    = 0.6,
                # Keep noise scoring >= logit 2.0 (~88% signal probability — a
                # confident false positive). Use keep_threshold_sigmoided=<p> to
                # set the same bar as a probability instead (raw wins if both are
                # given). logit 5.0 (~0.993) keeps almost nothing until the model
                # is well trained -- measured on a 1-epoch model the hardest mined
                # noise peaks near logit ~6, p99 ~3.
                keep_threshold_raw = 2.0,
                warmup_epochs     = 10,
                mine_iters        = 200,
                hard_dataset_dir  = str(DATASET_DIR),
                max_total_samples = 30_000_000,
                mine_seed         = 150914,
            ),
        ],
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
        num_components = loss_function.num_components,
    )

    # ── Epoch loop (train + mine, validate every 5) ───────────────────────────
    start_epoch = 0
    if os.path.exists(ckpt_mgr.latest_path):
        start_epoch = ckpt_mgr.load_latest(map_location=cfg.device)
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.num_epochs):
        trainer(nepoch=epoch)
        logger.log(trainer.loss_components, epoch, split="training")
        if epoch % 5 == 0 or epoch == cfg.num_epochs - 1:
            vanilla_val(nepoch=epoch)
            logger.log(vanilla_val.loss_components, epoch, split="validation")
        ckpt_mgr.save(epoch=epoch,
                      val_loss=float(trainer.loss_components[epoch][0].item()))


if __name__ == "__main__":
    run_hard()

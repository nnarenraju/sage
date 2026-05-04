#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
O3b hard-noise-mining training script.

Compared to the baseline O3b run, the only additions are:

  1. GlitchOversampledNoiseSampler — class-balanced oversampling of O3b
     glitch windows (from GravitySpy catalog) so rare-but-dangerous glitch
     classes are not drowned out by Scattered_Light.

  2. Hard noise replay — a periodic mining pass identifies the background
     windows the model most strongly false-alarms on and replays them at a
     controlled fraction of every training batch.

Everything else (loss function, model, optimiser, scheduler) is identical
to the baseline O3b run.

Run from the runs/o3b_hardmining directory:
    python3 train.py
"""

import os
import torch

torch._dynamo.config.verbose  = False
torch._inductor.config.debug  = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.cuda.empty_cache()
torch._dynamo.reset()

from sage.core.config import get_cfg, get_data_cfg

# Signal sampler
from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
from sage.data.waveform import HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler

# Noise sampler + hard mining
from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess
from sage.data.noise import GlitchOversampledNoiseSampler
from sage.data.noise.hard_mining import HardSampleBuffer, HardSampleMiner

# Preprocessing
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning

# Model and loss
from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic
from sage.architecture.custom_losses import BCEWithPEsigmaLoss
from sage.core.logger import HDF5LossLogger

# Optimiser / scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Graph + factories
from sage.core.graph import Preprocessor
from sage.factory.training import SageHardMiningTraining
from sage.factory.validation import SageUncompiledValidation
from sage.utils.checkpoint import CheckpointManager

from config import set_configs


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def make_training_graph():
    training_param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
    waveform_project       = ConstantProjection()
    target_snr_sampler     = HalfNorm(scale=4.0, loc=5.0, seed=150914)
    snrscaler              = OptimalSNRRescaler(target_snr_sampler)
    training_signal_sampler = IMRPhenomPv2(
        training_param_sampler, waveform_project, augment=snrscaler,
    )
    recolour = RecolourPostprocess(
        p_recolour=0.37,
        recolour_dataset_dir="/local/scratch/igr/nnarenraju/search/o3a",
    )
    # Class-balanced oversampling of high-SNR O3b glitches.
    # 10 % of each training batch is replaced by GPS-aligned O3b glitch
    # windows drawn with equal probability from every GravitySpy class.
    training_noise_sampler = GlitchOversampledNoiseSampler(
        postprocess_fn  = recolour,
        prefetch        = 8,
        seed            = 150914,
        catalog_files   = [
            (0, "/local/scratch/igr/nnarenraju/gwspy/H1_O3b.csv"),
            (1, "/local/scratch/igr/nnarenraju/gwspy/L1_O3b.csv"),
        ],
        min_snr         = 15.0,
        glitch_frac     = 0.10,
        class_balanced  = True,
    )
    return training_signal_sampler, training_noise_sampler, training_param_sampler.bounds


def make_validation_graph():
    validation_param_sampler = read_from_config("./gwconfig.yaml", seed=170817)
    waveform_project         = ConstantProjection()
    target_snr_sampler       = HalfNorm(scale=4.0, loc=5.0, seed=170817)
    snrscaler                = OptimalSNRRescaler(target_snr_sampler)
    validation_signal_sampler = IMRPhenomPv2(
        validation_param_sampler, waveform_project, augment=snrscaler,
    )
    validation_noise_sampler = MemmapNoiseSampler(
        postprocess_fn=None, prefetch=8, seed=170817, training=False,
    )
    return validation_signal_sampler, validation_noise_sampler


def make_processor(bounds):
    whitener      = FiducialWhitening()
    dyadic_binning = DyadicPyramidBinning(bounds)
    mrsampler      = MultirateSampler(binning_method=dyadic_binning)
    return Preprocessor([whitener, mrsampler])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sage():

    set_configs()
    cfg, data_cfg = get_cfg(), get_data_cfg()

    training_signal_sampler, training_noise_sampler, bounds = make_training_graph()
    validation_signal_sampler, validation_noise_sampler     = make_validation_graph()
    processor = make_processor(bounds)

    # ── Model ──────────────────────────────────────────────────────────
    model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="instancenorm",
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print("Model compiled with torch.compile.")

    # ── Loss (identical to baseline O3b) ───────────────────────────────
    loss_function = BCEWithPEsigmaLoss(
        regression_weight = 0.005,
        coupling_weight   = 0.005,
    )

    # ── Optimiser / scheduler ─────────────────────────────────────────
    optimiser = optim.Adam(
        model.parameters(), lr=2e-4, weight_decay=1e-6,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimiser, T_0=5, T_mult=2, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    # ── Hard noise buffer and miner ───────────────────────────────────
    # Capacity 2048: top-2048 hardest noise windows per mining pass.
    hard_noise_buffer = HardSampleBuffer(capacity=2048)

    # Mine 100K background windows per pass, keep hardest 2048.
    miner = HardSampleMiner(
        hard_noise_buffer = hard_noise_buffer,
        n_mine_noise      = 100_000,
    )

    # ── Training loop ─────────────────────────────────────────────────
    train_sage = SageHardMiningTraining(
        signal_sampler     = training_signal_sampler,
        noise_sampler      = training_noise_sampler,
        processor          = processor,
        model              = model,
        loss_function      = loss_function,
        optimiser          = optimiser,
        scheduler          = scheduler,
        scaler             = scaler,
        miner              = miner,
        hard_noise_buffer  = hard_noise_buffer,
        num_iterations     = cfg.training_iterations,
        num_epochs         = cfg.num_epochs,
        hard_noise_frac    = 0.15,
        mine_every_n_epochs = 5,
    )

    validate_sage = SageUncompiledValidation(
        validation_signal_sampler,
        validation_noise_sampler,
        processor,
        model,
        loss_function,
        num_iterations = cfg.validation_iterations,
        num_epochs     = cfg.num_epochs,
    )

    ckpt_mgr = CheckpointManager(
        cfg=cfg,
        data_cfg=data_cfg,
        model=model,
        optimizer=optimiser,
        scheduler=scheduler,
        scaler=scaler,
    )

    # ── Epoch loop ────────────────────────────────────────────────────
    logger = HDF5LossLogger(
        path           = os.path.join(cfg.export_dir, "losses.h5"),
        num_epochs     = cfg.num_epochs,
        num_components = train_sage.loss_function.num_components,
    )

    for nepoch in range(cfg.num_epochs):

        print(f"Epoch {nepoch}: Training Sage (hard-noise mining)")
        train_sage(nepoch=nepoch)
        logger.log(train_sage.loss_components, nepoch, split="training")

        if (nepoch + 1) % 5 == 0 or nepoch == 0:
            print(f"Epoch {nepoch}: Validating Sage")
            validate_sage(nepoch=nepoch)
            logger.log(validate_sage.loss_components, nepoch, split="validation")

            val_loss = validate_sage.loss_components[nepoch][0].item()
            ckpt_mgr.save(epoch=nepoch, val_loss=val_loss)


if __name__ == "__main__":
    run_sage()

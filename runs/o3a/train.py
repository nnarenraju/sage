#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : exec.py
Description     : Short description of the file

Created on 2026-03-16 11:29:03

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import os
import torch

# Optional extra silence
torch._dynamo.config.verbose = False
torch._inductor.config.debug = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.cuda.empty_cache()

torch._dynamo.reset()


# LOCAL
from sage.core.config import get_cfg, get_data_cfg
from sage.utils.servers import get_server

# Signal Sampler
from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2

# Noise Sampler
from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess

# Preprocessing
from sage.data.waveform import HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning

# Model and loss
from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic
from sage.architecture.custom_losses import BCEWithPEsigmaLoss
from sage.core.logger import (
    HDF5LossLogger,
    setup_logging,
    get_logger,
)

logger = get_logger("sage.run.o3a")

# Optimiser and scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# SageGraph
from sage.core.graph import Preprocessor
from sage.factory.training import SageVanillaTraining
from sage.factory.validation import SageVanillaValidation

import os as _os, importlib as _il
# Config module to run: named by SAGE_CONFIG (default "config"). Make a
# per-network config with `cp config.py config_<DETS>.py`, edit `detectors`
# + `export_dir`, and pick it at launch. No network logic lives in code.
set_configs = _il.import_module(_os.environ.get("SAGE_CONFIG", "config")).set_configs
from sage.utils.checkpoint import CheckpointManager


def make_training_graph():

    # Make the signal sampler
    training_param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
    waveform_project = ConstantProjection()
    target_snr_sampler = HalfNorm(scale=4.0, loc=5.0, seed=150914)
    snrscaler = OptimalSNRRescaler(target_snr_sampler)
    training_signal_sampler = IMRPhenomPv2(
        training_param_sampler,
        waveform_project,
        augment=snrscaler,
    )

    # Make the noise sampler
    # O3a trains on O3a noise and recolours toward the O3b epoch
    recolour = RecolourPostprocess(
        p_recolour=0.37,
        recolour_dataset_dir=get_server().dataset_dir("O3b"),
    )
    training_noise_sampler = MemmapNoiseSampler(
        postprocess_fn=recolour, prefetch=8, seed=150914
    )

    return (
        training_signal_sampler,
        training_noise_sampler,
        training_param_sampler.bounds,
    )


def make_validation_graph():

    # Make the signal sampler
    validation_param_sampler = read_from_config("./gwconfig.yaml", seed=170817)
    waveform_project = ConstantProjection()
    target_snr_sampler = HalfNorm(scale=4.0, loc=5.0, seed=170817)
    snrscaler = OptimalSNRRescaler(target_snr_sampler)
    validation_signal_sampler = IMRPhenomPv2(
        validation_param_sampler,
        waveform_project,
        augment=snrscaler,
    )

    # Make the noise sampler
    validation_noise_sampler = MemmapNoiseSampler(
        postprocess_fn=None, prefetch=8, seed=170817, training=False
    )

    return validation_signal_sampler, validation_noise_sampler


def make_processor(bounds):

    # Preprocessing
    whitener = FiducialWhitening()
    dyadic_binning = DyadicPyramidBinning(bounds)
    mrsampler = MultirateSampler(binning_method=dyadic_binning)
    processor = Preprocessor([whitener, mrsampler])

    return processor


def run_sage():

    set_configs()
    cfg, data_cfg = get_cfg(), get_data_cfg()

    # Logging: console + <export_dir>/logs/run.log. After set_configs, since
    # that is what tells us where this run writes. SAGE_LOG_LEVEL=DEBUG for the
    # verbose console format.
    log_path = setup_logging(cfg.export_dir)
    logger.info("Run: %s | detectors %s", cfg.export_dir, cfg.detectors)
    logger.info("Logging to %s", log_path)

    # Training, validation and processor
    training_signal_sampler, training_noise_sampler, bounds = make_training_graph()
    validation_signal_sampler, validation_noise_sampler = make_validation_graph()
    processor = make_processor(bounds)

    # Model and optimisation
    model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="instancenorm",
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print("Model compiled with torch.compile!")

    loss_function = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    optimiser = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    train_sage = SageVanillaTraining(
        training_signal_sampler,
        training_noise_sampler,
        processor,
        model,
        loss_function,
        optimiser,
        scheduler,
        scaler=scaler,
        num_iterations=cfg.training_iterations,
        num_epochs=cfg.num_epochs,
    )

    validate_sage = SageVanillaValidation(
        validation_signal_sampler,
        validation_noise_sampler,
        processor,
        model,
        loss_function,
        num_iterations=cfg.validation_iterations,
        num_epochs=cfg.num_epochs,
    )

    ckpt_mgr = CheckpointManager(
        cfg=cfg,
        data_cfg=data_cfg,
        model=model,
        optimizer=optimiser,
        scheduler=scheduler,
        scaler=scaler,
    )

    ## TRAINING LOOP

    loss_logger = HDF5LossLogger(
        path=os.path.join(cfg.export_dir, "losses.h5"),
        num_epochs=cfg.num_epochs,
        num_components=train_sage.loss_function.num_components,
    )

    for nepoch in range(cfg.num_epochs):

        # TRAINING
        train_sage(nepoch=nepoch)
        loss_logger.log(train_sage.loss_components, nepoch, split="training")

        # VALIDATION
        if (nepoch + 1) % 5 == 0 or nepoch == 0:
            validate_sage(nepoch=nepoch)
            loss_logger.log(validate_sage.loss_components, nepoch, split="validation")

            # Saving total loss and checkpointing
            val_loss = validate_sage.loss_components[nepoch][0].item()
            ckpt_mgr.save(epoch=nepoch, val_loss=val_loss)

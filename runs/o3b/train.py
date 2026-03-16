#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : exec.py
Description     : Short description of the file

Created on 2026-03-16 11:29:03

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import torch

# LOCAL
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig

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
from sage.architecture.network import MSCNN1D_2DResNetCBAM
from sage.architecture.custom_losses import BCEWithPEregLoss

# Optimiser and scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# SageGraph
from sage.core.graph import Preprocessor
from sage.factory.training import SageUncompiledTraining
from sage.factory.validation import SageUncompiledValidation

# Configs
from config import O3aCFG, O3aDataCFG

# Datasets
from dataset import get_timeline, download_dataset, make_psds


def make_training_graph():

    # Make the signal sampler
    training_param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
    waveform_project = ConstantProjection()
    target_snr_sampler = HalfNorm(scale=4.0, loc=5.0, seed=150914)
    snrscaler = OptimalSNRRescaler(target_snr_sampler)
    training_signal_sampler = IMRPhenomPv2(
        training_param_sampler, waveform_project, augment=snrscaler
    )

    # Make the noise sampler
    recolour = RecolourPostprocess(p_recolour=0.37)
    training_noise_sampler = MemmapNoiseSampler(
        postprocess_fn=recolour, prefetch=4, seed=150914
    )

    return training_signal_sampler, training_noise_sampler


def make_validation_graph():

    # Make the signal sampler
    waveform_project = ConstantProjection()
    validation_param_sampler = read_from_config("./gwconfig.yaml", seed=170817)
    validation_signal_sampler = IMRPhenomPv2(validation_param_sampler, waveform_project)

    # Make the noise sampler
    validation_noise_sampler = MemmapNoiseSampler(
        postprocess_fn=None, prefetch=4, seed=170817
    )

    return validation_signal_sampler, validation_noise_sampler


def make_processor(training_param_sampler):

    # Preprocessing
    whitener = FiducialWhitening()
    dyadic_binning = DyadicPyramidBinning(training_param_sampler.bounds)
    mrsampler = MultirateSampler(binning_method=dyadic_binning)
    processor = Preprocessor([whitener, mrsampler])

    return processor


def get_configs():

    # Read configs
    cfg = BaseConfig(O3aCFG())
    data_cfg = BaseDataConfig(O3aDataCFG())

    # Register configurations for the Sage run
    register_configs(cfg, data_cfg)

    return cfg, data_cfg


def run():

    # Shared configs
    cfg, data_cfg = get_configs()

    # Make datasets
    tq = get_timeline(data_cfg)
    download_dataset(tq, data_cfg)
    for det in ["H1", "L1", "V1"]:
        make_psds(det, data_cfg)

    # Training, validation and processor
    training_signal_sampler, training_noise_sampler = make_training_graph()
    validation_signal_sampler, validation_noise_sampler = make_validation_graph()
    processor = make_processor()

    # Model and optimisation
    model = MSCNN1D_2DResNetCBAM(
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="instancenorm",
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    loss_function = BCEWithPEregLoss(regression_weight=0.3)
    optimiser = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=1, eta_min=1e-6)

    train_sage = SageUncompiledTraining(
        training_signal_sampler,
        training_noise_sampler,
        processor,
        model,
        loss_function,
        optimiser,
        scheduler,
        num_iterations=cfg.training_iterations,
        num_epochs=cfg.num_epochs,
    )

    validate_sage = SageUncompiledValidation(
        validation_signal_sampler,
        validation_noise_sampler,
        processor,
        model,
        loss_function,
        num_iterations=cfg.validation_iterations,
        num_epochs=cfg.num_epochs,
    )

    ## TRAINING LOOP

    for nepoch in range(cfg.num_epochs):
        train_sage(nepoch=nepoch)
        if nepoch % 5 == 0:
            validate_sage(nepoch=nepoch)

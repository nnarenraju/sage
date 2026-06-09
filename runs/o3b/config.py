#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config.py
Description     : Short description of the file

Created on 2026-03-16 11:45:28

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
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

# Configs
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig


class O3bCFG:

    export_dir = "./run_export"
    batch_size = 128
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    num_epochs = 80
    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)


class O3bDataCFG:

    data_dir = "/data/wiay/nnarenraju/data_release/o3b_dataset/data_dir"
    training_noise_files = [
        "/data/wiay/nnarenraju/data_release/o3b_dataset/data_H1_O3b.bin",
        "/data/wiay/nnarenraju/data_release/o3b_dataset/data_L1_O3b.bin",
    ]
    validation_noise_files = [
        "/data/wiay/nnarenraju/data_release/o3a_dataset/data_H1_O3a.bin",
        "/data/wiay/nnarenraju/data_release/o3a_dataset/data_L1_O3a.bin",
    ]
    sample_rate = 2048.0  # Hz
    noise_low_frequency_cutoff = 15.0  # Hz
    signal_low_frequency_cutoff = 20.0  # Hz
    sample_length_in_s = 12.0  # seconds
    padding_length_in_s = 2.0  # seconds


def _register():

    # Read configs
    cfg = BaseConfig(O3bCFG())
    data_cfg = BaseDataConfig(O3bDataCFG())

    # Register configurations for the Sage run
    register_configs(cfg, data_cfg)
    print("Registered cfg and data_cfg!")


def set_configs():

    # Register Shared configs with Sage
    _register()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config.py
Description     : O4b run configuration.

__author__        = Narenraju Nagarajan
__license__       = MIT Licence
__status__        = ['inUsage']
"""

# Packages
import torch

# Configs
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig

# Server-specific paths (one switch: SAGE_SERVER). See sage/utils/servers.py.
from sage.utils.servers import get_server
_SRV = get_server()

# O4b lives in its own isolated release dir (see dataset.py _RELEASE_DIRNAME).
_DSDIR = f"{_SRV.data_root}/data_release_o4b"
# Cross-run validation noise (O4a) lives in its own isolated dir too.
_DSDIR_VAL = f"{_SRV.data_root}/data_release_o4a"


class O4bCFG:

    export_dir = "./run_export"
    batch_size = 128
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1", "V1"]
    # Observing run(s) whose noise this model trains on. Drives the hard-mining
    # bank filename/metadata (hardbank_<runs>_<dets>.h5). For multi-run O4
    # training this widens to e.g. ["O3a", "O3b", "O4a", "O4b"].
    train_runs = ["O4b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0
    num_epochs = 80
    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)


class O4bDataCFG:

    data_dir = f"{_DSDIR}/data_dir"
    training_noise_files = [
        f"{_DSDIR}/data_H1_O4b.bin",
        f"{_DSDIR}/data_L1_O4b.bin",
        f"{_DSDIR}/data_V1_O4b.bin",
    ]
    validation_noise_files = [
        f"{_DSDIR_VAL}/data_H1_O4a.bin",
        f"{_DSDIR_VAL}/data_L1_O4a.bin",
    ]
    sample_rate = 2048.0  # Hz
    noise_low_frequency_cutoff = 15.0  # Hz
    signal_low_frequency_cutoff = 20.0  # Hz
    sample_length_in_s = 12.0  # seconds
    padding_length_in_s = 2.0  # seconds


def _register():
    cfg = BaseConfig(O4bCFG())
    data_cfg = BaseDataConfig(O4bDataCFG())
    register_configs(cfg, data_cfg)
    print("Registered cfg and data_cfg!")


def set_configs():
    _register()

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

# Server-specific paths (one switch: SAGE_SERVER). See sage/utils/servers.py.
from sage.utils.servers import get_server
_SRV = get_server()


class O3bCFG:

    export_dir = "./run_export"
    batch_size = 128
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1"]
    # Observing run(s) whose noise this model trains on. Drives the hard-mining
    # bank filename/metadata (hardbank_<runs>_<dets>.h5). For multi-run O4
    # training this becomes e.g. ["O3a", "O3b", "O4a"].
    train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0  # set >0 (e.g. 0.05) to enable dropout + MC-dropout uncertainty
    # Fraction of the *noise* budget (B - S = half the batch at class_balance
    # 0.5) replaced by non-astrophysical pairs (signal+noise / signal+signal')
    # to train the consistency heads to reject incoherent coincidences. Eats
    # noise slots only, so class balance is preserved. TRAINING ONLY.
    p_non_astrophysical = 0.0
    num_epochs = 80
    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)


class O3bDataCFG:

    # Derived PSDs (recolour/segment) live under the run's dataset dir; the raw
    # noise .bin lives flat in its release dir (O3b = default "data_release";
    # O3a was downloaded into the isolated "data_release_o3a").
    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [
        _SRV.noise_bin("H1", "O3b"),                       # data_release/
        _SRV.noise_bin("L1", "O3b"),
    ]
    validation_noise_files = [
        _SRV.noise_bin("H1", "O3a", "data_release_o3a"),   # isolated dir
        _SRV.noise_bin("L1", "O3a", "data_release_o3a"),
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

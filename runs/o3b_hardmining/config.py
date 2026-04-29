#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
O3b hard-mining run configuration.

Identical hardware setup to the base O3b run.  Key differences:
  - BCEWithFARLoss (pAUC + focal mix) instead of BCEWithPEsigmaLoss
  - SageHardMiningTraining with hard-noise / hard-signal replay + adv noise
  - Slightly more training budget per epoch to absorb mining overhead
"""

import torch

from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig


class O3bHardMiningCFG:

    export_dir   = "./run_export"
    batch_size   = 128
    device       = "cuda:0"
    dtype        = torch.float32
    detectors    = ["H1", "L1"]
    do_point_estimate = ["tc", "mchirp"]
    autocast     = True
    class_balance = 0.5
    clip_norm    = 1.0
    num_epochs   = 80
    training_iterations   = int(2_000_000 / batch_size)
    validation_iterations = int(200_000  / batch_size)


class O3bHardMiningDataCFG:

    data_dir = "/local/scratch/igr/nnarenraju/search/o3b/data_dir"
    training_noise_files = [
        "/local/scratch/igr/nnarenraju/search/o3b/data_release/data_H1_O3b.bin",
        "/local/scratch/igr/nnarenraju/search/o3b/data_release/data_L1_O3b.bin",
    ]
    validation_noise_files = [
        "/local/scratch/igr/nnarenraju/search/o3a/data_release/data_H1_O3a.bin",
        "/local/scratch/igr/nnarenraju/search/o3a/data_release/data_L1_O3a.bin",
    ]
    sample_rate              = 2048.0
    noise_low_frequency_cutoff  = 15.0
    signal_low_frequency_cutoff = 20.0
    sample_length_in_s       = 12.0
    padding_length_in_s      = 2.0


def _register():
    cfg      = BaseConfig(O3bHardMiningCFG())
    data_cfg = BaseDataConfig(O3bHardMiningDataCFG())
    register_configs(cfg, data_cfg)
    print("Registered O3b hard-mining cfg and data_cfg.")


def set_configs():
    _register()

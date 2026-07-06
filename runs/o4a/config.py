#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config.py
Description     : O4a run configuration.

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

# O4a lives in its own isolated release dir (see dataset.py _RELEASE_DIRNAME).
_DSDIR = f"{_SRV.data_root}/data_release_o4a"


class O4aCFG:

    export_dir = "./run_export"
    batch_size = 128
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1"]
    # Observing run(s) whose noise this model trains on (roadmap: O4a trains on
    # pooled O3a + O3b noise, recoloured toward O4a). Also drives the hard-mining
    # bank filename/metadata (hardbank_<runs>_<dets>.h5).
    train_runs = ["O3a", "O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0
    num_epochs = 80
    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)


class O4aDataCFG:

    data_dir = f"{_DSDIR}/data_dir"
    # Multi-run training noise: pool O3a + O3b. Each run is read from its own
    # release dir and carries its own derived-PSD data_dir (for the per-run
    # segment whitening PSDs the recolour step keys by (run, segment)). At train
    # time these are recoloured toward O4a (the eval run).
    training_noise = [
        _SRV.run_noise("O3a", O4aCFG.detectors, "data_release_o3a"),
        _SRV.run_noise("O3b", O4aCFG.detectors, "data_release"),
    ]
    # Eval on the actual target run (O4a), read raw (no recolour).
    validation_noise_files = [
        f"{_DSDIR}/data_H1_O4a.bin",
        f"{_DSDIR}/data_L1_O4a.bin",
    ]
    sample_rate = 2048.0  # Hz
    noise_low_frequency_cutoff = 15.0  # Hz
    signal_low_frequency_cutoff = 20.0  # Hz
    sample_length_in_s = 12.0  # seconds
    padding_length_in_s = 2.0  # seconds


def _register():
    cfg = BaseConfig(O4aCFG())
    data_cfg = BaseDataConfig(O4aDataCFG())
    register_configs(cfg, data_cfg)
    print("Registered cfg and data_cfg!")


def set_configs():
    _register()

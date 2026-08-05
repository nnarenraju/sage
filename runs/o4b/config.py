#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config.py
Description     : O4b run configuration.

__author__        = Narenraju Nagarajan
__license__       = GPL-3.0-or-later
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
# O4a (pooled into training noise) lives in its own flat isolated dir.
_O4A_DIR = f"{_SRV.data_root}/data_release_o4a"


class O4bCFG:

    # Runs export to /work, never home: filling the home quota mid-run
    # corrupts checkpoints (Errno 122). Mirrors the o3a/o3b layout,
    # sage_runs/<run>/prod_<DETS>.
    export_dir = f"{_SRV.data_root}/sage_runs/o4b/prod_HLV"
    # Fiducial ASDs are per-detector and shared across a run's networks, so
    # they stay with the repo rather than moving with export_dir.
    fiducial_dir = "./run_export/fiducial_psds"
    batch_size = 128
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1", "V1"]
    # Observing run(s) whose noise this model trains on (roadmap: O4b trains on
    # pooled O3a + O3b + O4a noise, recoloured toward O4b). Also drives the
    # hard-mining bank filename/metadata (hardbank_<runs>_<dets>.h5).
    train_runs = ["O3a", "O3b", "O4a"]
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
    # Multi-run training noise: pool O3a + O3b + O4a, each read from its own
    # release dir with its own derived-PSD data_dir. O4a uses the flat O4 layout
    # (data_release_o4a/data_dir); O3a/O3b use the O3-style <run>_dataset dir.
    # Recoloured toward O4b (the eval run) at train time.
    training_noise = [
        _SRV.run_noise("O3a", O4bCFG.detectors, "data_release_o3a"),
        _SRV.run_noise("O3b", O4bCFG.detectors, "data_release"),
        _SRV.run_noise("O4a", O4bCFG.detectors, "data_release_o4a",
                       data_dir=f"{_O4A_DIR}/data_dir"),
    ]
    # Eval on the actual target run (O4b), read raw (no recolour).
    validation_noise_files = [
        f"{_DSDIR}/data_H1_O4b.bin",
        f"{_DSDIR}/data_L1_O4b.bin",
        f"{_DSDIR}/data_V1_O4b.bin",
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

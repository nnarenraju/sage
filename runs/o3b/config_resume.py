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

    # One detector-set network per config. For LV / HV / HLV, copy this config
    # and change `detectors` + `export_dir` together (e.g. detectors=["L1","V1"]
    # + export_dir="./run_export_LV"); everything else (noise files, model,
    # recolour, hard-mining bank) follows from `detectors` automatically.
    export_dir = "./run_export_resume"
    # Fiducial PSDs are per-detector and shared across a run's networks, so they
    # live in one place regardless of which detector set this config trains.
    fiducial_dir = "./run_export/fiducial_psds"
    batch_size = 128
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1"]          # HL network
    # Observing run(s) whose noise this model trains on. Drives the hard-mining
    # bank filename/metadata (hardbank_<runs>_<dets>.h5).
    train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0  # set >0 (e.g. 0.05) to enable dropout + MC-dropout uncertainty
    num_epochs = 40                   # RESUME TEST: cannot finish in one 30-min segment
    mine_schedule = [0]               # mine at epoch 0
    keep_threshold_raw = -5.0
    mine_iters = 60
    bank_dir = "/work/nagarajan/hard_mining_resume"   # throwaway bank
    training_iterations = 400          # RESUME TEST: ~30 s/epoch on H100
    validation_iterations = 80


class O3bDataCFG:

    # Derived PSDs (recolour/segment) live under the run's dataset dir; the raw
    # noise .bin lives flat in its release dir (O3b = default "data_release";
    # O3a was downloaded into the isolated "data_release_o3a").
    # Noise files follow O3bCFG.detectors, so switching the network only needs
    # the CFG edit above (detectors + export_dir).
    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [
        _SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors            # data_release/
    ]
    validation_noise_files = [
        _SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3bCFG.detectors
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

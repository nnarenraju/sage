#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config.py
Description     : Short description of the file

Created on 2026-03-16 11:45:28

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
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
import torch

# Configs
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig

# Server-specific paths (one switch: SAGE_SERVER). See sage/utils/servers.py.
from sage.utils.servers import get_server
_SRV = get_server()


class O3aCFG:

    # One detector-set network per config. For LV / HV / HLV, copy this config
    # and change `detectors` + `export_dir` together (e.g. detectors=["L1","V1"]
    # + export_dir="./run_export_LV"); everything else (noise files, model,
    # recolour, hard-mining bank) follows from `detectors` automatically.
    export_dir = "/work/nagarajan/sage_runs/o3a/prod_HLV"
    # Fiducial PSDs are per-detector and shared across a run's networks, so they
    # live in one place regardless of which detector set this config trains.
    fiducial_dir = "./run_export/fiducial_psds_o3ab"
    batch_size = 64
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1", "V1"]    # HLV network
    # Observing run(s) whose noise this model trains on. Drives the hard-mining
    # bank filename/metadata (hardbank_<runs>_<dets>.h5).
    train_runs = ["O3a"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0  # set >0 (e.g. 0.05) to enable dropout + MC-dropout uncertainty
    num_epochs = 128               # ~3.5 days at 14.2 it/s (measured); hard cap 4 days
    warmup_steps = 20_000          # linear LR warmup (~0.6 epoch at batch 64)
    ema_decay = 0.9999             # per-step weight EMA (the deliverable model)
    keep_last_ckpts = 2            # keep 2 newest epoch_N restart points; best.pt/ema.pt always kept

    # New pipeline features -- explicit = ON for production:
    use_blurpool = True            # anti-aliased BlurPool downsampling (front + backend)
    use_resnet_cd = True           # ResNet-C deep stem + ResNet-D avg-down
    recolour_dr_gain = 0.5         # data-driven k*sigma(f) recolour PSD augmenter (0.0 = off)

    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)


class O3aDataCFG:

    # O3a noise was downloaded into the isolated "data_release_o3a"; O3b (used
    # for validation) lives in the default "data_release". Noise files follow
    # O3aCFG.detectors, so switching the network only needs the CFG edit above.
    data_dir = _SRV.data_dir("O3a")
    training_noise_files = [
        _SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3aCFG.detectors
    ]
    validation_noise_files = [
        _SRV.noise_bin(d, "O3b") for d in O3aCFG.detectors   # default data_release/
    ]
    sample_rate = 2048.0  # Hz
    noise_low_frequency_cutoff = 15.0  # Hz
    signal_low_frequency_cutoff = 20.0  # Hz
    sample_length_in_s = 12.0  # seconds
    padding_length_in_s = 2.0  # seconds


def _register():

    cfg = BaseConfig(O3aCFG())
    data_cfg = BaseDataConfig(O3aDataCFG())
    register_configs(cfg, data_cfg)
    print("Registered cfg and data_cfg!")


def set_configs():

    _register()

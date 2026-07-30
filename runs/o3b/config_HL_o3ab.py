#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config_HL_o3ab.py -- production HL recipe, whitened with a COMBINED O3a+O3b fiducial.

IDENTICAL to config_HL (the run_export_HL baseline that plateaus at val BCE 0.2333)
in EVERY setting -- 128 epochs, hard mining, recolour_dr_gain=0.5, BlurPool +
ResNet-C/D, GroupNorm (train_hard.py) -- with ONE change:

    fiducial_dir = "./run_export/fiducial_psds_o3ab"   # combined, not O3b-only

The combined fiducial is the element-wise max of the O3b fiducial ASD and the
smoothed O3a median ASD, so it suppresses the UNION of O3a and O3b spectral lines.
Motivation (2026-07-27 diagnosis): our validation is O3a whitened by the O3b-only
fiducial, which does NOT contain H1's O3a lines -> H1 O3a whitened-noise floor is
2.7-3.0x ideal (vs 1.08x on O3b) -> poisons every H1-containing combo's validation
(LV, the H1-free pair, is our best at 0.2105). The combined fiducial drops H1 O3a
floor 3.0x -> 1.05x and makes train (O3b) and val (O3a) whiten to a CONSISTENT
floor. This run tests whether closing that whitening domain-gap recovers the O3a
validation BCE toward the old paper run's in-domain 0.185.

Single-variable A/B: compare val trajectory head-to-head against run_export_HL.

Launch:  ./submit.sh chain config_HL_o3ab 3
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

    export_dir = "./run_export_HL_o3ab"                    # fresh, isolated
    fiducial_dir = "./run_export/fiducial_psds_o3ab"       # COMBINED O3a+O3b (the ONLY change)
    batch_size = 64
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1"]          # HL network
    train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0
    num_epochs = 128               # match config_HL exactly
    warmup_steps = 20_000
    ema_decay = 0.9999
    keep_last_ckpts = 2

    use_blurpool = True
    use_resnet_cd = True
    recolour_dr_gain = 0.5

    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)


class O3bDataCFG:

    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [
        _SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors
    ]
    validation_noise_files = [
        _SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3bCFG.detectors
    ]
    sample_rate = 2048.0
    noise_low_frequency_cutoff = 15.0
    signal_low_frequency_cutoff = 20.0
    sample_length_in_s = 12.0
    padding_length_in_s = 2.0


def _register():
    cfg = BaseConfig(O3bCFG())
    data_cfg = BaseDataConfig(O3bDataCFG())
    register_configs(cfg, data_cfg)
    print("Registered cfg and data_cfg!  (HL production + COMBINED O3a+O3b fiducial)")


def set_configs():
    _register()

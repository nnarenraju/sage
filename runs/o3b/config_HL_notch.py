#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config_HL_notch.py -- production HL recipe, COMBINED O3a+O3b fiducial + FFT line notch.

IDENTICAL to config_HL_o3ab (job 107444, the combined-fiducial run) in every setting,
with ONE addition:

    use_line_notch = True

Exp 1 from the SP survey (Boll 1979 spectral subtraction; Thomson 1982 F-test). After
whitening by the combined max(A,B) fiducial, bins flagged as spectral lines have their
magnitude pulled DOWN to the local-median floor (suppress-only, gain<=1, phase kept).
The combined fiducial suppresses lines only to the MEDIAN level it is built from; wandering
/ flaring lines (esp. L1) leave residual power. Offline check (2026-07-28): the notch takes
L1 O3a whitened-noise sigma from 1.18x -> 0.99x ideal (median) and p90 13x -> 3x, with
signal power retained = 1.00000 (lines are noise-only bins). gain<=1 => can never amplify.

Single-variable A/B vs job 107444 (combined fiducial, no notch): does removing the residual
line power lift the noise-ranking-stat tail off the detection region further and close more
of the gap to the old paper run?

Launch:  ./submit.sh chain config_HL_notch 3
"""

# Packages
import torch

# Configs
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig

from sage.utils.servers import get_server
_SRV = get_server()


class O3bCFG:

    export_dir = "./run_export_HL_notch"                   # fresh, isolated
    fiducial_dir = "./run_export/fiducial_psds_o3ab"       # combined O3a+O3b
    use_line_notch = True                                  # THE single addition vs 107444
    batch_size = 64
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1"]
    train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0
    num_epochs = 128
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
    print("Registered cfg and data_cfg!  (HL production + combined fiducial + LINE NOTCH)")


def set_configs():
    _register()

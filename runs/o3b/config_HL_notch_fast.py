#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config_HL_notch_fast.py -- IDENTICAL to config_HL_notch (combined O3a+O3b fiducial +
FFT line notch + full production recipe: blurpool, resnet-c/d, mining, GroupNorm,
warmup, cosine-NO-restarts) with ONE change:

    anneal_fraction = 0.5

The cosine LR anneals to eta_min over the FIRST HALF of the (post-warmup) run, then
holds at eta_min for the second half. Motivation (2026-07-28 gain-timing check on the
128-ep baseline): ~90% of the validation improvement happens by epoch 20, then the long
high-LR tail buys ~0.013 over 108 epochs -- the model gains little from sustained high LR.
This front-loads the decay to reach the productive low-LR regime sooner.

A/B partner: config_HL_notch (anneal_fraction=1.0, the usual single cosine). Same
everything else. No min-max, no arcsinh (magnitude is information; let the model learn
glitch-vs-signal).

Launch:  ./submit.sh chain config_HL_notch_fast 3
"""

import torch

from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()


class O3bCFG:

    export_dir = "/work/nagarajan/sage_runs/o3b/run_export_HL_notch_fast"  # /work (NOT home: avoids the home-quota checkpoint corruption)
    fiducial_dir = "./run_export/fiducial_psds_o3ab"      # combined O3a+O3b
    use_line_notch = True                                 # FFT line notch
    anneal_fraction = 0.5                                 # THE single difference vs config_HL_notch
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
    print("Registered cfg and data_cfg!  (HL production + combined fiducial + notch + FASTER cosine)")


def set_configs():
    _register()

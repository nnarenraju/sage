#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config_ema_smoke.py -- TEMPORARY tiny GPU smoke for the code added SINCE the
last (passing) resume-chain test: the official-torch per-step EMA
(AveragedModel + get_ema_multi_avg_fn) under torch.compile + AMP, and the
post-training calibrate_ema() path (torch.optim.swa_utils.update_bn + the
ema-vs-best validation compare). Also re-exercises hard mining with the new
bias-anneal + hard-sample-age logging.

Mirrors config_HL (O3b, H1/L1, batch 64, real data/fiducials/priors) but SHORT,
with throwaway export/bank dirs. Intended to be run as a SINGLE job:
    SAGE_CONFIG=config_ema_smoke python -c \
        'from train_hard import run_hard, calibrate_ema; run_hard(); calibrate_ema()'

Epoch map (num_epochs=5): validate at ep 0 & 4 -> best.pt written; mine at
ep 1 & 3 -> bias/age exercised; EMA every batch -> ema.pt written each epoch.

Not for production. Delete run_export_ema_smoke/ and the throwaway bank after.
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

    export_dir = "./run_export_ema_smoke"         # THROWAWAY
    fiducial_dir = "./run_export/fiducial_psds"   # real, shared (read-only)
    batch_size = 64                               # match production
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1"]                      # HL network
    train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0

    # ── Short run: EMA accumulates over ~1500 steps; val at ep 0 & 4 (best.pt). ─
    num_epochs = 5
    warmup_steps = 100             # small (production 20_000)
    ema_decay = 0.99               # short averaging window for a short run
    training_iterations = 300      # fast epoch (~21 s at 14 it/s)
    validation_iterations = 30     # validates at ep 0 & 4 -> writes best.pt

    # ── Hard mining ON: small but exercises bias-anneal + hard-sample-age on GPU.
    mine_schedule = 2              # mine when (nepoch+1) % 2 == 0 -> ep 1, 3
    mine_iters = 60                # fast mine (production 6000)
    keep_threshold_raw = 0.0
    hard_bias_prob = 0.2
    n_cells = 256                  # small archive for a fast, clean CVT build
    bank_dir = "/work/nagarajan/hard_mining_emasmoke"   # THROWAWAY

    # ── calibrate_ema() overrides: keep the post-training step fast for a smoke.
    bn_calib_batches = 50          # production default 500
    calib_val_iters = 20           # production uses validation_iterations


class O3bDataCFG:

    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [
        _SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors
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
    cfg = BaseConfig(O3bCFG())
    data_cfg = BaseDataConfig(O3bDataCFG())
    register_configs(cfg, data_cfg)
    print("Registered cfg and data_cfg!")


def set_configs():
    _register()

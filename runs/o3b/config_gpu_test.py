#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config_gpu_test.py -- TEMPORARY short end-to-end + RESUME validation config.

Mirrors config_HL (O3b, H1/L1, batch 64, real data/fiducials/priors) but with a
short run length + throwaway export/bank dirs, so a 2-segment `chain` (submitted
with a short per-segment wall-time) is guaranteed to wall-kill mid-run and resume.

Exercises the FULL production pipeline: torch.compile(max-autotune) on the new
BlurPool / ResNet-C-D / GroupNorm architecture, bf16 autocast, per-step EMA,
recolour k*sigma(f) domain randomisation, hard mining (mine -> bank -> bias ->
+/-2s jitter replay -> recolour -> reevaluate) across the resume boundary, and
checkpoint/RNG/EMA/mining-bank resume.

Not for production. Delete run_export_gpu_test/ and the throwaway bank afterwards.
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

    export_dir = "./run_export_gpu_test"          # THROWAWAY
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

    # ── Short run so a 2-segment chain (short --time) resumes mid-run ──────────
    num_epochs = 40                # long enough to exceed one short segment
    warmup_steps = 200             # small (production is 20_000)
    ema_decay = 0.999              # short averaging window for a short run
    training_iterations = 600      # fast epoch
    validation_iterations = 50     # validates every 5 epochs (best.pt path)

    # ── Hard mining: small but exercises the FULL path (new jitter/recolour/
    #    diversity), and fires several times on BOTH sides of the resume boundary.
    mine_schedule = 3              # mine at ep 2,5,8,...,38 ((nepoch+1)%3)
    mine_iters = 60                # fast mine (production 1500)
    keep_threshold_raw = 0.0
    hard_bias_prob = 0.2
    n_cells = 256                  # small archive for a fast, clean CVT build
    bank_dir = "/work/nagarajan/hard_mining_gputest"   # THROWAWAY
    # emitter_batch_size / novelty_dist / novelty_weight / descriptor_dim ->
    # production defaults (exercise the real diversity settings).


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

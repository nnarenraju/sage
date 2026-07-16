#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config_hlv_test.py -- TEMPORARY GPU test with two goals:
  1. Exercise the HLV (3-detector -> 3-channel) path end-to-end. The smoke ran
     HL (2ch); HLV feeds 3 channels into the backend via in_channels=num_detectors
     -- validate build/compile/train/EMA/mine/validate on the 3-channel network.
  2. Measure PRODUCTION timing on the WORST-CASE variant (HLV = most compute):
     - clean steady-state training it/s -> per-epoch time (x128),
     - ONE mine event at the PRODUCTION mine_iters=6000, n_cells=2048 -> per-event
       cost (x19 front-load events).
     Extrapolate: does 128 epochs + 19 mine events finish within the 4-day cap?

Mirrors config_HLV (O3b, H1/L1/V1, batch 64, real data) but short, with a
throwaway export/bank dir. Not for production. Delete run_export_hlv_test/ and
the throwaway bank after.
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

    export_dir = "./run_export_hlv_test"          # THROWAWAY
    fiducial_dir = "./run_export/fiducial_psds"   # real, shared (read-only)
    batch_size = 64                               # match production
    device = "cuda:0"
    dtype = torch.float32
    detectors = ["H1", "L1", "V1"]                # HLV network -> 3 channels
    train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0

    # ── Short: enough for a stable steady-state it/s after compile warmup. ──────
    num_epochs = 3                 # ep0 (compile+train), ep1 (train+mine), ep2 (clean)
    warmup_steps = 100
    ema_decay = 0.99
    training_iterations = 1000     # stable it/s sample (production 31_250)
    validation_iterations = 20

    # ── Mining at PRODUCTION scale: measure one real mine event's wall time. ────
    mine_schedule = 2              # mine once, at end of ep 1
    mine_iters = 6000              # PRODUCTION value (smoke used 60)
    keep_threshold_raw = 0.0
    hard_bias_prob = 0.2
    n_cells = 2048                 # PRODUCTION archive size (smoke used 256)
    bank_dir = "/work/nagarajan/hard_mining_hlvtest"   # THROWAWAY
    # emitter_batch_size / novelty_dist / novelty_weight / descriptor_dim /
    # max_embeddings -> production defaults (exercise the real mining cost).


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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config_base.py  (O3b)
Description     : Shared production config factory for the O3b networks.

Every O3b network (HL / HV / LV / HLV) is identical except for ``detectors`` and
``export_dir``.  The per-detector ``config_*.py`` modules are thin wrappers that
call :func:`register`; ALL real settings live here so they can never drift
between the four networks.

__license__       = GPL-3.0-or-later
"""

# Packages
import torch

# Configs
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig

# Server-specific paths (one switch: SAGE_SERVER). See sage/utils/servers.py.
from sage.utils.servers import get_server
_SRV = get_server()

# Combined O3a+O3b line-notched fiducial ASDs (LocalLineNotch K=4), SHARED by both
# the O3a and O3b runs. Absolute path (never home-relative). Built by build_fiducial.py.
FIDUCIAL_DIR = "/work/nagarajan/sage_runs/fiducial_psds_o3ab"


def make_configs(detectors, export_dir, norm_type="instancenorm"):
    """Build the (training, data) config classes for one detector-set network."""

    class O3bCFG:

        # --- per-network (assigned below) ---------------------------------
        # detectors, export_dir

        # --- shared production settings -----------------------------------
        fiducial_dir = FIDUCIAL_DIR
        batch_size = 64
        device = "cuda:0"
        dtype = torch.float32
        # Trains on O3b noise; validates on the cross-domain O3a run (data cfg).
        train_runs = ["O3b"]
        do_point_estimate = ["tc", "mchirp"]
        autocast = True
        class_balance = 0.5
        clip_norm = 100.0              # global-norm gradient clip
        dropout = 0.0                  # >0 (e.g. 0.05) enables dropout + MC-dropout
        num_epochs = 128               # 3 chained <=2-day jobs cover this (./submit.sh chain <cfg> 3)
        warmup_steps = 20_000          # linear LR warmup (~0.6 epoch at batch 64)
        ema_decay = 0.9999             # per-step weight EMA (the deliverable model)
        keep_last_ckpts = 2            # keep 2 newest epoch_N restart points; best/ema always kept

        # Model / pipeline (explicit = ON for production)
        use_blurpool = True            # anti-aliased BlurPool downsampling (front + backend)
        use_resnet_cd = True           # ResNet-C deep stem + ResNet-D avg-down
        recolour_dr_gain = 0.5         # data-driven k*sigma(f) recolour PSD augmenter (0.0 = off)

        # LR schedule: PolynomialLR(power) decay peak->eta_min, then a short flat
        # tail at eta_min (see train_hard.py). power>1 leaves the peak faster than
        # cosine's flat top; the flat tail lets EMA settle.
        lr_anneal_power = 2.0
        lr_flat_tail_epochs = 4

        # Hard mining
        mine_iters = 8000              # CMA-MAE generations per mining round
        bias_replays_per_epoch = 2.0   # hard-bias 0->0.2 crossover at a 200k active bank

        training_iterations = int(2_000_000 / batch_size)
        validation_iterations = int(200_000 / batch_size)

    O3bCFG.detectors = list(detectors)
    O3bCFG.export_dir = export_dir
    O3bCFG.norm_type = norm_type    # backend norm; "instancenorm" (default) | "groupnorm"

    class O3bDataCFG:

        # O3b noise lives flat in the default "data_release"; O3a (validation) lives
        # in the isolated "data_release_o3a". Noise files follow `detectors`.
        data_dir = _SRV.data_dir("O3b")
        training_noise_files = [
            _SRV.noise_bin(d, "O3b") for d in detectors            # data_release/
        ]
        validation_noise_files = [
            _SRV.noise_bin(d, "O3a", "data_release_o3a") for d in detectors
        ]
        sample_rate = 2048.0  # Hz
        noise_low_frequency_cutoff = 15.0  # Hz
        signal_low_frequency_cutoff = 20.0  # Hz
        sample_length_in_s = 12.0  # seconds
        padding_length_in_s = 2.0  # seconds

    return O3bCFG, O3bDataCFG


def register(detectors, export_dir, norm_type="instancenorm"):
    """Instantiate + register the (training, data) configs for one network."""
    cfg_cls, data_cfg_cls = make_configs(detectors, export_dir, norm_type)
    register_configs(BaseConfig(cfg_cls()), BaseDataConfig(data_cfg_cls()))
    print(f"Registered cfg and data_cfg! (detectors={list(detectors)}, "
          f"export_dir={export_dir})")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config_base.py  (O4b)
Description     : Shared production config factory for the O4b-target networks.

Every O4b network (HL / HV / LV / HLV) is identical except for ``detectors``,
``export_dir`` and which runs the noise pool can draw from. The per-detector
``config_*.py`` modules are thin wrappers that call :func:`register`; ALL real
settings live here so they can never drift between networks.

Training noise is recoloured toward O4b (the eval run) at train time; validation
reads raw O4b. Settings match the O3 production runs exactly so the networks stay
directly comparable -- only the noise run, recolour target and export path differ.

Noise run and Virgo
-------------------
HL trains on O4a ALONE, mirroring the O3 production recipe (one run to train, the
other to validate) and keeping the whitening reference, the fiducial-defined SNR
label and the recolour target all inside O4. See runs/o4a/config_base.py for why
pooling O3 into an O4 target is self-defeating.

The Virgo networks cannot do this: O4a carries H1 and L1 only (Virgo did not
observe in O4a), and ``sage/data/noise/real_noise.py`` raises when a pooled run
lacks a network's detector. O3a + O3b are the ONLY Virgo training data that exist,
so HV / LV / HLV keep the O3a + O3b pool -- and therefore also need their own
fiducial (H1/L1/V1 over O3a+O3b+O4b), which is NOT built yet. Those runs are
deferred; see scratch/o4b_KNOWN_ISSUES.md.

__license__       = GPL-3.0-or-later
"""

# Packages
import torch

# Configs
from sage.core.config import register_configs
from sage.data.noise.real_noise import register_noise_bad_segments
from sage.core.base_classes import BaseConfig, BaseDataConfig

# Server-specific paths (one switch: SAGE_SERVER). See sage/utils/servers.py.
from sage.utils.servers import get_server
_SRV = get_server()

# O4-only line-notched fiducial ASDs (LocalLineNotch K=4), SHARED by the two O4
# HL networks. Absolute path (never home-relative). Built by build_fiducial.py:
# H1/L1 pool O4a+O4b -- the exact analogue of the O3 runs sharing a pooled O3a+O3b
# fiducial.
FIDUCIAL_DIR = "/work/nagarajan/sage_runs/fiducial_psds_o4"

# The Virgo networks train on O3a+O3b and validate on O4b, so their whitening
# reference has to pool O3a+O3b+O4b across H1/L1/V1 -- a different fiducial, NOT
# the O4-only one above. It is deliberately NOT built yet: pointing the Virgo
# networks here makes them fail loudly on a missing fiducial dir instead of
# silently training against an O4-only reference. See scratch/o4b_KNOWN_ISSUES.md.
FIDUCIAL_DIR_V1 = "/work/nagarajan/sage_runs/fiducial_psds_o3o4b"

# The O4 releases keep their derived banks in a flat <release>/data_dir; the O3
# releases use <data_release>/<run>_dataset/data_dir (Server.data_dir).
O4A_DIR = f"{_SRV.data_root}/data_release_o4a"
O4B_DIR = f"{_SRV.data_root}/data_release_o4b"


def make_configs(detectors, export_dir, norm_type="instancenorm"):
    """Build the (training, data) config classes for one detector-set network."""

    class O4bCFG:

        # --- per-network (assigned below) ---------------------------------
        # detectors, export_dir

        # --- shared production settings -----------------------------------
        fiducial_dir = FIDUCIAL_DIR_V1 if "V1" in detectors else FIDUCIAL_DIR
        batch_size = 64
        device = "cuda:0"
        dtype = torch.float32
        # Observing run(s) whose noise this model trains on (validation is the O4b
        # target, wired in the data cfg). Mirrors the noise wiring below: HL trains
        # on O4a alone, the Virgo networks on O3a+O3b (the only Virgo data there
        # is). Drives the hard-mining bank filename/metadata
        # (hardbank_<runs>_<dets>.h5) -- so the HL bank is named for O4a, the run it
        # trains on, not for its O4b target.
        train_runs = ["O3a", "O3b"] if "V1" in detectors else ["O4a"]
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
        # Bank-entry / active-set bar on the ranking stat. Raising it does NOT buy
        # tail exposure -- the bias anneal is proportional to the active count, so
        # bias*share is invariant above ~1 -- it only costs variability.
        keep_threshold_raw = 1.0
        # Focal replay weighting: draw a mined window with weight
        # sigmoid(stat)**gamma. Concentrates replay on the FAR-setting tail, which
        # the threshold alone cannot do. Bounded: <= 2**gamma spread. 0.0 = uniform.
        hard_focal_gamma = 3.0

        training_iterations = int(2_000_000 / batch_size)
        validation_iterations = int(200_000 / batch_size)

    O4bCFG.detectors = list(detectors)
    O4bCFG.export_dir = export_dir
    O4bCFG.norm_type = norm_type    # backend norm; "instancenorm" (default) | "groupnorm"

    class O4bDataCFG:

        # data_dir holds the per-segment whitening ASDs the recolour step keys by
        # (run, segment), so it must be the TRAINING run's. (Validation does not
        # recolour, so the flat validation wrap never reads it.) Same convention as
        # o3b, whose data_dir is O3b's while it validates on O3a.
        #
        # HL takes real_noise.py's flat single-run path (the one both O3 production
        # runs use); the Virgo networks take the multi-run pooled path, since O3a
        # and O3b are the only Virgo training data that exist. Whichever attribute
        # is absent simply never resolves -- _resolve_runs reads training_noise via
        # getattr and falls through to training_noise_files.
        if "V1" in detectors:
            data_dir = _SRV.data_dir("O3a")
            training_noise = [
                _SRV.run_noise("O3a", detectors, "data_release_o3a"),
                _SRV.run_noise("O3b", detectors, "data_release"),
            ]
        else:
            data_dir = f"{O4A_DIR}/data_dir"
            training_noise_files = [
                _SRV.noise_bin(d, "O4a", "data_release_o4a") for d in detectors
            ]
        # Eval on the actual target run (O4b), read raw (no recolour).
        validation_noise_files = [
            _SRV.noise_bin(d, "O4b", "data_release_o4b") for d in detectors
        ]
        sample_rate = 2048.0  # Hz
        noise_low_frequency_cutoff = 15.0  # Hz
        signal_low_frequency_cutoff = 20.0  # Hz
        sample_length_in_s = 12.0  # seconds
        padding_length_in_s = 2.0  # seconds

    return O4bCFG, O4bDataCFG


def register(detectors, export_dir, norm_type="instancenorm"):
    """Instantiate + register the (training, data) configs for one network."""
    cfg_cls, data_cfg_cls = make_configs(detectors, export_dir, norm_type)
    register_configs(BaseConfig(cfg_cls()), BaseDataConfig(data_cfg_cls()))
    register_noise_bad_segments()
    print(f"Registered cfg and data_cfg! (detectors={list(detectors)}, "
          f"export_dir={export_dir})")

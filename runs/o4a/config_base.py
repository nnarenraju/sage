#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config_base.py  (O4a)
Description     : Shared production config factory for the O4a-target networks.

The O4a search target has no Virgo (Virgo did not observe in O4a), so this run
set is HL only. Training noise is O4b ALONE, recoloured toward O4a at train time;
validation reads raw O4a. Settings are identical to the O3 production runs so the
networks stay directly comparable -- only the noise run, recolour target and
export path differ.

Why O4b alone
-------------
This mirrors the O3 production recipe, which trains each network on a SINGLE run
and validates on the other (o3a: train O3a, validate O3b). Pooling O3 into an O4
target is self-defeating: the SNR label is defined against the fiducial ASD
(OptimalSNRRescaler), so with an O4-matched fiducial an O3 window whitens to
~1.6 and an injection labelled SNR 8 lands near an effective 5 -- and with
p_recolour = 0.37 that describes 63% of the O3 windows. O4b alone supplies 25,634
H1 / 35,648 L1 segments, MORE than either completed O3 production run trained on,
and both the recoloured and the raw fraction stay on target.

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
# fiducial. Pooling both evaluated runs is what keeps each one's lines notched in
# the whitening reference while the broadband floor stays O4-matched.
#
# The O4b Virgo networks train on O3a+O3b and so need a DIFFERENT fiducial
# (H1/L1/V1 pooling O3a+O3b+O4b); that build is deferred -- see
# scratch/o4b_KNOWN_ISSUES.md.
FIDUCIAL_DIR = "/work/nagarajan/sage_runs/fiducial_psds_o4"

# The O4 releases keep their derived banks in a flat <release>/data_dir; the O3
# releases use <data_release>/<run>_dataset/data_dir (Server.data_dir).
O4A_DIR = f"{_SRV.data_root}/data_release_o4a"
O4B_DIR = f"{_SRV.data_root}/data_release_o4b"


def make_configs(detectors, export_dir, norm_type="instancenorm"):
    """Build the (training, data) config classes for one detector-set network."""

    class O4aCFG:

        # --- per-network (assigned below) ---------------------------------
        # detectors, export_dir

        # --- shared production settings -----------------------------------
        fiducial_dir = FIDUCIAL_DIR
        batch_size = 64
        device = "cuda:0"
        dtype = torch.float32
        # Observing run whose noise this model trains on (validation is the O4a
        # target, wired in the data cfg). Drives the hard-mining bank
        # filename/metadata (hardbank_<runs>_<dets>.h5) -- so this run's bank is
        # named for O4b, the run it trains on, not for its O4a target.
        train_runs = ["O4b"]
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

    O4aCFG.detectors = list(detectors)
    O4aCFG.export_dir = export_dir
    O4aCFG.norm_type = norm_type    # backend norm; "instancenorm" (default) | "groupnorm"

    class O4aDataCFG:

        # data_dir holds the per-segment whitening ASDs the recolour step keys by
        # (run, segment), so it must be the TRAINING run's -- O4b. (Validation does
        # not recolour, so the flat validation wrap never reads it.) Same convention
        # as o3a, whose data_dir is O3a's while it validates on O3b.
        data_dir = f"{O4B_DIR}/data_dir"
        # Single-run training noise: O4b, recoloured toward O4a (the eval run) in
        # train_hard.py. The flat *_noise_files form takes real_noise.py's
        # single-run path, the same one both O3 production runs use.
        training_noise_files = [
            _SRV.noise_bin(d, "O4b", "data_release_o4b") for d in detectors
        ]
        # Eval on the actual target run (O4a), read raw (no recolour).
        validation_noise_files = [
            _SRV.noise_bin(d, "O4a", "data_release_o4a") for d in detectors
        ]
        sample_rate = 2048.0  # Hz
        noise_low_frequency_cutoff = 15.0  # Hz
        signal_low_frequency_cutoff = 20.0  # Hz
        sample_length_in_s = 12.0  # seconds
        padding_length_in_s = 2.0  # seconds

    return O4aCFG, O4aDataCFG


def register(detectors, export_dir, norm_type="instancenorm"):
    """Instantiate + register the (training, data) configs for one network."""
    cfg_cls, data_cfg_cls = make_configs(detectors, export_dir, norm_type)
    register_configs(BaseConfig(cfg_cls()), BaseDataConfig(data_cfg_cls()))
    register_noise_bad_segments()
    print(f"Registered cfg and data_cfg! (detectors={list(detectors)}, "
          f"export_dir={export_dir})")

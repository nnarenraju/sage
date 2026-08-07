#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O3b HL network, GroupNorm(1,D) input norm -- A/B against config_HL (InstanceNorm).

Identical to config_HL in every other respect (config_base). This is the
notch-in-place normalisation A/B: does GroupNorm(1,D) still lose to InstanceNorm
now that the LocalLineNotch fiducial removes the line-inflated input std that
originally skewed it? Separate export dir so the two runs never share checkpoints.

__license__ = GPL-3.0-or-later
"""

from config_base import register


def set_configs():
    register(detectors=["H1", "L1"],
             export_dir="/work/nagarajan/sage_runs/o3b/prod_HL_gn",
             norm_type="groupnorm")

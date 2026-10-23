#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config_LV.py  (O4b)
Description     : L1+V1 network for the O4b search target.

All settings live in config_base.py. This network contains Virgo, so config_base
pools O3a + O3b -- O4a has no V1, and O3a/O3b are the only Virgo training data
that exist. It therefore cannot follow the single-O4-run recipe the HL networks
use, and it needs its own fiducial (H1/L1/V1 over O3a+O3b+O4b).

BLOCKED, two reasons -- see scratch/o4b_KNOWN_ISSUES.md:
  1. that fiducial (fiducial_psds_o3o4b) has not been built;
  2. the O4b V1 bank still triggers the NaN bug that corrupted the last V1
     fiducial, so it must be fixed BEFORE the build.

__license__       = GPL-3.0-or-later
"""

from config_base import register
from sage.utils.servers import get_server

EXPORT_DIR = f"{get_server().data_root}/sage_runs/o4b/production_run_LV"


def set_configs():
    register(["L1","V1"], EXPORT_DIR)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O3a LV network -- thin wrapper; all settings live in config_base.py.

__license__ = GPL-3.0-or-later
"""

from config_base import register


def set_configs():
    register(detectors=["L1", "V1"],
             export_dir="/work/nagarajan/sage_runs/o3a/prod_LV")

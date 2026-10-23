#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O4a default network (HL) -- thin wrapper; all settings live in config_base.py.

Kept as an exact alias of config_HL so `./submit.sh <task>` with no config
argument resolves to the same run as `./submit.sh <task> config_HL`, matching
runs/o3a and runs/o3b.

__license__ = GPL-3.0-or-later
"""

from config_base import register


def set_configs():
    register(detectors=["H1", "L1"],
             export_dir="/work/nagarajan/sage_runs/o4a/production_run_HL")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config_HL.py  (O4b)
Description     : H1+L1 network for the O4b search target.

O4a carries no Virgo, so only HL can pool it; the Virgo networks for this target
need per-run detector subsetting first. All settings
live in config_base.py.

__license__       = GPL-3.0-or-later
"""

from config_base import register
from sage.utils.servers import get_server

EXPORT_DIR = f"{get_server().data_root}/sage_runs/o4b/production_run_HL"


def set_configs():
    register(["H1", "L1"], EXPORT_DIR)

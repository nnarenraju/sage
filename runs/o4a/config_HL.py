#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config_HL.py  (O4a)
Description     : H1+L1 network for the O4a search target.

O4a has no Virgo, so HL is the only network this target supports. All settings
live in config_base.py.

__license__       = GPL-3.0-or-later
"""

from config_base import register
from sage.utils.servers import get_server

EXPORT_DIR = f"{get_server().data_root}/sage_runs/o4a/production_run_HL"


def set_configs():
    register(["H1", "L1"], EXPORT_DIR)

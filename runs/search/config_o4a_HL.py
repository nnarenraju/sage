#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : config_o4a_HL.py
Description   : Search campaign over O4a, Hanford-Livingston.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from config_base import make_spec

CHECKPOINT = ""
TRAINING_CONFIG = ""
FIDUCIAL_DIR = ""


def get_spec():
    """The O4a campaign specification."""
    return make_spec(
        observing_run="O4a",
        checkpoint=CHECKPOINT,
        training_config=TRAINING_CONFIG,
        fiducial_dir=FIDUCIAL_DIR,
        detectors=("H1", "L1"),
        tag="o4a_HL",
    )

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

# Not yet configured: no network has been trained on O4a. Filling these in is all this
# campaign needs -- the shape is identical to config_o3a_HL.py, which is live.
CHECKPOINT = ""
TRAINING_CONFIG = ""
FIDUCIAL_DIR = ""


def get_spec():
    """
    The O4a campaign specification.

    Raises until the three constants above are set. A spec assembled from empty paths
    fails validation with a message about an unset checkpoint, which says what is wrong
    but not why; this says why.
    """
    unset = [
        name
        for name, value in (
            ("CHECKPOINT", CHECKPOINT),
            ("TRAINING_CONFIG", TRAINING_CONFIG),
            ("FIDUCIAL_DIR", FIDUCIAL_DIR),
        )
        if not value
    ]
    if unset:
        raise NotImplementedError(
            f"the O4a campaign is not configured: {', '.join(unset)} unset in "
            f"{__file__}. No network has been trained on O4a yet; config_o3a_HL.py is "
            "the live campaign and has the same shape"
        )
    return make_spec(
        observing_run="O4a",
        checkpoint=CHECKPOINT,
        training_config=TRAINING_CONFIG,
        fiducial_dir=FIDUCIAL_DIR,
        detectors=("H1", "L1"),
        tag="o4a_HL",
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : config_o3a_HL.py
Description   : Search campaign over O3a, Hanford-Livingston.

Created on 2026-08-19

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The development campaign: a network trained on O3b, searching O3a. The two runs do not
overlap, so no strain the network was trained on is searched.

The fiducial spectra follow the network, not the strain. These are the combined O3a+O3b
spectra the network was trained under, which is what makes an O3b-trained network on O3a
data coherent rather than a distribution shift; they are not the O3a-only spectra.
"""

from config_base import make_spec

CHECKPOINT = "/work/nagarajan/sage_runs/o3b/production_run_HL/CHECKPOINTS/best.pt"
TRAINING_CONFIG = "runs/o3b/config_HL.py"
FIDUCIAL_DIR = "/work/nagarajan/sage_runs/fiducial_psds_o3ab"


def get_spec():
    """The O3a campaign specification."""
    return make_spec(
        observing_run="O3a",
        checkpoint=CHECKPOINT,
        training_config=TRAINING_CONFIG,
        fiducial_dir=FIDUCIAL_DIR,
        detectors=("H1", "L1"),
        tag="o3a_HL",
        # The release was built on the DATA flag alone. CBC_CAT1 was measured against it
        # over 600 ks of each run and coincides with DATA exactly (100.0%), so requiring
        # it here would remove nothing and only assert a veto this campaign did not apply.
        # Stated rather than defaulted: the livetime must not claim a vetoing it did not do.
        data=dict(apply_cat1=False),
    )

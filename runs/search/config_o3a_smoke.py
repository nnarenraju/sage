#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : config_o3a_smoke.py
Description   : Shallow O3a campaign, to exercise every stage end to end.

Created on 2026-08-22

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The same network, strain, population and pairing as :mod:`config_o3a_HL`, at a depth
chosen so the chain finishes in hours rather than days. It exists to answer "does every
stage run, and does what each writes satisfy the next" -- not to measure anything. A
false-alarm rate from a third of a year of background has a floor of about three per
year, which is above every event worth finding, so nothing here is a result.

Its own tag and its own directory. A shallow run sharing either with the production
campaign would overwrite that campaign's products in place while recording them under a
different hash, leaving it reporting every stage incomplete with nothing left to complete
it from.
"""

from config_base import SEARCH_ROOT, make_spec

CHECKPOINT = "/work/nagarajan/sage_runs/o3b/production_run_HL/CHECKPOINTS/best.pt"
TRAINING_CONFIG = "runs/o3b/config_HL.py"
FIDUCIAL_DIR = "/work/nagarajan/sage_runs/fiducial_psds_o3ab"


def get_spec():
    """The shakedown campaign."""
    return make_spec(
        observing_run="O3a",
        checkpoint=CHECKPOINT,
        training_config=TRAINING_CONFIG,
        fiducial_dir=FIDUCIAL_DIR,
        detectors=("H1", "L1"),
        tag="o3a_HL_smoke",
        # A third of a year: four slides of the 14.5 d foreground. Enough for the tail
        # fit to have something to fit and for every downstream stage to have a real
        # input, and small enough to finish in an afternoon.
        background_yr=0.32,
        data=dict(apply_cat1=False),
        engine=dict(use_frontend_cache=True),
        # The frozen GWOSC cache lives with the production campaign. Pointed at rather
        # than copied, and certainly rather than re-fetched: the comparison must be
        # against the same bytes, and the catalogue stage runs offline by design.
        catalogue=dict(
            cache_dir=SEARCH_ROOT / "o3a_HL" / "catalogue" / "cache"
        ),
        injection=dict(
            hyperposterior_path=SEARCH_ROOT
            / "o3a_HL"
            / "injections"
            / "hyperposterior_gwtc3_pp.json",
            # The population is drawn at its own distances with no SNR rescaling, so the
            # great majority of injections are missed by construction -- that missed
            # fraction is the sensitivity. Only the recovered tail enters p(x | signal),
            # so the count that matters is how many land near the detection threshold,
            # not how many are drawn. At 5,000 the smoke campaign put a single trigger
            # inside the fitted support, the monotonicity gate failed, and the loudest
            # candidate came out with p_astro = nan.
            #
            # 5.0 M draws keep ~4.4 M after the chirp-mass cut (87.2% measured) and cost
            # ~2.2 h at the measured 542 injections/s.
            n_draw=5_000_000,
        ),
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Injection campaign, drawn from the GWTC-3 population.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Injections are drawn here, not taken from a published sensitivity-estimate release. That
is what sgwc-1 does: ``injection_study.ipynb`` samples intrinsic parameters from the
GWTC-3 Power-Law + Peak model at its MAP hyperposterior sample, adds extrinsic parameters
from the PyCBC prior the network was trained under, and keeps the injections whose chirp
mass falls inside that prior. The set is then injected into real strain and scored by the
same engine the search uses, and its ranking statistics become ``p(x|signal)`` for
p_astro -- which is why the draw has to match the training prior rather than an external
release's.
"""

from sage.search._lazy import lazy_exports

_EXPORTS = {
    "DrawPDF": "drawpdf",
    "InjectionCampaign": "campaign",
    "InjectionRelease": "release",
    "LVKInjectionSet": "ingest",
    "sample_intrinsic": "population",
    "sample_intrinsic_torch": "population",
    "match_injections": "matching",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Injection campaign built on the public LVK injection sets.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The injected population is taken from the published sensitivity-estimate release rather
than regenerated. Using the same injections the reference pipelines were assessed on is
what makes a sensitivity comparison meaningful; a locally drawn population would only
be comparable to itself.
"""

from sage.search._lazy import lazy_exports

_EXPORTS = {
    "DrawPDF": "drawpdf",
    "InjectionCampaign": "campaign",
    "InjectionRelease": "release",
    "LVKInjectionSet": "ingest",
    "TargetPopulation": "population",
    "match_injections": "matching",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

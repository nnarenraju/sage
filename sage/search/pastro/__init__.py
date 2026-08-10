#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Astrophysical probability from a Poisson mixture over ranked triggers.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Signal and noise rates are inferred jointly from the trigger set and then marginalised
over, so p_astro carries the uncertainty on those rates rather than conditioning on a
point estimate.

The construction rests on assumptions that are checked here rather than assumed:
triggers must be independent, which requires clustering first; the ranking statistic
must order events by likelihood ratio, which is gated explicitly; and the densities must
share one support and one threshold, since a probability formed from two differently
truncated densities is decided by the truncation rather than by evidence.
"""

from sage.search._lazy import lazy_exports

_EXPORTS = {
    "Category": "categories",
    "CommonSupport": "support",
    "Density": "density",
    "PAstroTable": "assign",
    "RatePosterior": "rates",
    "assign_pastro": "assign",
    "check_monotonicity": "monotonic",
    "fit_rates": "rates",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

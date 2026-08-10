#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Per-candidate characterization.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A significance number alone does not establish a candidate. This subpackage assembles
the supporting evidence: that the data around it were behaving, that the signal is
consistent between detectors, that its inferred parameters are physical, and that an
independent method sees the same thing.

Sage estimates only coalescence time and chirp mass, so masses, spins, distance and sky
position come from separate parameter estimation. That stack is heavy and is invoked as
a subprocess in its own environment rather than imported here.
"""

from sage.search._lazy import lazy_exports

# ``consistency_tests`` is the public name for ``consistency.run_all``: the bare verb
# reads well inside its own module and says nothing at package level.
_EXPORTS = {
    "CandidateReport": "report",
    "CandidateStrain": "strain",
    "DataQualityReport": "dq",
    "characterize": "report",
    "consistency_tests": ("consistency", "run_all"),
    "followup_matched_filter": "followup_mf",
    "qscan": "qscan",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Sensitivity: VT, reference-mass points, ranges and pipeline comparison.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from sage.search._lazy import lazy_exports

# ``sensitive_distance`` previously named only a figure builder, which would have made
# this package import the figure layer. The physical quantity lives in ``ranges``; the
# builder calls it.
_EXPORTS = {
    "PipelineComparison": "compare",
    "VTResult": "vt",
    "compare_pipelines": "compare",
    "fiducial_points": "fiducial",
    "purity": "purity",
    "sensitive_distance": ("ranges", "sensitive_distance_mpc"),
    "vt_estimate": "vt",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

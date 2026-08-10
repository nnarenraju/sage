#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Production search subpackage: streaming search, background, FAR/IFAR,
                injections/VT, p_astro, candidates, catalogue comparison, figures.

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

__version__ = "0.0.1"

# Re-exports are lazy: importing sage.search must not pull torch or h5py. Asserted by
# tests/test_search_package.py, which checks it in a subprocess.
_EXPORTS = {
    "AnalysisGrid": "grid",
    "SearchDataSpec": "dataprep",
    "SearchGeometry": "geometry",
    "SearchResult": "pipeline",
    "SearchSpec": "spec",
    "Segment": "segments",
    "SlidePlan": "slides",
    "cluster_triggers": "cluster",
    "far_of_stat": "far",
    "load_search_model": "checkpoint",
    "load_segments": "segments",
    "open_store": "store",
    "run_followup": "pipeline",
    "run_search": "pipeline",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

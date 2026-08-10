#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Publication and data-release artefacts.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Everything here is generated from the candidate table and the figure data products, so
the paper, the machine-readable catalogue and the released numbers cannot disagree with
each other or with the analysis.
"""

from sage.search._lazy import lazy_exports

# ``latex_tables`` named no symbol: the tables live in ``tables.py`` as one function per
# table, so they are exported individually rather than behind an umbrella name that had
# nothing behind it.
_EXPORTS = {
    "bundle": "search_summary",
    "candidate_table": "tables",
    "catalogue_json": ("catalogue_json", "to_json"),
    "comparison_table": "tables",
    "configuration_table": "tables",
    "livetime_table": "tables",
    "search_summary": ("search_summary", "summary_table"),
    "sensitivity_table": "tables",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

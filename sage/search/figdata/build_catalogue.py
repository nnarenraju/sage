#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_catalogue.py
Description   : Figure data for catalogue comparison.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from pathlib import Path
from typing import Dict, Optional

from sage.search.figdata.product import FigData


def comparison_matrix(spec) -> FigData:
    """
    The event-by-catalogue comparison.

    Carries per-catalogue coverage alongside membership, so an event outside a
    catalogue's searched region is distinguishable from one it searched and missed.
    """
    raise NotImplementedError


def significance_agreement(spec) -> FigData:
    """
    Significance from Sage against each catalogue, for commonly found events.

    Values are kept in their native conventions with those conventions recorded, since
    significance is not directly comparable between groups.
    """
    raise NotImplementedError


def overlap_sets(spec) -> FigData:
    """Membership counts for each combination of catalogues."""
    raise NotImplementedError


def recovery_of_known_events(spec) -> FigData:
    """Which published events Sage recovers, and at what significance."""
    raise NotImplementedError


def build(spec, figures: Optional[list] = None) -> Dict[str, Path]:
    """Build every catalogue-comparison figure data product."""
    raise NotImplementedError

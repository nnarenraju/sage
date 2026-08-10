#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_significance.py
Description   : Figure data for search significance and background validity.

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


def cumulative_vs_ifar(spec) -> FigData:
    """Candidate counts against inverse false-alarm rate, with the expected band."""
    raise NotImplementedError


def statistic_distributions(spec) -> FigData:
    """Foreground and background ranking-statistic distributions."""
    raise NotImplementedError


def far_versus_statistic(spec) -> FigData:
    """
    The statistic-to-rate mapping, with the fitted tail.

    The measured range and the extrapolated range are carried separately so the figure
    can mark where counting ends and the fit begins.
    """
    raise NotImplementedError


def pastro_curves(spec) -> FigData:
    """Astrophysical probability against ranking statistic and against rate."""
    raise NotImplementedError


def foreground_rate_from_injections(spec) -> FigData:
    """Predicted foreground counts from injections, against what was observed."""
    raise NotImplementedError


def window_offset_stability(spec) -> FigData:
    """Score stability under analysis-window shifts, for a signal and a noise trigger."""
    raise NotImplementedError


def background_validity(spec) -> FigData:
    """Background self-calibration, over-dispersion and per-slide livetime retention."""
    raise NotImplementedError


def build(spec, figures: Optional[list] = None) -> Dict[str, Path]:
    """Build every significance figure data product."""
    raise NotImplementedError

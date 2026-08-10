#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : calibration.py
Description   : Background self-calibration checks.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A background estimate is only usable if background triggers themselves follow the
expected exponential distribution in IFAR. The leave-one-slide-out test measures this
directly: each slide is assessed against the FAR curve built from the others.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    """Observed versus expected cumulative background counts, with bands."""

    ifar_yr: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    max_sigma_deviation: float
    passed: bool

    def as_dict(self) -> dict:
        """Flat dict for the manifest."""
        raise NotImplementedError


def loso_calibration(
    slide_stats: Sequence[np.ndarray],
    slide_livetimes_s: Sequence[float],
    sigma_tolerance: float = 3.0,
) -> CalibrationResult:
    """Assess each slide against a FAR curve built from all the others."""
    raise NotImplementedError


def effective_independent_slides(
    slide_stats: Sequence[np.ndarray], threshold: float
) -> float:
    """
    Effective number of independent slides above a threshold.

    Falls below the nominal count when lags are correlated, which inflates the honest
    uncertainty on IFAR.
    """
    raise NotImplementedError

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : vt.py
Description   : Importance-sampled sensitive volume-time.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Sensitivity is a Monte Carlo sum over recovered injections, reweighted from the
distribution they were drawn from to the target population.

Two conventions matter and are enforced rather than documented. Injections are drawn
uniformly in detector-frame wall time, so the multiplier is total wall time, not summed
science time; duty cycle is already encoded in which injections were recovered. And the
estimate is only meaningful where the injected distribution covers the target, which is
checked before any sum is formed.

The Monte Carlo error is reported with every value. An estimate resting on too few
effective samples is flagged and excluded from published figures rather than dropped
silently.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VTResult:
    """Sensitive volume-time with its Monte Carlo uncertainty."""

    vt: float
    vt_err: float
    n_effective: float
    n_found: int
    n_generated: int
    analysis_time_s: float
    far_threshold_per_yr: float
    population: str
    relative_error: float
    plottable: bool

    @property
    def sensitive_volume(self) -> float:
        """``vt / analysis_time`` in comoving volume units."""
        raise NotImplementedError

    def as_dict(self) -> dict:
        """Flat dict for the figure data products."""
        raise NotImplementedError


def vt_estimate(
    injections,
    match,
    population,
    analysis_time_s: float,
    far_threshold_per_yr: float = 1.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
    max_relative_error: float = 0.25,
) -> VTResult:
    """
    Reweighted sensitive volume-time at a FAR threshold.

    Parameters
    ----------
    analysis_time_s : float
        Wall-clock span of the analysed observing run.
    max_relative_error : float
        Above this the result is marked not plottable; the value is still returned.
    """
    raise NotImplementedError


def effective_samples(log_weights: np.ndarray) -> float:
    """Effective sample count of an importance-weighted sum."""
    raise NotImplementedError


def check_convergence(result: VTResult, n_detected: Optional[int] = None) -> Dict[str, object]:
    """
    Report whether an estimate rests on enough effective samples.

    Returns a verdict and the numbers behind it; it does not raise, so a campaign is
    never aborted by one poorly-converged reference point.
    """
    raise NotImplementedError


def vt_vs_threshold(
    injections, match, population, analysis_time_s: float, thresholds_per_yr: Sequence[float]
) -> Dict[float, VTResult]:
    """Sensitivity as a function of the FAR threshold."""
    raise NotImplementedError


def vt_vs_parameter(
    injections,
    match,
    population,
    analysis_time_s: float,
    parameter: str,
    bin_edges: np.ndarray,
    far_threshold_per_yr: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Sensitivity binned in one intrinsic parameter."""
    raise NotImplementedError

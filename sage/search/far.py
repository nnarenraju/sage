#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : far.py
Description   : FAR and IFAR with conservative counting, plus the cumulative-vs-IFAR curve.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

FAR uses the conservative ``(1 + n_b) / T_b`` counting. ``T_b`` is always the summed
per-slide livetime from the slide plan; there is no closed form for it. Beyond the
measured background the curve is continued by the fitted tail, and the extrapolated
region is reported with its uncertainty band rather than silently.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from sage.search.background import BackgroundSet
from sage.search.tail import TailFit

SECONDS_PER_JULIAN_YEAR: float = 31557600.0


@dataclass
class FarCurve:
    """The statistic-to-FAR mapping for one observing run and removal mode."""

    stat: np.ndarray
    far_per_yr: np.ndarray
    n_louder: np.ndarray
    background_livetime_s: float
    foreground_livetime_s: float
    removal: str
    tail: Optional[TailFit] = None
    ifar_cap_yr: float = 1000.0

    def far_of(self, stat: np.ndarray) -> np.ndarray:
        """Interpolate FAR at arbitrary statistic values."""
        raise NotImplementedError

    def ifar_of(self, stat: np.ndarray) -> np.ndarray:
        """Inverse FAR in years, capped and flagged where extrapolated."""
        raise NotImplementedError

    def is_extrapolated(self, stat: np.ndarray) -> np.ndarray:
        """Whether a statistic lies beyond the measured background."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write ``far/far_curve_<run>_<removal>.h5``."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "FarCurve":
        """Read a persisted FAR curve."""
        raise NotImplementedError


def far_of_stat(stat: np.ndarray, background_stats: np.ndarray, livetime_s: float) -> np.ndarray:
    """Conservative FAR per second: ``(1 + n_b(>= stat)) / T_b``."""
    raise NotImplementedError


def build_far_curve(
    background: BackgroundSet,
    foreground_livetime_s: float,
    tail: Optional[TailFit] = None,
    ifar_cap_yr: float = 1000.0,
) -> FarCurve:
    """Assemble the FAR curve, capping IFAR relative to the measured background."""
    raise NotImplementedError


def cumulative_vs_ifar(
    foreground_stats: np.ndarray,
    curve: FarCurve,
    sigma_levels: Sequence[int] = (1, 2, 3),
) -> dict:
    """
    Cumulative count of candidates at or above each IFAR, with Poisson bands.

    The expected background curve is ``T_analysis / IFAR``; the bands are Poisson
    quantiles about it.
    """
    raise NotImplementedError


def p_value_from_ifar(ifar_yr: np.ndarray, observation_time_s: float) -> np.ndarray:
    """``1 - exp(-T / IFAR)`` for a single trial."""
    raise NotImplementedError

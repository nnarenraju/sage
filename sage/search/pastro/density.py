#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : density.py
Description   : Component densities over the ranking statistic and chirp mass.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Densities are estimated from the samples themselves, with the bandwidth derived from the
data. Smoothing a pre-binned histogram instead makes the effective resolution a property
of the binning, and tying bin edges to the observed extremes makes the whole density
move whenever more background is added, which prevents the result from converging.

Boundary correction is applied at the truncation edge so that mass is not lost there.
The network estimates chirp mass alongside the ranking statistic, so densities can be
resolved in both; the extra dimension is what lets a candidate's mass count as evidence
rather than being carried through unused.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from sage.search.pastro.support import CommonSupport


class Density(Protocol):
    """A normalised component density over the common support."""

    def log_prob(self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None) -> np.ndarray:
        """Log density at the given points."""
        ...

    def normalisation(self) -> float:
        """Integral over the common support; must be one to numerical tolerance."""
        ...


@dataclass
class TruncatedKDE:
    """Kernel density estimate, truncated and renormalised on the common support."""

    support: CommonSupport
    bandwidth: np.ndarray
    samples: np.ndarray
    weights: Optional[np.ndarray] = None
    boundary_corrected: bool = True

    def log_prob(self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None) -> np.ndarray:
        """Log density at the given points."""
        raise NotImplementedError

    def normalisation(self) -> float:
        """Integral over the common support."""
        raise NotImplementedError

    def resample_bandwidth(self, rule: str = "scott") -> np.ndarray:
        """Recompute the bandwidth from the samples."""
        raise NotImplementedError


def signal_density(
    injection_stats: np.ndarray,
    support: CommonSupport,
    injection_mchirp: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> Density:
    """
    Foreground density from recovered injections.

    Injections are reweighted to the assumed astrophysical population, so the density
    describes signals as they would arrive rather than as they were drawn.
    """
    raise NotImplementedError


def noise_density(
    background_stats: np.ndarray,
    support: CommonSupport,
    background_mchirp: Optional[np.ndarray] = None,
    tail: Optional[object] = None,
    background_livetime_s: float = 0.0,
    foreground_livetime_s: float = 0.0,
) -> Density:
    """
    Background density from time-slid triggers, blended into a fitted tail.

    The tail model is shared with the false-alarm-rate layer so the two cannot describe
    the same background differently.
    """
    raise NotImplementedError


def verify_normalisation(density: Density, atol: float = 1e-6) -> float:
    """Assert a density integrates to one over the common support."""
    raise NotImplementedError

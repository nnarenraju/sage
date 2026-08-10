#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : population.py
Description   : Target populations, expressed as densities only.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Target populations are evaluated, never sampled: sensitivity is obtained by reweighting
the published injections, so a population only needs a density. Reweighting is valid
only where the injected distribution has support everywhere the target does, and that
condition is checked rather than assumed.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class SupportViolation(ValueError):
    """Raised when a target population extends beyond the injected support."""


@dataclass
class TargetPopulation:
    """A density over the parameters used for reweighting."""

    name: str
    parameters: Tuple[str, ...]

    def log_prob(self, injections) -> np.ndarray:
        """Log target density at each injection."""
        raise NotImplementedError

    def support(self) -> Dict[str, Tuple[float, float]]:
        """Bounds of the target's support, per parameter."""
        raise NotImplementedError


@dataclass
class PowerLawPlusPeak(TargetPopulation):
    """Primary-mass power law with a Gaussian component and low-mass smoothing."""

    alpha: float = 0.0
    m_min: float = 0.0
    m_max: float = 0.0
    lam: float = 0.0
    mu_peak: float = 0.0
    sigma_peak: float = 0.0
    delta_m: float = 0.0
    beta_q: float = 0.0

    def log_prob(self, injections) -> np.ndarray:
        """Log density under the power-law-plus-peak mass model."""
        raise NotImplementedError


@dataclass
class LogNormalMassPoint(TargetPopulation):
    """
    A narrow log-normal about a fiducial component mass.

    Used to quote sensitivity at reference masses. Points whose mass support falls
    outside the network's training range are reported as such rather than plotted,
    since a reweighted number there reflects extrapolation, not measured sensitivity.
    """

    m1_ref: float = 0.0
    m2_ref: float = 0.0
    log_width: float = 0.1

    def log_prob(self, injections) -> np.ndarray:
        """Log density of the reference-mass kernel."""
        raise NotImplementedError


def check_support(
    target: TargetPopulation, scored_support: Dict[str, Tuple[float, float]]
) -> None:
    """Raise :class:`SupportViolation` when the target extends beyond what was scored."""
    raise NotImplementedError


def scored_support(injections, selection: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
    """
    Bounds of the injection subset actually scored by the search.

    Defined once for the whole campaign, before scoring, so that every later target can
    be checked against the same region.
    """
    raise NotImplementedError

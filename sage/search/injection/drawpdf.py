#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : drawpdf.py
Description   : The injected draw distribution and its Jacobians.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Importance reweighting divides a target density by the density the injections were
drawn from, so the draw density must be expressed in exactly the same variables as the
target. The release ships log draw probabilities; this module reconstructs the analytic
form so the two can be checked against each other before any weighting is trusted.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DrawPDF:
    """The documented injected distribution, factorised over parameter groups."""

    mass_pdf: object
    spin_pdf: object
    redshift_pdf: object
    angle_pdf: object

    def log_prob(self, injections, variables: Sequence[str]) -> np.ndarray:
        """Log draw density in the requested variable set."""
        raise NotImplementedError

    def validate_against_release(
        self, injections, rtol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compare the analytic reconstruction with the release's own log draw column.

        A mismatch means the variable set or a Jacobian is wrong, which would bias every
        sensitivity number silently.
        """
        raise NotImplementedError


def broken_power_law_pdf(
    m: np.ndarray, breakpoints: Sequence[float], slopes: Sequence[float]
) -> np.ndarray:
    """Normalised piecewise power law over primary mass."""
    raise NotImplementedError


def secondary_mass_pdf(m1: np.ndarray, m2: np.ndarray, m_min: float) -> np.ndarray:
    """Conditional density of the secondary mass given the primary."""
    raise NotImplementedError


def comoving_redshift_pdf(z: np.ndarray, z_max: float) -> np.ndarray:
    """Density uniform in comoving volume and source-frame time, truncated at ``z_max``."""
    raise NotImplementedError


def spin_jacobian(from_convention: str, to_convention: str, injections) -> np.ndarray:
    """Log Jacobian converting between cartesian and polar spin parameterisations."""
    raise NotImplementedError

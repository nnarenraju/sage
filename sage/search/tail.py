#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : tail.py
Description   : Peaks-over-threshold tail fitting, shared by far.py and pastro.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One tail model serves both the FAR extrapolation and the p_astro noise density. The
exponential branch is used only where the shape parameter is not distinguishable from
zero; otherwise the fitted shape is carried through, so the two consumers cannot
disagree about the same data.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TailFit:
    """A generalised-Pareto fit above a threshold, with uncertainty."""

    threshold: float
    scale: float
    shape: float
    covariance: np.ndarray
    n_exceedances: int
    exponential_preferred: bool
    lrt_p_value: float
    ad_p_value: float

    @property
    def finite_endpoint(self) -> Optional[float]:
        """Upper endpoint when the shape is negative, else ``None``."""
        raise NotImplementedError

    def survival(self, stat: np.ndarray) -> np.ndarray:
        """Exceedance probability above the fit threshold."""
        raise NotImplementedError

    def survival_band(self, stat: np.ndarray, level: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Credible band on the survival function from the fit covariance."""
        raise NotImplementedError


def choose_threshold(stats: np.ndarray, min_exceedances: int = 500) -> float:
    """Select a POT threshold by stability of the fitted shape."""
    raise NotImplementedError


def fit_tail(
    stats: np.ndarray,
    threshold: Optional[float] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> TailFit:
    """Fit a generalised Pareto tail by maximum likelihood with a bootstrap covariance."""
    raise NotImplementedError


def exponential_lrt(stats: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Likelihood-ratio test of shape == 0; returns ``(statistic, p_value)``."""
    raise NotImplementedError

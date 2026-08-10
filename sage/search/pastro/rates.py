#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : rates.py
Description   : Joint inference of the component rates.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The rates are inferred from the trigger set itself under a Poisson mixture and then
marginalised over, so the reported probability reflects how well the rates are known.

The estimator is the one derived in ``docs/references/arxiv_1302.5341.pdf`` (Farr et al.,
"Counting And Confusion"). Its Eq. (12) gives the likelihood conditioned on per-event
foreground/background flags; marginalising those flags gives the rate posterior, Eq. (21)::

    p(Rf, Rb, th | d, N)  proportional to
        prod_i [ Rf fhat(x_i, th) + Rb bhat(x_i, th) ]
        * exp[-(Rf + Rb)] * p(th) / sqrt(Rf * Rb)

The trailing ``1/sqrt(Rf Rb)`` is the Jeffreys prior on the two rates. The same posterior
appears as Eq. (10) of ``docs/references/arxiv_2305.00071.pdf`` in count form, with
``Lambda_s = R_s T`` and ``Lambda_n = R_n T`` from its Eq. (4).

A check on any implementation: in the foreground-dominated limit Eq. (35) of the same
reference reduces the posterior to ``Rf^(N - 1/2) exp(-Rf)``, peaked at ``Rf = N - 1/2``,
where the half is contributed by the Jeffreys prior.

The inference is parameterised by the total rate and the fraction belonging to each
component. Working directly in the individual rates loses precision when one component
outnumbers the other by many orders of magnitude, which is the normal situation here; the
change of variables carries a Jacobian that must be applied with the prior.

The likelihood assumes independent triggers, so the input must already be clustered; the
constructor refuses an unclustered set rather than silently producing a rate inflated by
the number of windows per event. The grid is bracketed automatically from the data, so
the answer cannot depend on a hand-chosen range.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RatePosterior:
    """Posterior over the component rates."""

    categories: Tuple[str, ...]
    total_grid: np.ndarray
    fraction_grid: np.ndarray
    log_posterior: np.ndarray
    n_triggers: int
    clustered: bool
    prior: str

    @property
    def map_rates(self) -> Dict[str, float]:
        """Rates at the posterior mode."""
        raise NotImplementedError

    def mean_rates(self) -> Dict[str, float]:
        """Posterior mean of each component rate."""
        raise NotImplementedError

    def credible_interval(self, category: str, level: float = 0.9) -> Tuple[float, float]:
        """Credible interval on one component's rate."""
        raise NotImplementedError

    def marginal(self, category: str) -> Tuple[np.ndarray, np.ndarray]:
        """Marginal posterior for one component rate, over its own axis."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write the rate posterior and its provenance."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "RatePosterior":
        """Read a persisted rate posterior."""
        raise NotImplementedError


def fit_rates(
    stats: np.ndarray,
    densities: Dict[str, object],
    support,
    mchirp: Optional[np.ndarray] = None,
    clustered: bool = False,
    prior: str = "jeffreys",
    n_grid: int = 512,
) -> RatePosterior:
    """
    Infer the component rates from the observed triggers.

    Parameters
    ----------
    clustered : bool
        Must be true. The mixture likelihood treats triggers as independent draws.
    prior : str
        Scale-invariant by default, with the change-of-variable factor for the
        parameterisation applied consistently.
    """
    raise NotImplementedError


def bracket_grid(
    stats: np.ndarray, densities: Dict[str, object], n_grid: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose grid ranges wide enough to contain the posterior, from the data."""
    raise NotImplementedError


def log_prior(total: np.ndarray, fraction: np.ndarray, kind: str = "jeffreys") -> np.ndarray:
    """Log prior in the total-and-fraction parameterisation, including its Jacobian."""
    raise NotImplementedError

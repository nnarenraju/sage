#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : assign.py
Description   : Per-trigger astrophysical probability.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Each trigger's probability is the component's share of the mixture at its position,
averaged over the rate posterior, following Eq. (11) of
``docs/references/arxiv_2305.00071.pdf``::

    p_astro(x) = int dLs dLn  [ Ls p(x|S) / ( Ls p(x|S) + Ln p(x|0) ) ]
                              * p(Ls, Ln | {x}, N)

The average is taken over the full rate grid rather than at a point estimate, which is
what produces a credible interval alongside the value. Section V of that reference adopts
a preliminary cut of one false alarm per half day when applying this to real triggers;
the equivalent threshold here is set once in :mod:`sage.search.pastro.support` and shared
by every component density.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from sage.search.pastro.rates import RatePosterior


@dataclass
class PAstroTable:
    """Per-trigger probabilities with credible intervals."""

    gps: np.ndarray
    stat: np.ndarray
    mchirp: Optional[np.ndarray]
    probabilities: Dict[str, np.ndarray]
    lower: Dict[str, np.ndarray]
    upper: Dict[str, np.ndarray]
    attrs: Dict[str, object]

    def __len__(self) -> int:
        """Number of triggers."""
        raise NotImplementedError

    def astrophysical(self) -> np.ndarray:
        """Summed probability over the astrophysical components."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write the per-trigger table."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "PAstroTable":
        """Read a persisted table."""
        raise NotImplementedError


def assign_pastro(
    stats: np.ndarray,
    densities: Dict[str, object],
    posterior: RatePosterior,
    mchirp: Optional[np.ndarray] = None,
    credible_level: float = 0.9,
) -> PAstroTable:
    """Evaluate each trigger's component probabilities, marginalised over the rates."""
    raise NotImplementedError


def sum_consistency(table: PAstroTable, posterior: RatePosterior) -> Dict[str, float]:
    """
    Compare the summed probability against the inferred rate.

    Summing p_astro over the analysed set should recover the inferred signal rate; a
    disagreement means the densities and the rate inference describe different data.
    """
    raise NotImplementedError

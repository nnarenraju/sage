#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : support.py
Description   : The shared threshold, support and quadrature grid.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every density in the mixture is truncated at the same threshold, renormalised over the
same region and evaluated on the same grid. Truncating one component but not another
makes the ratio above the untruncated region a property of the truncation rather than of
the data, and the resulting probability saturates for reasons unrelated to evidence.

The threshold is expressed in false-alarm-rate units so it means the same thing across
observing runs and across changes to the ranking statistic's scale.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CommonSupport:
    """The region and grid over which every component density is defined."""

    stat_lo: float
    stat_hi: float
    n_stat: int
    mchirp_lo: Optional[float] = None
    mchirp_hi: Optional[float] = None
    n_mchirp: Optional[int] = None
    threshold_far_per_day: float = 2.0
    threshold_stat: float = 0.0

    @property
    def is_2d(self) -> bool:
        """Whether the densities resolve chirp mass as well as ranking statistic."""
        raise NotImplementedError

    def grid(self) -> Tuple[np.ndarray, ...]:
        """Quadrature nodes for each axis."""
        raise NotImplementedError

    def cell_volume(self) -> np.ndarray:
        """Quadrature weights matching :meth:`grid`."""
        raise NotImplementedError

    def contains(self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None) -> np.ndarray:
        """Mask of points inside the support."""
        raise NotImplementedError


def build_support(
    far_curve,
    threshold_far_per_day: float = 2.0,
    stat_pad: float = 1.0,
    n_stat: int = 512,
    mchirp_bounds: Optional[Tuple[float, float]] = None,
    n_mchirp: Optional[int] = None,
) -> CommonSupport:
    """Derive the shared support from the analysis threshold and observed range."""
    raise NotImplementedError


def stat_at_far(far_curve, far_per_day: float) -> float:
    """Ranking statistic corresponding to a false-alarm rate."""
    raise NotImplementedError

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : slides.py
Description   : Time-slide ladder generation and exact per-slide livetime.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Lags are stratified over ``[min_separation, tau_max]`` rather than packed against the
minimum. Packing them re-pairs one loud glitch against nearly the same stretch of the
other detector on every slide, at precisely the lag scale where detector noise is most
correlated, which inflates the effective number of independent background samples.

A slide is a lag **per detector relative to a reference**, so a network of ``D`` detectors
has ``D - 1`` independent lags and the ladder is a lattice in that many dimensions. Two
consequences follow, and the second is easy to miss:

* more detectors give many more distinct slides for the same ``tau_max``, so a
  three-detector background is cheaper per year of background than a two-detector one,
  even though each slide retains less livetime;
* the minimum separation has to hold for **every pair**, which for three detectors
  includes the difference between two lagged detectors. An implementation written for two
  detectors only ever checks lags against the reference and will happily emit a slide in
  which Livingston and Virgo sit within a light-travel time of each other, quietly
  admitting genuine coincidences into the background.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from sage.search.geometry import SearchGeometry


@dataclass(frozen=True)
class Slide:
    """
    One lag assignment. ``slide_id == 0`` is zero-lag.

    ``offsets_s`` maps every detector in the network to its lag, including the reference
    at zero, so a slide is self-describing and no caller has to know which detector was
    held fixed.
    """

    slide_id: int
    offsets_s: dict
    n_windows: int
    livetime_s: float


@dataclass
class SlidePlan:
    """
    The full ladder for one run, with per-slide livetime measured, not derived.

    Background livetime is always ``sum(slide.livetime_s)``. Per-slide retention falls
    with lag, so ``n_slides * T_zerolag`` is never a valid substitute.
    """

    slides: List[Slide]
    reference_detector: str
    seed: int
    min_separation_s: float
    tau_max_s: float

    @classmethod
    def build(
        cls,
        geometry: SearchGeometry,
        segments_by_detector: dict,
        n_slides: int,
        reference_detector: str = "H1",
        min_separation_s: float = 20.0,
        tau_max_s: float = 8192.0,
        guard_s: float = 4.0,
        seed: int = 0,
    ) -> "SlidePlan":
        """Draw stratified lags and measure each slide's coincident livetime."""
        raise NotImplementedError

    @property
    def background_livetime_s(self) -> float:
        """Exact ``T_b``: the sum of per-slide livetimes."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Slide]:
        """Iterate slides in id order."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write ``slides/slide_plan.h5``, including the frozen keep threshold."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "SlidePlan":
        """Read a persisted slide plan."""
        raise NotImplementedError


def stratified_lags(
    n_slides: int,
    n_lagged_detectors: int,
    min_separation_s: float,
    tau_max_s: float,
    stride_samples: int,
    sample_rate: float,
    seed: int,
) -> np.ndarray:
    """
    Draw ``n_slides`` lag vectors stratified over ``[min_separation, tau_max]``.

    Parameters
    ----------
    n_lagged_detectors : int
        One fewer than the network size: the reference detector is never slid.

    Returns
    -------
    ndarray
        ``(n_slides, n_lagged_detectors)`` of lags in seconds.

    Notes
    -----
    Lags are multiples of the stride, so a slid window lands on the same lattice and no
    resampling is implied. Zero lag is excluded, and every drawn vector satisfies
    :func:`pairwise_separations_ok`, which is the constraint that distinguishes a network
    of three from a network of two.
    """
    raise NotImplementedError


def pairwise_separations_ok(lags: np.ndarray, min_separation_s: float) -> np.ndarray:
    """
    Whether each lag vector keeps every detector pair apart.

    Checks the lags themselves, which separate each slid detector from the reference, and
    every difference between them, which separates the slid detectors from each other. A
    vector failing either would place two detectors close enough in slid time for a real
    coincidence to survive into the background.

    Parameters
    ----------
    lags : ndarray
        ``(n_slides, n_lagged_detectors)``, or ``(n_lagged_detectors,)`` for one slide.

    Returns
    -------
    ndarray
        Boolean, one entry per slide.
    """
    raise NotImplementedError


def minimum_separation_s(
    geometry: SearchGeometry, detectors: Sequence[str], guard_s: float
) -> float:
    """
    Smallest admissible lag: window content + light travel + guard.

    The light-travel term is the maximum over every detector pair, so adding Virgo to a
    two-detector network raises the floor from about 10 ms to about 27 ms.
    """
    raise NotImplementedError

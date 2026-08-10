#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : ranges.py
Description   : Detector range, horizon distance and surveyed time-volume.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Range summarises instrument sensitivity over time and provides the horizontal axis for
presenting detections against surveyed volume rather than calendar time.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class RangeSeries:
    """Range as a function of time for one detector."""

    detector: str
    gps: np.ndarray
    range_mpc: np.ndarray

    def median(self) -> float:
        """Median range over the run."""
        raise NotImplementedError

    def duty_cycle(self) -> float:
        """Fraction of the run with usable data."""
        raise NotImplementedError


def inspiral_range_mpc(asd, m1: float = 1.4, m2: float = 1.4, snr_threshold: float = 8.0) -> float:
    """Sky- and orientation-averaged range for a fiducial binary."""
    raise NotImplementedError


def horizon_distance_mpc(asd, m1: float, m2: float, snr_threshold: float = 8.0) -> float:
    """Optimally oriented and located distance at the threshold signal-to-noise ratio."""
    raise NotImplementedError


def range_time_series(release_dir, detector: str, run: str, cadence_s: float = 600.0) -> RangeSeries:
    """Estimate range at a regular cadence across an observing run."""
    raise NotImplementedError


def surveyed_time_volume(
    ranges: Sequence[RangeSeries], coincident_intervals: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """
    Cumulative surveyed time-volume.

    Uses the second most sensitive detector's range, so the measure reflects the
    coincident network rather than the best single instrument.
    """
    raise NotImplementedError


def sensitive_distance_mpc(vt: float, analysis_time_s: float) -> float:
    """
    Radius of the sphere whose volume-time equals a measured sensitive volume-time.

    A distance is easier to compare against a detector range than a volume-time is, and
    it is the quantity conventionally quoted for a search. Defined by
    ``VT = (4/3) pi D^3 T``, so ``D = (3 VT / (4 pi T))^(1/3)``.

    Parameters
    ----------
    vt : float
        Sensitive volume-time, in Mpc^3 yr.
    analysis_time_s : float
        Analysed time the volume-time was measured over.

    Returns
    -------
    float
        Sensitive distance in Mpc.

    Notes
    -----
    Lives here rather than in the figure layer because it is a physical quantity that the
    tables, the candidate store and the figures all quote. A copy inside a figure builder
    would put an analysis result somewhere that is meant to compute nothing.
    """
    raise NotImplementedError

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : qscan.py
Description   : Constant-Q spectrograms around a candidate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A spectrogram shows whether the excess power looks like a coalescence or like a known
instrumental artefact. Several durations are produced at a fixed quality factor, because
morphology that distinguishes the two is not visible at a single time scale.

The transform is computed and stored here; drawing happens in the plotting layer from
the stored array, so the figure can be redrawn without refetching data. At the signal
strengths a search operates near, a track is often not visible by eye, and the
reconstruction overlay carries the argument instead.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DURATIONS_S: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
DEFAULT_Q: float = 20.0
DEFAULT_FRANGE: Tuple[float, float] = (20.0, 512.0)


@dataclass
class QScan:
    """A time-frequency map for one detector and duration."""

    detector: str
    duration_s: float
    q: float
    times: np.ndarray
    frequencies: np.ndarray
    energy: np.ndarray
    gps: float

    def peak(self) -> Tuple[float, float, float]:
        """Location and value of the maximum."""
        raise NotImplementedError


def qscan(
    strain,
    detector: str,
    gps: float,
    duration_s: float = 1.0,
    q: float = DEFAULT_Q,
    frange: Tuple[float, float] = DEFAULT_FRANGE,
    whiten: bool = True,
) -> QScan:
    """Compute one constant-Q spectrogram."""
    raise NotImplementedError


def qscan_panel(
    strain,
    detectors: Sequence[str],
    gps: float,
    durations_s: Sequence[float] = DEFAULT_DURATIONS_S,
    q: float = DEFAULT_Q,
) -> Dict[Tuple[str, float], QScan]:
    """Compute the multi-duration set for every detector."""
    raise NotImplementedError


def expected_track(
    chirp_mass: float, gps: float, f_low: float = 20.0, f_high: float = 512.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frequency evolution implied by a chirp mass.

    Overlaid on the spectrogram so the visible structure can be compared against what
    the recovered parameters predict.
    """
    raise NotImplementedError

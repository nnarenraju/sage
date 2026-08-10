#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : dq.py
Description   : Data-quality assessment around a candidate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Each task returns a probability that the observed data quality could arise by chance, so
the verdict is a stated threshold on those probabilities rather than a judgement call.

Times flagged as unusable are removed before the search runs, so a surviving candidate
has already passed that filter; what remains is to ask whether the data around it were
behaving well enough for its significance to mean what it says.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

DEFAULT_P_THRESHOLD: float = 0.05


@dataclass
class DQTask:
    """One check and its outcome."""

    name: str
    p_value: float
    passed: bool
    detail: Dict[str, object]


@dataclass
class DataQualityReport:
    """Combined verdict for one candidate."""

    gps: float
    tasks: Dict[str, DQTask]
    vetoed: bool
    min_p_value: float
    threshold: float = DEFAULT_P_THRESHOLD

    def failures(self) -> Tuple[str, ...]:
        """Names of the tasks below threshold."""
        raise NotImplementedError

    def as_dict(self) -> dict:
        """Flat dict for the candidate table."""
        raise NotImplementedError


def stationarity(strain, detector: str, window_s: float = 64.0) -> DQTask:
    """Compare short- and long-baseline spectra for a change in the noise level."""
    raise NotImplementedError


def excess_power(strain, detector: str, window_s: float = 2.0) -> DQTask:
    """Test for transient excess power near the candidate beyond the Gaussian expectation."""
    raise NotImplementedError


def nearby_transients(
    gps: float, detector: str, window_s: float = 100.0, cache=None
) -> DQTask:
    """Look for catalogued transients close to the candidate time."""
    raise NotImplementedError


def glitch_classification(strain, detector: str) -> DQTask:
    """Classify the local time-frequency morphology against known glitch families."""
    raise NotImplementedError


def auxiliary_witness(gps: float, detector: str, cache=None) -> DQTask:
    """Check whether auxiliary channels indicate an instrumental origin."""
    raise NotImplementedError


def observing_state(gps: float, detectors: Sequence[str], cache=None) -> DQTask:
    """Confirm every detector was observing nominally at the candidate time."""
    raise NotImplementedError


def assess(
    strain,
    gps: float,
    detectors: Sequence[str],
    threshold: float = DEFAULT_P_THRESHOLD,
    cache=None,
) -> DataQualityReport:
    """Run every task and combine them into a verdict."""
    raise NotImplementedError

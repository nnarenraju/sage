#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : matching.py
Description   : Associate search triggers with injections.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The matching rule follows the published sensitivity-estimate procedure: search events
coincident with a real catalogue candidate are removed first, each remaining event is
attributed to the nearest injection in time, and where one injection collects several
events the most significant is kept.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class InjectionMatch:
    """Found/missed outcome and recovered quantities per injection."""

    inj_id: np.ndarray
    found: np.ndarray
    far_per_yr: np.ndarray
    stat: np.ndarray
    rec_gps: np.ndarray
    dt_s: np.ndarray
    rec_mchirp: np.ndarray

    @property
    def n_found(self) -> int:
        """Number of injections recovered at the configured threshold."""
        raise NotImplementedError

    def at_threshold(self, far_per_yr: float) -> np.ndarray:
        """Found mask at an arbitrary FAR threshold."""
        raise NotImplementedError


def match_injections(
    injections,
    triggers,
    far_curve,
    known_events_gps: Optional[np.ndarray] = None,
    exclusion_window_s: float = 0.25,
    found_far_yr: float = 1.0,
) -> InjectionMatch:
    """Apply the removal, nearest-in-time and best-of-duplicates rules in order."""
    raise NotImplementedError


def time_residual_check(match: InjectionMatch, tolerance_s: float = 0.1) -> dict:
    """
    Summarise the recovered-minus-injected time residual.

    A biased or broad residual indicates a timing convention error upstream and
    invalidates the matching, so this runs as a tripwire rather than a diagnostic.
    """
    raise NotImplementedError

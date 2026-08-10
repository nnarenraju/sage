#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : ood.py
Description   : In-distribution / out-of-distribution classification of events.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

An event is in-distribution when enough of its mass posterior lies inside the box the
network was trained on. The fraction is computed from a random subsample of posterior
samples rather than a truncation of the sample list, since sample files are not always
stored in a random order.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OODResult:
    """Posterior mass inside the trained region."""

    id_fraction: float
    n_samples: int
    frame: str
    is_ood: bool

    def as_dict(self) -> dict:
        """Flat dict for the candidate table."""
        raise NotImplementedError


def id_fraction(
    mass1: np.ndarray,
    mass2: np.ndarray,
    box: Tuple[float, float] = (7.0, 50.0),
    n_subsample: Optional[int] = 10000,
    seed: int = 0,
) -> OODResult:
    """Fraction of posterior mass with both components inside the trained box."""
    raise NotImplementedError


def classify_event(
    posterior,
    box: Tuple[float, float] = (7.0, 50.0),
    frame: str = "detector",
    threshold: float = 0.5,
) -> OODResult:
    """
    Classify one event from its posterior samples.

    The frame matters: the network is trained on detector-frame masses, so a
    source-frame comparison mislabels redshifted events. Both are computed and reported.
    """
    raise NotImplementedError


def read_posterior_masses(path, frame: str = "detector") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load component masses from a posterior file.

    Raises on a missing or unreadable dataset rather than returning empty arrays, so a
    read failure cannot be mistaken for an out-of-distribution verdict.
    """
    raise NotImplementedError

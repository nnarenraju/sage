#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : posteriors.py
Description   : Fetch and read published posterior samples.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Posterior files are read directly rather than through an analysis library, so ingest
does not depend on the parameter-estimation stack being installed.

Release files usually contain several analyses plus a combined set; the combined one is
used unless a specific waveform is requested. Some sources ship importance weights,
which must be applied before the samples represent the posterior.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class PosteriorUnavailable(LookupError):
    """Raised when an event has no readable posterior."""


@dataclass
class Posterior:
    """Equal-weight posterior samples for one event."""

    event: str
    source: str
    analysis: str
    samples: Dict[str, np.ndarray]
    weights: Optional[np.ndarray] = None

    def __len__(self) -> int:
        """Number of samples."""
        raise NotImplementedError

    def masses(self, frame: str = "source") -> Tuple[np.ndarray, np.ndarray]:
        """Component masses in the requested frame."""
        raise NotImplementedError

    def to_equal_weight(self, seed: int = 0) -> "Posterior":
        """Resample to equal weight where importance weights are present."""
        raise NotImplementedError

    def credible_interval(self, parameter: str, level: float = 0.9) -> Tuple[float, float]:
        """Credible interval on one parameter."""
        raise NotImplementedError


def fetch(event: str, cache, catalogue: Optional[str] = None) -> Path:
    """Download an event's posterior release into the cache."""
    raise NotImplementedError


def read(path: str | Path, analysis: Optional[str] = None) -> Posterior:
    """Read posterior samples, preferring the combined analysis."""
    raise NotImplementedError


def available_analyses(path: str | Path) -> Tuple[str, ...]:
    """List the analyses present in a posterior file."""
    raise NotImplementedError

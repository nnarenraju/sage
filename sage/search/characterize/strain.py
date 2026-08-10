#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : strain.py
Description   : Fetch and condition strain around a candidate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every characterization task works from one fetch per candidate, so the spectrogram, the
data-quality checks, the follow-up filter and the reconstruction all describe the same
data. Re-fetching per task invites conditioning differences that then show up as
disagreements between the tasks.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CandidateStrain:
    """Conditioned strain and spectra around one candidate."""

    gps: float
    detectors: Tuple[str, ...]
    sample_rate: float
    data: Dict[str, np.ndarray]
    t0: float
    psd: Optional[Dict[str, np.ndarray]] = None
    source: str = ""

    def segment(self, detector: str, dt: float) -> np.ndarray:
        """Extract a symmetric window about the candidate."""
        raise NotImplementedError

    def whitened(self, detector: str) -> np.ndarray:
        """Whiten with the stored spectrum."""
        raise NotImplementedError


def fetch(
    gps: float,
    detectors: Sequence[str],
    half_width_s: float = 256.0,
    sample_rate: float = 2048.0,
    source: str = "release",
    cache=None,
) -> CandidateStrain:
    """
    Load strain around a candidate.

    Prefers the local release so that characterization sees the same samples the search
    scored, falling back to open data where the release does not cover the time.
    """
    raise NotImplementedError


def estimate_psd(
    strain: CandidateStrain, detector: str, exclude_s: float = 8.0, method: str = "median"
) -> np.ndarray:
    """
    Estimate the spectrum from data around, but not on, the candidate.

    A median average limits the influence of unrelated transients in the estimation
    window, and the candidate itself is excluded so a loud signal does not raise its own
    noise floor.
    """
    raise NotImplementedError

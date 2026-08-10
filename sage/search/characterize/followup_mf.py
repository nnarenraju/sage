#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : followup_mf.py
Description   : Matched-filter follow-up at the inferred parameters.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A network score is a single number with no internal structure to interrogate. Filtering
the same data against a template bank around the inferred parameters recovers the
quantities that established searches use to separate signals from artefacts: how well
the data match a coalescence, how the match is distributed across frequency, and whether
the two detectors agree.

It also produces the signal-to-noise time series that sky localisation needs, which a
network score alone does not provide.

The recovered template's parameters describe the best-matching filter, not the source;
they are reported as such and never presented as parameter estimates.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FollowupResult:
    """Matched-filter quantities for one candidate."""

    gps: float
    detectors: Tuple[str, ...]
    snr_series: Dict[str, np.ndarray]
    snr_times: Dict[str, np.ndarray]
    peak_snr: Dict[str, float]
    peak_time: Dict[str, float]
    chisq: Dict[str, float]
    chisq_dof: Dict[str, int]
    reweighted_snr: Dict[str, float]
    network_snr: float
    template: Dict[str, float]

    def time_delay_s(self, a: str, b: str) -> float:
        """Arrival-time difference between two detectors."""
        raise NotImplementedError

    def consistent_with_light_travel(self, tolerance_s: float = 0.005) -> bool:
        """Whether the arrival-time difference is physically allowed."""
        raise NotImplementedError


def narrow_bank(
    chirp_mass: float, chirp_mass_sigma: float, n_templates: int = 32, mass_ratio_range=(1.0, 8.0)
) -> Sequence[Dict[str, float]]:
    """Build a small template bank around the inferred chirp mass."""
    raise NotImplementedError


def followup_matched_filter(
    strain,
    detectors: Sequence[str],
    gps: float,
    chirp_mass: float,
    chirp_mass_sigma: float,
    search_window_s: float = 0.5,
    f_low: float = 20.0,
) -> FollowupResult:
    """Filter the candidate against a narrow bank and return the best match."""
    raise NotImplementedError


def signal_consistency_chisq(strain, detector: str, template, n_bins: int = 16) -> Tuple[float, int]:
    """
    Test whether the match accumulates across frequency as a real signal would.

    A transient that is loud but unlike a coalescence produces its match in a few bands
    rather than steadily, which this separates.
    """
    raise NotImplementedError

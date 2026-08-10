#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : skymap.py
Description   : Sky localisation for a candidate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Localisation can be built either from the follow-up filter's signal-to-noise time series
or from posterior samples once sampling has run. The first is available immediately; the
second is the one quoted alongside parameter estimates.

Two detectors localise to a broad ring rather than a compact region, which is reported
plainly rather than presented as a tighter constraint than the network supports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence


@dataclass
class SkymapResult:
    """A localisation and its summary statistics."""

    candidate: str
    path: Path
    area_50_deg2: float
    area_90_deg2: float
    distance_mean_mpc: float
    distance_std_mpc: float
    method: str

    def as_dict(self) -> dict:
        """Flat dict for tables."""
        raise NotImplementedError


def from_snr_series(followup, outdir: str | Path) -> SkymapResult:
    """Localise from the follow-up filter's signal-to-noise time series and spectra."""
    raise NotImplementedError


def from_posterior(pe_result, outdir: str | Path) -> SkymapResult:
    """Localise from posterior samples."""
    raise NotImplementedError


def summarise(path: str | Path) -> Dict[str, float]:
    """Credible areas, distance and searched area for a stored localisation."""
    raise NotImplementedError

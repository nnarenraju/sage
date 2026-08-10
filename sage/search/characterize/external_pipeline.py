#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : external_pipeline.py
Description   : Independent confirmation with an established pipeline.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A candidate seen by an independent method with different failure modes is far stronger
than one seen by a single pipeline, and correlated significance across methods is what
established analyses look for.

A short run over one stretch of data yields a significance relative to that stretch, not
a rate calibrated against a full background, so its result is reported as corroboration
with its scope stated rather than as a second calibrated rate.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence


@dataclass
class ExternalPipelineResult:
    """What an independent pipeline recovers at a candidate time."""

    candidate: str
    pipeline: str
    found: bool
    network_snr: Optional[float]
    reweighted_snr: Optional[float]
    template: Dict[str, float]
    far_per_yr: Optional[float]
    background_scope: str
    note: str = ""

    def as_dict(self) -> dict:
        """Flat dict for the comparison table."""
        raise NotImplementedError


def run_pipeline(
    candidate,
    strain,
    pipeline: str = "pycbc",
    chunk_s: float = 4096.0,
    outdir: Optional[Path] = None,
) -> ExternalPipelineResult:
    """Analyse the data around a candidate with an independent pipeline."""
    raise NotImplementedError


def recover_known_events(
    events, pipeline: str = "pycbc"
) -> Dict[str, ExternalPipelineResult]:
    """
    Confirm the follow-up configuration recovers known events.

    Establishes that a non-detection at a candidate time reflects the data rather than a
    misconfigured follow-up.
    """
    raise NotImplementedError

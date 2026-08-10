#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : report.py
Description   : Per-candidate characterization driver.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Assembles the evidence for one candidate into a single record. The data-quality verdict
produced here is what completes the candidate table, since the confident tier depends on
both significance and a clean verdict.

Cost is graded: the cheap checks run for every candidate on the public list, while full
sampling is reserved for those that pass the significance bar.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence


@dataclass
class CandidateReport:
    """Everything gathered about one candidate."""

    candidate: str
    gps: float
    dq: object
    qscans: Dict
    followup: Optional[object]
    consistency: Dict
    external: Optional[object]
    pe: Optional[object]
    skymap: Optional[object]
    ood: Optional[object]

    def verdict(self) -> Dict[str, object]:
        """Combined assessment, including the data-quality outcome."""
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        """Write the record and its figure data."""
        raise NotImplementedError

    def to_markdown(self, path: str | Path) -> Path:
        """Write a readable per-candidate summary."""
        raise NotImplementedError


def characterize(
    candidate,
    spec,
    level: str = "screen",
    outdir: Optional[Path] = None,
) -> CandidateReport:
    """
    Characterize one candidate.

    Parameters
    ----------
    level : {"screen", "full"}
        Screening covers data quality, spectrograms, the follow-up filter and the
        consistency tests. The full level adds parameter estimation, localisation and
        independent-pipeline confirmation.
    """
    raise NotImplementedError


def characterize_all(
    candidates, spec, screen_tier: int = 0, full_tier: int = 1
) -> Dict[str, CandidateReport]:
    """Characterize a candidate list, choosing the level from each candidate's tier."""
    raise NotImplementedError

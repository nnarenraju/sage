#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : pe.py
Description   : Parameter estimation for a candidate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Masses, spins, distance and sky position come from stochastic sampling of the full
likelihood; the network's own outputs are point estimates of two quantities and are used
only to set the analysis window and priors.

More than one waveform model is run and the results combined with equal weight, so that
quoted intervals include the systematic difference between models rather than only the
statistical width of one.

The sampling stack is invoked as a subprocess in its own environment. Faster likelihood
approximations and learned samplers are supported as cross-checks, but the quoted result
comes from full sampling, which is what published analyses are compared against.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class PESettings:
    """Sampler, waveform and prior configuration."""

    waveforms: Sequence[str] = ()
    sampler: str = "dynesty"
    nlive: int = 1000
    prior_masses: str = "detector-frame-uniform"
    prior_distance: str = "comoving-volume-and-time"
    marginalise_calibration: bool = True
    relative_binning: bool = False
    environment: str = ""


@dataclass
class PEResult:
    """Posterior samples and evidence for one candidate."""

    candidate: str
    gps: float
    outdir: Path
    per_waveform: Dict[str, Path]
    combined: Optional[Path]
    log_evidence: Dict[str, float]
    settings: PESettings

    def summary(self, level: float = 0.9) -> Dict[str, tuple]:
        """Median and credible interval for the reported parameters."""
        raise NotImplementedError

    def model_agreement(self) -> Dict[str, float]:
        """Spread between waveform models, as a systematic-uncertainty indicator."""
        raise NotImplementedError


def write_config(candidate, strain, settings: PESettings, outdir: str | Path) -> Path:
    """Write the sampler configuration for one candidate."""
    raise NotImplementedError


def submit(config: str | Path, settings: PESettings, wait: bool = False) -> PEResult:
    """Launch sampling as a job in the parameter-estimation environment."""
    raise NotImplementedError


def combine_waveforms(results: Sequence[Path], outdir: str | Path) -> Path:
    """Combine per-waveform posteriors with equal weight."""
    raise NotImplementedError


def cross_check_fast_likelihood(result: PEResult, tolerance: float = 0.1) -> Dict[str, float]:
    """
    Compare an approximated likelihood against full evaluation on one candidate.

    Run once per campaign to justify using the faster path elsewhere.
    """
    raise NotImplementedError

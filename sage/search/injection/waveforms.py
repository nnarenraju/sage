#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : waveforms.py
Description   : Waveform generation and detector projection for injections.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Waveforms are generated to match the settings under which the published injection set
was defined, so that a recovered injection means the same thing here as in the
reference analyses. Sage's own GPU approximants are used where they are verified
against the reference implementation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WaveformSettings:
    """Generation settings pinned to the injection release."""

    approximant: str = "IMRPhenomXPHM"
    f_ref: float = 16.0
    f_lower: float = 10.0
    f_final: float = 8192.0
    sample_rate: float = 2048.0
    multibanding: bool = False


class InjectionGenerator:
    """Generate polarisations for a batch of injections."""

    def __init__(self, settings: WaveformSettings, device: str = "cuda") -> None:
        raise NotImplementedError

    def generate(self, params: Dict[str, np.ndarray]):
        """Return frequency-domain plus and cross polarisations."""
        raise NotImplementedError

    def optimal_snr(self, params: Dict[str, np.ndarray], asds) -> np.ndarray:
        """Network optimal SNR against the reference spectra."""
        raise NotImplementedError


class ExactProjection:
    """Project polarisations onto detectors with full time-delay and antenna response."""

    def __init__(self, detectors: Sequence[str]) -> None:
        raise NotImplementedError

    def project(
        self,
        hp,
        hc,
        ra: np.ndarray,
        dec: np.ndarray,
        psi: np.ndarray,
        gps: np.ndarray,
    ):
        """Return per-detector strain, including light-travel delays."""
        raise NotImplementedError

    def time_delay_s(self, detector: str, ra: np.ndarray, dec: np.ndarray, gps: np.ndarray) -> np.ndarray:
        """Geocentre-to-detector arrival-time delay."""
        raise NotImplementedError

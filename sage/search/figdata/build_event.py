#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_event.py
Description   : Figure data for per-candidate figures.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Per-candidate products are written once per candidate and read by both the individual
figures and the composite event page, so the two cannot disagree.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

from sage.search.figdata.product import FigData


def spectrograms(spec, candidate) -> FigData:
    """Multi-duration time-frequency maps per detector, with the expected track."""
    raise NotImplementedError


def whitened_reconstruction(spec, candidate) -> FigData:
    """Whitened data, the recovered model and the residual, per detector."""
    raise NotImplementedError


def posterior_samples(spec, candidate) -> FigData:
    """Posterior samples for the corner figure, across waveform models."""
    raise NotImplementedError


def localisation(spec, candidate) -> FigData:
    """Sky localisation and its credible areas."""
    raise NotImplementedError


def snr_series(spec, candidate) -> FigData:
    """Signal-to-noise time series from the follow-up filter."""
    raise NotImplementedError


def spectra(spec, candidate) -> FigData:
    """Amplitude spectra at the candidate, with the signal's frequency track."""
    raise NotImplementedError


def consistency_summary(spec, candidate) -> FigData:
    """Outcomes of the consistency tests, for the event page."""
    raise NotImplementedError


def waveform_consistency(spec) -> FigData:
    """Agreement between the recovered model and the data, across all candidates."""
    raise NotImplementedError


def build(spec, candidates: Optional[Sequence] = None) -> Dict[str, Path]:
    """Build per-candidate figure data for the requested candidates."""
    raise NotImplementedError

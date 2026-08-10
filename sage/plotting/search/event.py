#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : event.py
Description   : Per-candidate figures and the composite event page.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from typing import Optional, Sequence


def plot_spectrograms(fd, axes=None):
    """
    Multi-duration time-frequency maps per detector.

    The track implied by the recovered chirp mass is overlaid so the visible structure
    can be judged against what the parameters predict.
    """
    raise NotImplementedError


def plot_whitened_reconstruction(fd, axes=None):
    """Whitened data, the recovered model with its uncertainty, and the residual."""
    raise NotImplementedError


def plot_corner(fd, fig=None):
    """Posterior for the reported parameters, overlaid across waveform models."""
    raise NotImplementedError


def plot_skymap(fd, ax=None):
    """Sky localisation with its credible contours."""
    raise NotImplementedError


def plot_snr_series(fd, ax=None):
    """Signal-to-noise time series per detector, with the arrival-time difference marked."""
    raise NotImplementedError


def plot_spectra(fd, ax=None):
    """Amplitude spectra at the candidate, with the signal's frequency track."""
    raise NotImplementedError


def plot_consistency_summary(fd, ax=None):
    """Outcome of each consistency test, with tests that do not apply marked as such."""
    raise NotImplementedError


def plot_waveform_consistency(fd, axes=None):
    """Agreement between the recovered model and the data, across all candidates."""
    raise NotImplementedError


def event_page(fds, candidate: str, fig=None):
    """Compose the per-candidate figures into a single page."""
    raise NotImplementedError

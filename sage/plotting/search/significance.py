#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Significance and background figures of the search; every function
is a stub.

Filename      : significance.py
Description   : Search significance and background figures.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from typing import Optional


def plot_cumulative_vs_ifar(fd, ax=None):
    """
    Candidate counts against inverse false-alarm rate.

    The expected background is drawn with shaded uncertainty bands, and
    candidates beyond the measured background are marked as bounds
    rather than points.

    Parameters
    ----------
    fd : object
        Figure data product.
    ax : matplotlib.axes.Axes or None
        Axis to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError


def plot_statistic_distributions(fd, ax=None):
    """
    Foreground and background ranking-statistic distributions.

    Parameters
    ----------
    fd : object
        Figure data product.
    ax : matplotlib.axes.Axes or None
        Axis to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError


def plot_far_versus_statistic(fd, ax=None):
    """
    The statistic-to-rate mapping with its fitted tail.

    The region beyond the measured background is shaded so an
    extrapolated rate is never read as a measured one.

    Parameters
    ----------
    fd : object
        Figure data product.
    ax : matplotlib.axes.Axes or None
        Axis to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError


def plot_pastro_versus_statistic(fd, ax=None):
    """
    Astrophysical probability against ranking statistic.

    Parameters
    ----------
    fd : object
        Figure data product.
    ax : matplotlib.axes.Axes or None
        Axis to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError


def plot_pastro_versus_far(fd, ax=None):
    """
    Astrophysical probability against false-alarm rate, with published
    events.

    Parameters
    ----------
    fd : object
        Figure data product.
    ax : matplotlib.axes.Axes or None
        Axis to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError


def plot_foreground_rate(fd, ax=None):
    """
    Predicted against observed foreground counts.

    Parameters
    ----------
    fd : object
        Figure data product.
    ax : matplotlib.axes.Axes or None
        Axis to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError


def plot_window_offset_stability(fd, ax=None):
    """
    Score against analysis-window offset, for a signal and a noise
    trigger.

    Parameters
    ----------
    fd : object
        Figure data product.
    ax : matplotlib.axes.Axes or None
        Axis to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError


def plot_background_validity(fd, axes=None):
    """
    Background calibration, over-dispersion and per-slide livetime
    retention.

    Parameters
    ----------
    fd : object
        Figure data product.
    axes : sequence of matplotlib.axes.Axes or None
        Axes to draw on.

    Raises
    ------
    NotImplementedError
        Always; the figure is not implemented.
    """
    raise NotImplementedError

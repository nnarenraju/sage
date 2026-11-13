#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figures of the search configuration; every function is a stub.

Filename      : meta.py
Description   : Figures describing the search configuration.

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


def plot_training_prior(fd, axes=None):
    """
    The trained parameter distribution, with the searched region marked.

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


def plot_pipeline_diagram(fd, ax=None):
    """
    Stage graph of the analysis.

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


def plot_network_response(fd, ax=None):
    """
    Network output around a known event.

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


def plot_calibration(fd, ax=None):
    """
    Calibration of the reported probabilities against outcomes.

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

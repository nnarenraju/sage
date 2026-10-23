#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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
    """The trained parameter distribution, with the searched region marked."""
    raise NotImplementedError


def plot_pipeline_diagram(fd, ax=None):
    """Stage graph of the analysis."""
    raise NotImplementedError


def plot_network_response(fd, ax=None):
    """Network output around a known event."""
    raise NotImplementedError


def plot_calibration(fd, ax=None):
    """Calibration of the reported probabilities against outcomes."""
    raise NotImplementedError

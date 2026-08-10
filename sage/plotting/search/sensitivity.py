#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : sensitivity.py
Description   : Sensitivity, injection recovery and pipeline comparison figures.

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


def plot_vt_versus_far(fd, axes=None):
    """Sensitive volume-time against false-alarm-rate threshold, per population."""
    raise NotImplementedError


def plot_vt_versus_parameter(fd, axes=None):
    """Sensitivity binned in total mass, chirp mass, mass ratio and redshift."""
    raise NotImplementedError


def plot_pipeline_comparison(fd, ax=None):
    """
    Sensitivity against the reference pipelines.

    Where the comparison could not be restricted to a common coincidence type, the axis
    label states it rather than implying a like-for-like measurement.
    """
    raise NotImplementedError


def plot_injection_recovery(fd, axes=None):
    """Found and missed injections in the mass and distance planes."""
    raise NotImplementedError


def plot_efficiency(fd, ax=None):
    """Recovery efficiency against signal amplitude, at fixed false-alarm rate."""
    raise NotImplementedError


def plot_sensitive_distance(fd, ax=None):
    """Sensitive distance against mass."""
    raise NotImplementedError


def plot_range_over_time(fd, axes=None):
    """Detector range across the run, with its distribution."""
    raise NotImplementedError


def plot_surveyed_volume(fd, ax=None):
    """Cumulative detections against surveyed time-volume."""
    raise NotImplementedError


def plot_pastro_reliability(fd, ax=None):
    """Predicted against realised astrophysical fraction."""
    raise NotImplementedError

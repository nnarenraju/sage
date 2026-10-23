#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : catalogue.py
Description   : Catalogue comparison and detected-population figures.

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


def plot_comparison_matrix(fd, ax=None):
    """
    Events against catalogues.

    Distinguishes three states per cell: found, searched but not found, and outside that
    catalogue's searched region, since the last carries no information about the event.
    """
    raise NotImplementedError


def plot_overlap(fd, ax=None):
    """Membership counts for each combination of catalogues."""
    raise NotImplementedError


def plot_significance_agreement(fd, ax=None):
    """Significance from Sage against each catalogue, for commonly found events."""
    raise NotImplementedError


def plot_recovery_of_known_events(fd, ax=None):
    """Published events recovered by Sage, coloured by significance."""
    raise NotImplementedError


def plot_mass_plane(fd, ax=None):
    """
    Detected events in the component-mass plane.

    New candidates are drawn distinctly from published ones, with the searched region
    outlined and the mass frame stated on the axis.
    """
    raise NotImplementedError


def plot_parameter_planes(fd, axes=None):
    """Mass, mass ratio and effective spin against redshift."""
    raise NotImplementedError


def plot_spin_prior_sensitivity(fd, axes=None):
    """The same events under two spin priors, side by side."""
    raise NotImplementedError


def plot_population_shift(fd, axes=None):
    """Population inference with and without the new candidates."""
    raise NotImplementedError

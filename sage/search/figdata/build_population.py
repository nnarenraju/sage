#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_population.py
Description   : Figure data for the detected population.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Where new candidates do not measurably move a population inference, the population
figures add nothing beyond what is already published; the shift is measured first and
the figures are produced only if it is material.
"""

from pathlib import Path
from typing import Dict, Optional

from sage.search.figdata.product import FigData


def mass_plane(spec) -> FigData:
    """
    Detected events in the component-mass plane.

    New candidates are distinguished from previously published ones, and the mass frame
    is recorded so the axis can state it.
    """
    raise NotImplementedError


def parameter_planes(spec) -> FigData:
    """Mass, mass ratio and effective spin against redshift."""
    raise NotImplementedError


def spin_prior_sensitivity(spec) -> FigData:
    """
    The same events under two spin priors.

    Effective-spin conclusions can depend on the prior, so both are shown rather than
    one being chosen silently.
    """
    raise NotImplementedError


def population_shift(spec) -> FigData:
    """Population inference with and without the new candidates, and the shift between."""
    raise NotImplementedError


def build(spec, figures: Optional[list] = None) -> Dict[str, Path]:
    """Build the population figure data products, subject to the shift being material."""
    raise NotImplementedError

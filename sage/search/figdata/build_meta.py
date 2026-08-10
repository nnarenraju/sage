#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_meta.py
Description   : Figure data describing the search itself.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from pathlib import Path
from typing import Dict, Optional

from sage.search.figdata.product import FigData


def training_prior(spec) -> FigData:
    """
    The parameter distribution the network was trained on.

    Marks the searched region, which bounds where a sensitivity statement applies.
    """
    raise NotImplementedError


def pipeline_diagram(spec) -> FigData:
    """Stage graph and configuration, taken from the stage registry and the spec."""
    raise NotImplementedError


def network_response(spec) -> FigData:
    """The network's output around a known event, showing the trigger's shape."""
    raise NotImplementedError


def calibration(spec) -> FigData:
    """Calibration of the reported probabilities against outcomes."""
    raise NotImplementedError


def build(spec, figures: Optional[list] = None) -> Dict[str, Path]:
    """Build the descriptive figure data products."""
    raise NotImplementedError

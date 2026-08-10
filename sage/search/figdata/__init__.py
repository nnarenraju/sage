#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Figure data products: the persisted input to every figure.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Each figure is built from a file containing exactly the numbers it shows, written by an
analysis stage and read by a plotting function that computes nothing. A figure can then
be redrawn without rerunning the analysis, the numbers behind it can be released
alongside it, and a plot cannot disagree with the analysis it depicts.

Builders here read analysis products and write these files; they never recompute an
analysis quantity that a stage already produced.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

from sage.search.figdata.product import FigData

__all__ = ["FigData", "build", "load", "manifest", "verify"]


def build(spec, figures: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """
    Build the requested figure data products from the analysis outputs.

    Dispatches to the builders named in the figure declarations. Those imports happen
    inside this function rather than at module scope: every builder imports
    :class:`~sage.search.figdata.product.FigData` from this package, so importing them
    eagerly here would make the package and its builders import each other.
    """
    raise NotImplementedError


def load(spec, figure: str) -> FigData:
    """Load one figure's data product."""
    raise NotImplementedError


def verify(spec, figures: Optional[Sequence[str]] = None) -> Dict[str, bool]:
    """Check every product exists and matches the spec it was built from."""
    raise NotImplementedError


def manifest(spec, path: str | Path) -> Path:
    """Write the index of figures, their data products and their provenance."""
    raise NotImplementedError

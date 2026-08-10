#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : product.py
Description   : The figure data product: the numbers behind one figure.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Separate from the package ``__init__`` so that the builders can import the container
without importing the package that dispatches to them. Every builder needs this type, and
the dispatcher needs every builder, so leaving it in ``__init__`` makes the two import
each other.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class FigData:
    """
    The numbers behind one figure.

    Written by an analysis stage and read by a plotting function that computes nothing,
    so a figure can be redrawn without rerunning the analysis, the numbers behind it can
    be released alongside it, and a plot cannot disagree with what it depicts.
    """

    figure: str
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    scalars: Dict[str, object] = field(default_factory=dict)
    attrs: Dict[str, object] = field(default_factory=dict)

    def require(self, *names: str) -> None:
        """
        Assert the named arrays are present before drawing.

        Raises naming every missing array at once rather than failing on the first, since
        a builder that dropped one field has usually dropped several.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        """Write atomically, so an interrupted build leaves no half-written product."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "FigData":
        """Read a figure data product."""
        raise NotImplementedError

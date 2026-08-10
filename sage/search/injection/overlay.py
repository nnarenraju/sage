#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : overlay.py
Description   : Add injections into streamed real strain.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Injections are added to the same real-noise stream the zero-lag search reads, through
the same reader, so recovered sensitivity reflects the noise the search actually sees.
Injections falling outside the analysed segments are retained and counted as missed:
duty cycle is encoded in the found/missed outcome, and dropping them would inflate
sensitivity.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence

import numpy as np

from sage.search.reader import StreamingStrainReader, WindowBatch


@dataclass
class OverlayPlan:
    """Which injections land in which blocks, and where."""

    inj_id: np.ndarray
    gps: np.ndarray
    block_id: np.ndarray
    in_analysed_segment: np.ndarray

    def for_block(self, block_id: int) -> np.ndarray:
        """Indices of injections overlapping one block."""
        raise NotImplementedError

    @property
    def n_outside(self) -> int:
        """Injections outside analysed data; these count as missed."""
        raise NotImplementedError


def plan_overlay(injections, grid, geometry, assoc_window_s: float = 12.0) -> OverlayPlan:
    """
    Assign injections to blocks.

    The association window must be smaller than the spacing between injections in the
    stream being overlaid, or two injections can be attributed to one window.
    """
    raise NotImplementedError


class InjectedStrainReader(StreamingStrainReader):
    """A streaming reader that adds projected injections into each block."""

    def __init__(self, *args, injections=None, plan: Optional[OverlayPlan] = None,
                 generator=None, projection=None, **kwargs) -> None:
        raise NotImplementedError

    def iter_block(self, block) -> Iterator[WindowBatch]:
        """Yield batches with injections added."""
        raise NotImplementedError

    def _add_injections(self, block_strain: np.ndarray, block) -> np.ndarray:
        """Sum projected strain into a block before windowing."""
        raise NotImplementedError

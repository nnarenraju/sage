#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : background.py
Description   : Slide collation and the inclusive / exclusive / hierarchical sets.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Background is always clustered before it is counted. Hierarchical removal follows
GWTC-5.0: a candidate whose FAR falls below the removal threshold is taken out of the
background used to assess less significant candidates, working down in significance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sage.search.triggers import StatHistogram, TriggerTable

REMOVAL_MODES: Tuple[str, ...] = ("inclusive", "exclusive", "hierarchical")


@dataclass
class BackgroundSet:
    """Clustered background statistics with their exact accumulated livetime."""

    stats: np.ndarray
    livetime_s: float
    n_slides: int
    removal: str
    histogram: Optional[StatHistogram] = None
    removed_gps: Optional[np.ndarray] = None

    def n_above(self, stat: float) -> int:
        """Number of background events at or above ``stat``."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write ``background/bg_<removal>.h5`` for one observing run."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundSet":
        """Read a persisted background set."""
        raise NotImplementedError


def collate_slides(
    shard_paths: Sequence[str | Path],
    slide_plan,
    cluster_window_s: float,
    linkage: str = "peak",
) -> BackgroundSet:
    """Cluster every slide's triggers and accumulate the inclusive background."""
    raise NotImplementedError


def exclusive_background(
    background: BackgroundSet, zerolag_clustered: TriggerTable, veto_window_s: float
) -> BackgroundSet:
    """Drop background events coincident with any zero-lag trigger."""
    raise NotImplementedError


def hierarchical_removal(
    background: BackgroundSet,
    zerolag_clustered: TriggerTable,
    far_threshold_per_yr: float = 1e-2,
    veto_window_s: float = 1.0,
    max_iterations: int = 100,
) -> BackgroundSet:
    """
    Iteratively remove significant foreground from the background estimate.

    Candidates are removed in descending significance; each removal re-estimates the
    FAR of the remaining, less significant candidates.
    """
    raise NotImplementedError


def overdispersion_lrt(counts: np.ndarray) -> dict:
    """
    Poisson vs negative-binomial likelihood-ratio test on binned trigger counts.

    Reports whether the background is over-dispersed relative to Poisson, which is
    the condition under which simple order-statistic counting of FAR is valid.
    """
    raise NotImplementedError

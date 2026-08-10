#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : cluster.py
Description   : The single trigger clustering implementation.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A dense trigger train has to be reduced to independent events before it can be counted,
whether as foreground candidates or as background. The convention followed here is the
production matched-filter one: within a clustering window, keep the highest-ranked
trigger and discard the rest, so each surviving trigger represents one event.

The window is a configured quantity rather than a fixed constant, since it is bounded
below by the ranking statistic's autocorrelation scale and above by the shortest
separation at which two genuine signals must remain resolvable. Catalogue-level event
grouping uses a wider window than trigger-level clustering.

Two linkage rules are provided. ``peak`` measures separation from the loudest trigger
in the open cluster and bounds a cluster's extent at one window; it is the default.
``gap`` measures separation from the most recent trigger, which allows a cluster to
chain indefinitely through a continuous train and is retained only for comparison.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClusterResult:
    """Cluster representatives and their extents."""

    rep_index: np.ndarray
    times: np.ndarray
    stats: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    size: np.ndarray

    def __len__(self) -> int:
        """Number of clusters."""
        raise NotImplementedError

    def payload(self, columns: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Carry extra columns through by representative index."""
        raise NotImplementedError


def cluster_triggers(
    times: np.ndarray,
    stats: np.ndarray,
    window_s: float,
    linkage: str = "peak",
    payload: Optional[Dict[str, np.ndarray]] = None,
) -> ClusterResult:
    """
    Reduce a time-ordered trigger train to one representative per cluster.

    Parameters
    ----------
    times, stats : ndarray
        Trigger times and ranking statistics, ascending in time.
    window_s : float
        Maximum separation for two triggers to belong to the same cluster.
    linkage : {"peak", "gap"}
        Reference point for the separation test.
    payload : dict of ndarray, optional
        Extra per-trigger columns carried through by representative index.
    """
    raise NotImplementedError


def cluster_with_halo(
    times: np.ndarray,
    stats: np.ndarray,
    window_s: float,
    block_t0: float,
    block_t1: float,
    halo_s: float,
    linkage: str = "peak",
    payload: Optional[Dict[str, np.ndarray]] = None,
) -> ClusterResult:
    """
    Cluster one block while carrying a halo of neighbouring triggers.

    Clusters whose representative falls in the preceding block's halo are dropped, so a
    cluster straddling a block boundary is emitted once rather than counted twice.
    """
    raise NotImplementedError


def group_events(
    times: np.ndarray, window_s: float = 1.0, payload: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Group triggers into events for catalogue comparison.

    Returns a group label per trigger. The default window matches the convention used
    when comparing candidate lists across pipelines, where triggers within one second
    of each other are treated as the same event.
    """
    raise NotImplementedError

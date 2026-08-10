#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : features.py
Description   : Per-detector frontend feature cache for time slides.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Under a per-channel input norm the read, whiten, multirate and frontend stages depend
on one detector only, so they can be computed once per window and reused for every
lag; each slide then re-runs the backend alone. This is only valid when
``assert_separable`` passes, which excludes GroupNorm(1, D).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class CacheResidency:
    """Memory footprint of a block's cached features."""

    bytes_per_window_per_detector: int
    n_windows: int
    n_detectors: int

    @property
    def total_bytes(self) -> int:
        """Resident size for the block plus its lag halo."""
        raise NotImplementedError


class FrontendCache:
    """
    Hold per-detector frontend outputs for one block plus its lag halo.

    Parameters
    ----------
    device : str
        ``"cuda"`` keeps features resident on the GPU; ``"host"`` uses pinned host
        memory and trades PCIe bandwidth for a longer block.
    """

    def __init__(
        self,
        n_detectors: int,
        feature_shape: Tuple[int, ...],
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        raise NotImplementedError

    def put(self, detector: int, window_ids, features) -> None:
        """Store frontend outputs for a run of windows."""
        raise NotImplementedError

    def gather(self, detector: int, window_ids):
        """Retrieve features for a (possibly lag-shifted) set of window ids."""
        raise NotImplementedError

    def evict_before(self, window_id: int) -> None:
        """Drop features no longer reachable by any remaining lag."""
        raise NotImplementedError

    def residency(self) -> CacheResidency:
        """Current footprint."""
        raise NotImplementedError


def crossover_slides(f_full: float, f_front: float, f_back: float) -> float:
    """
    Number of slides at which caching becomes cheaper than re-running the full model.

    Uncached cost is ``n / f_full``; cached is ``1 / f_front + (1 + n) / f_back``.
    """
    raise NotImplementedError

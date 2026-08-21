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
        return int(
            self.bytes_per_window_per_detector * self.n_windows * self.n_detectors
        )


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
        import torch

        if n_detectors < 1:
            raise ValueError(f"n_detectors must be positive, got {n_detectors}")
        if not feature_shape or any(int(d) < 1 for d in feature_shape):
            raise ValueError(
                f"feature_shape must be a non-empty shape of positive extents, got "
                f"{feature_shape}"
            )
        if device not in ("cuda", "host", "cpu"):
            raise ValueError(
                f"unknown cache device {device!r}, expected cuda, host or cpu"
            )
        self.n_detectors = int(n_detectors)
        self.feature_shape = tuple(int(d) for d in feature_shape)
        self.device = str(device)
        self.dtype = getattr(torch, str(dtype))
        # One dict per detector, keyed by window id. A dict rather than a ring buffer
        # because eviction is driven by the lag halo rather than by age: with a
        # stratified ladder the earliest window still reachable is set by the largest
        # remaining lag, which is not a fixed distance behind the cursor.
        self._store: Tuple[Dict[int, "object"], ...] = tuple(
            {} for _ in range(self.n_detectors)
        )

    def _check_detector(self, detector: int) -> int:
        index = int(detector)
        if not 0 <= index < self.n_detectors:
            raise IndexError(
                f"detector {index} is outside the network of {self.n_detectors}"
            )
        return index

    def put(self, detector: int, window_ids, features) -> None:
        """
        Store frontend outputs for a run of windows.

        The features are detached and moved to the cache's own device before being kept.
        Holding a live view of a batch tensor instead would pin that whole batch in memory
        for as long as any one of its windows is reachable, and would keep it attached to
        an autograd graph that the search has no use for.
        """
        import torch

        index = self._check_detector(detector)
        ids = [int(value) for value in window_ids]
        values = features.detach()
        if len(values) != len(ids):
            raise ValueError(
                f"{len(ids)} window ids against {len(values)} feature rows; stored side "
                "by side they would attribute features to the wrong windows"
            )
        target = "cpu" if self.device in ("host", "cpu") else "cuda"
        for position, window_id in enumerate(ids):
            row = values[position]
            if tuple(row.shape) != self.feature_shape:
                raise ValueError(
                    f"feature for window {window_id} has shape {tuple(row.shape)}, "
                    f"expected {self.feature_shape}"
                )
            kept = row.to(device=target, dtype=self.dtype, copy=True)
            if self.device == "host":
                kept = kept.pin_memory()
            self._store[index][window_id] = kept

    def gather(self, detector: int, window_ids):
        """
        Retrieve features for a (possibly lag-shifted) set of window ids.

        A missing id raises rather than being skipped or zero-filled. Under a slide the
        follower's ids are shifted, so an id falling outside the cached halo means the
        halo was sized wrongly for the lag -- and a zero-filled feature would score as an
        ordinary quiet window, hiding the fault in the background rather than reporting it.
        """
        import torch

        index = self._check_detector(detector)
        ids = [int(value) for value in window_ids]
        held = self._store[index]
        missing = [value for value in ids if value not in held]
        if missing:
            span = f"[{min(held)}, {max(held)}]" if held else "empty"
            raise KeyError(
                f"detector {index} has no cached features for windows {missing[:8]}"
                f"{' and more' if len(missing) > 8 else ''}; the cache holds {span}. "
                "Under a lag the follower's window ids are shifted, so this means the "
                "halo is too small for the largest remaining lag"
            )
        if not ids:
            return torch.empty((0, *self.feature_shape), dtype=self.dtype)
        return torch.stack([held[value] for value in ids])

    def evict_before(self, window_id: int) -> None:
        """
        Drop features no longer reachable by any remaining lag.

        The caller supplies the boundary because only it knows the ladder: the earliest
        still-reachable window is the cursor minus the *largest remaining* lag, and a
        cache that evicted on age alone would drop exactly the windows the widest slide
        still needs.
        """
        boundary = int(window_id)
        for held in self._store:
            for stale in [value for value in held if value < boundary]:
                del held[stale]

    def residency(self) -> CacheResidency:
        """Current footprint."""
        import torch

        per_window = 1
        for extent in self.feature_shape:
            per_window *= int(extent)
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        held = max((len(store) for store in self._store), default=0)
        return CacheResidency(
            bytes_per_window_per_detector=int(per_window * itemsize),
            n_windows=int(held),
            n_detectors=int(self.n_detectors),
        )


def crossover_slides(f_full: float, f_front: float, f_back: float) -> float:
    """
    Number of slides at which caching becomes cheaper than re-running the full model.

    Uncached cost is ``n / f_full``; cached is ``1 / f_front + (1 + n) / f_back``.

    Setting the two equal and solving for ``n`` gives::

        n * (1/f_full - 1/f_back)  =  1/f_front + 1/f_back

    so the crossover exists only when ``f_back > f_full`` -- the backend has to be
    genuinely cheaper than the whole model, which it is precisely because the frontend
    work has been lifted out of it. When it is not, caching never pays at any depth and
    ``inf`` is returned rather than a negative number that would read as "always worth it".

    The ``1 +`` on the cached side is the zero-lag pass: the foreground is scored through
    the backend too, so a ladder of ``n`` slides costs ``n + 1`` backend passes in total.

    Returns
    -------
    float
        Slide count above which caching is cheaper, or ``inf`` when it never is. Not
        rounded: the fractional value is what a cost model wants, and rounding it here
        would hide how close a configuration sits to the boundary.
    """
    for name, value in (("f_full", f_full), ("f_front", f_front), ("f_back", f_back)):
        if not value > 0 or value != value or value == float("inf"):
            raise ValueError(f"{name} must be a positive finite rate, got {value}")
    if f_back <= f_full:
        # The backend alone is no faster than the whole model, so lifting the frontend out
        # bought nothing and no depth of ladder makes the cache pay for itself.
        return float("inf")
    return (1.0 / f_front + 1.0 / f_back) / (1.0 / f_full - 1.0 / f_back)

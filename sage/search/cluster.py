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

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

LINKAGES: Tuple[str, ...] = ("peak", "gap")


@dataclass
class ClusterResult:
    """Cluster representatives and their extents."""

    rep_index: np.ndarray
    times: np.ndarray
    stats: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    size: np.ndarray
    # Length of the trigger train this was reduced from. Kept because `rep_index` points
    # into that train, so it is what a payload column has to be as long as -- and after a
    # halo filter it is no longer recoverable from `size`.
    n_triggers: int = 0
    columns: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of clusters."""
        return int(self.rep_index.size)

    def payload(self, columns: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Carry extra columns through by representative index.

        Indexing rather than recomputing is the point: a representative's chirp mass and
        its statistic have to come from the same trigger, and a column re-derived from
        the cluster -- an average, a midpoint -- would describe no trigger that exists.
        """
        out: Dict[str, np.ndarray] = {}
        for name, values in (columns or {}).items():
            array = np.asarray(values)
            if array.ndim < 1 or array.shape[0] != self.n_triggers:
                raise ValueError(
                    f"payload column {name!r} has {array.shape[0] if array.ndim else 0} "
                    f"rows against the {self.n_triggers} triggers that were clustered; "
                    "a column of a different length would be indexed by position into "
                    "the wrong trigger"
                )
            out[name] = array[self.rep_index]
        return out


def _validated(times, stats, window_s: float, linkage: str):
    """Shared argument checking for the two clustering entry points."""
    times = np.asarray(times, dtype=np.float64)
    stats = np.asarray(stats, dtype=np.float64)
    if times.ndim != 1 or stats.ndim != 1:
        raise ValueError(
            f"times and stats must be one-dimensional; got shapes {times.shape} and "
            f"{stats.shape}"
        )
    if times.shape != stats.shape:
        raise ValueError(
            f"times and stats must be the same length; got {times.size} and {stats.size}"
        )
    if linkage not in LINKAGES:
        raise ValueError(f"linkage must be one of {LINKAGES}, got {linkage!r}")
    if not np.isfinite(window_s) or window_s < 0:
        raise ValueError(
            f"window_s must be finite and non-negative, got {window_s}; a negative "
            "window would leave every trigger its own cluster while claiming to have "
            "clustered them"
        )
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError(
            "triggers must be ascending in time; clustering a shuffled train silently "
            "produces clusters that are not contiguous in time"
        )
    return times, stats


def _empty_result(n_input: int) -> ClusterResult:
    """A result holding no clusters, in the dtypes a populated one uses."""
    return ClusterResult(
        rep_index=np.empty(0, dtype=np.int64),
        times=np.empty(0, dtype=np.float64),
        stats=np.empty(0, dtype=np.float64),
        t0=np.empty(0, dtype=np.float64),
        t1=np.empty(0, dtype=np.float64),
        size=np.empty(0, dtype=np.int64),
        n_triggers=int(n_input),
    )


def _peak_representatives(times: np.ndarray, stats: np.ndarray, window_s: float) -> np.ndarray:
    """
    Indices that are the loudest trigger within one window either side of themselves.

    This is the rule the module docstring states, read literally: a trigger survives when
    nothing within a window of it ranks higher. It is the same rule PyCBC applies to its
    own triggers and coincidences (``pycbc.events.coinc.cluster_over_time``), and the
    sweep below is theirs -- when a trigger is the maximum of its own window, every
    trigger up to that window's right edge is beaten by it, so the scan jumps there
    instead of re-examining each one; when the maximum lies to the right, the scan jumps
    straight to it. Both skips are what keep a dense glitch train linear rather than
    quadratic.

    Two properties follow that a greedy left-to-right sweep does not have.

    It does not depend on where the sweep started. Whether a trigger survives is decided
    by the triggers within one window of it and by nothing else, so clustering a block
    with a halo of at least one window gives exactly what clustering the whole run gives
    -- which is what lets the background be clustered in parallel at all.

    And a cluster cannot outgrow its window: every discarded trigger is within one window
    of something that beat it. A sweep that re-anchors on each new maximum chains through
    a rising train without bound, merging genuinely separate events.

    Ties are broken toward the earlier trigger, so equal statistics resolve the same way
    on every run and on every machine.

    One deliberate difference from PyCBC. Theirs takes both window edges with
    ``searchsorted`` default sides, giving the half-open ``[t - w, t + w)``: a trigger
    exactly one window later does not compete, while one exactly one window earlier does.
    That asymmetry lets two triggers exactly a window apart both survive when the later
    is louder. Here the window is symmetric and closed, ``[t - w, t + w]``, which agrees
    with :func:`_gap_representatives`'s ``>`` test at the same boundary -- so the two
    linkages cannot disagree about what "one window apart" means. On times that do not
    land exactly on the boundary the two give identical output, which
    ``test_search_cluster.py`` checks against PyCBC directly.
    """
    n = times.size
    if n == 0:
        return np.empty(0, dtype=np.int64)

    left = np.searchsorted(times, times - window_s, side="left")
    right = np.searchsorted(times, times + window_s, side="right")

    keep = np.zeros(n, dtype=np.int64)
    found = 0
    i = 0
    while i < n:
        lo, hi = left[i], right[i]
        if hi - lo == 1:            # nothing to compare against
            keep[found] = i
            found += 1
            i += 1
            continue
        # argmax takes the first maximum, which is the tie rule: the earliest wins.
        best = int(np.argmax(stats[lo:hi])) + lo
        if best == i:
            keep[found] = i
            found += 1
            i = hi              # everything up to the right edge is beaten by i
        elif best > i:
            i = best            # nothing between i and the maximum can survive
        else:
            i += 1
    return keep[:found]


def _gap_representatives(times: np.ndarray, stats: np.ndarray, window_s: float) -> np.ndarray:
    """
    Indices surviving single-linkage chaining: a gap wider than the window splits.

    This is the reference pipeline's rule (``benchmark/mlgwsc1/mlgwsc1.py::get_clusters``),
    kept so the engine can be compared against it trigger for trigger. It is not the
    default because a continuous train chains into one cluster however long it runs, so
    two genuine events separated by glitch activity merge into one.
    """
    n = times.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    # A gap strictly wider than the window opens a new cluster, matching the reference's
    # `(new - last) > threshold`. Equality therefore keeps two triggers together.
    breaks = np.flatnonzero(np.diff(times) > window_s) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [n]))
    # argmax takes the first maximum, which is the same tie rule as the peak sweep and
    # as the reference implementation.
    return np.array(
        [start + int(np.argmax(stats[start:stop])) for start, stop in zip(starts, stops)],
        dtype=np.int64,
    )


def _extents(
    times: np.ndarray, reps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign every trigger to the nearest representative and measure each cluster's span.

    The extents are what a later reader uses to say how long the trigger train behind a
    candidate lasted, so they describe the triggers that were discarded as well as the
    one that survived.
    """
    n = times.size
    if reps.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    rep_times = times[reps]
    right = np.searchsorted(rep_times, times)
    left = np.clip(right - 1, 0, reps.size - 1)
    right = np.clip(right, 0, reps.size - 1)
    take_right = np.abs(rep_times[right] - times) < np.abs(rep_times[left] - times)
    owner = np.where(take_right, right, left)

    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    sorted_times = times[order]
    edges = np.searchsorted(sorted_owner, np.arange(reps.size + 1))
    size = np.diff(edges).astype(np.int64)
    t0 = np.full(reps.size, np.nan)
    t1 = np.full(reps.size, np.nan)
    for index in range(reps.size):
        start, stop = edges[index], edges[index + 1]
        if stop > start:
            t0[index] = sorted_times[start]
            t1[index] = sorted_times[stop - 1]
        else:  # pragma: no cover - every representative owns at least itself
            t0[index] = t1[index] = rep_times[index]
    return t0, t1, size


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
        Maximum separation for two triggers to belong to the same cluster. Two triggers
        exactly ``window_s`` apart are in the same cluster; the test is inclusive, which
        is the reference pipeline's convention and keeps the boundary case decided rather
        than left to floating-point comparison.
    linkage : {"peak", "gap"}
        Reference point for the separation test.
    payload : dict of ndarray, optional
        Extra per-trigger columns carried through by representative index.

    Returns
    -------
    ClusterResult
        Representatives in ascending time order.
    """
    times, stats = _validated(times, stats, window_s, linkage)
    if times.size == 0:
        return _empty_result(0)

    if linkage == "peak":
        reps = _peak_representatives(times, stats, window_s)
    else:
        reps = _gap_representatives(times, stats, window_s)

    t0, t1, size = _extents(times, reps)
    result = ClusterResult(
        rep_index=reps,
        times=times[reps],
        stats=stats[reps],
        t0=t0,
        t1=t1,
        size=size,
        n_triggers=int(times.size),
    )
    if payload:
        result.columns = result.payload(payload)
    return result


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

    The halo must be at least one clustering window. Below that a trigger just inside the
    block cannot see everything that competes with it, and the block's answer stops
    matching the whole run's -- silently, and in the direction that adds a background
    event at every boundary, which lowers every FAR.

    Parameters
    ----------
    block_t0, block_t1 : float
        Half-open block bounds, ``[block_t0, block_t1)``. A representative exactly at
        ``block_t1`` belongs to the next block, so consecutive blocks partition the run.
    halo_s : float
        Extra time either side whose triggers compete but are not emitted.
    """
    if halo_s < window_s:
        raise ValueError(
            f"a halo of {halo_s} s is narrower than the {window_s} s clustering window, "
            "so a trigger at the block edge cannot see every trigger competing with it; "
            "blockwise clustering would then disagree with clustering the whole run"
        )
    if block_t1 < block_t0:
        raise ValueError(
            f"block bounds are reversed: [{block_t0}, {block_t1})"
        )
    whole = cluster_triggers(times, stats, window_s, linkage=linkage, payload=payload)
    keep = (whole.times >= block_t0) & (whole.times < block_t1)
    return ClusterResult(
        rep_index=whole.rep_index[keep],
        times=whole.times[keep],
        stats=whole.stats[keep],
        t0=whole.t0[keep],
        t1=whole.t1[keep],
        size=whole.size[keep],
        n_triggers=whole.n_triggers,
        columns={name: values[keep] for name, values in whole.columns.items()},
    )


def coincidence_time(
    times_by_detector: Dict[str, np.ndarray],
    offsets_s: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One time per coincidence: the mean over the detectors that participated in it.

    Following ``pycbc.events.coinc.cluster_coincs_multiifo``. A slid detector's trigger
    time is pulled back through its own offset first, so the mean sits where the
    coincidence would have been at zero lag rather than drifting with the slide -- and
    two coincidences a window apart in physical time stay a window apart however deep in
    the ladder they were found.

    A detector that did not participate is marked by a non-positive time and is left out
    of both the mean and the count, so an HL coincidence inside an HLV campaign is
    averaged over two detectors and not over three with a zero in it.

    Returns
    -------
    tuple of ndarray
        ``(time, n_detectors)``, the second being how many detectors contributed.
    """
    if not times_by_detector:
        raise ValueError("no detectors given")
    offsets_s = dict(offsets_s or {})
    names = list(times_by_detector)
    stacked = np.vstack(
        [
            np.asarray(times_by_detector[name], dtype=np.float64)
            - float(offsets_s.get(name, 0.0))
            for name in names
        ]
    )
    # The sentinel is tested on the raw time, before the offset is removed: subtracting a
    # lag from -1 would turn "did not participate" into an ordinary-looking time.
    raw = np.vstack(
        [np.asarray(times_by_detector[name], dtype=np.float64) for name in names]
    )
    participating = raw > 0
    n_detectors = participating.sum(axis=0)
    if np.any(n_detectors == 0):
        raise ValueError(
            "a coincidence with no participating detector cannot be given a time; "
            f"{int((n_detectors == 0).sum())} of {n_detectors.size} rows are empty"
        )
    total = np.where(participating, stacked, 0.0).sum(axis=0)
    return total / n_detectors, n_detectors.astype(np.int64)


def cluster_slides(
    times: np.ndarray,
    stats: np.ndarray,
    slide_ids: np.ndarray,
    window_s: float,
    linkage: str = "peak",
    payload: Optional[Dict[str, np.ndarray]] = None,
) -> ClusterResult:
    """
    Cluster every slide separately, in one pass.

    Each slide is an independent realisation of the background and must be clustered on
    its own: two triggers from different slides are not the same event however close in
    time, and letting them suppress one another would remove background events that were
    never coincident, lowering the count and every FAR taken from it.

    The separation is done the way PyCBC does it in
    ``pycbc.events.coinc.cluster_coincs`` -- each slide's times are displaced into a band
    of their own, wider than the data plus several windows, so one sweep over the
    displaced times can never compare across slides. Looping per slide would give the
    same answer; this does it in one pass over the sorted array.

    Where this departs from PyCBC is precision. They add ``span * slide_id`` to raw GPS
    and reach for ``longdouble`` to survive it. The displacement here is applied to times
    measured from the earliest trigger, so the values stay of order the run length rather
    than of order 1.2e9, and float64 carries them with room to spare.

    Parameters
    ----------
    slide_ids : ndarray
        Which slide each trigger came from. Zero-lag is an ordinary slide here.

    Returns
    -------
    ClusterResult
        Representatives across all slides, in ascending (slide, time) order.
        ``rep_index`` points into the input arrays.
    """
    times = np.asarray(times, dtype=np.float64)
    stats = np.asarray(stats, dtype=np.float64)
    slide_ids = np.asarray(slide_ids)
    if not (times.shape == stats.shape == slide_ids.shape):
        raise ValueError(
            f"times, stats and slide_ids must be the same length; got {times.size}, "
            f"{stats.size} and {slide_ids.size}"
        )
    if times.size == 0:
        return _empty_result(0)

    order = np.lexsort((times, slide_ids))
    ordered_times = times[order]
    ordered_slides = slide_ids[order]

    # A band wider than the data plus several windows, so no window can straddle two.
    # Taken from the extremes of the whole trigger set, not from the ends of the
    # slide-ordered array: `order` sorts by slide first, so `ordered_times[-1]` is the
    # last time of the last slide and `ordered_times[0]` the first of the first, whose
    # difference is not the time range and can be far smaller than it -- or negative.
    # Too narrow a band lets neighbouring slides overlap, and triggers from different
    # slides then suppress one another, which is the one thing this function exists to
    # prevent.
    first, last = float(times.min()), float(times.max())
    span = (last - first) + 10.0 * float(window_s)
    if not np.isfinite(span) or span <= 0.0:
        span = 10.0 * float(window_s) + 1.0
    relative = ordered_times - first
    bands = (ordered_slides - ordered_slides.min()).astype(np.float64)
    displaced = relative + span * bands

    inner = cluster_triggers(displaced, stats[order], window_s, linkage=linkage)
    reps = order[inner.rep_index]

    result = ClusterResult(
        rep_index=reps,
        times=times[reps],
        stats=stats[reps],
        # The extents are reported in real time, not displaced time.
        t0=inner.t0 - span * bands[inner.rep_index] + first,
        t1=inner.t1 - span * bands[inner.rep_index] + first,
        size=inner.size,
        n_triggers=int(times.size),
    )
    if payload:
        result.columns = result.payload(payload)
    return result


def group_events(
    times: np.ndarray, window_s: float = 1.0, payload: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Group triggers into events for catalogue comparison.

    Returns a group label per trigger. The default window matches the convention used
    when comparing candidate lists across pipelines, where triggers within one second
    of each other are treated as the same event.

    Single linkage, deliberately, and not the clustering rule above: this answers "do two
    pipelines mean the same event", where a chain of near-coincident times is one event,
    and there is no ranking statistic shared between pipelines to anchor on anyway.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(f"times must be one-dimensional; got shape {times.shape}")
    if not np.isfinite(window_s) or window_s < 0:
        raise ValueError(f"window_s must be finite and non-negative, got {window_s}")
    if times.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be ascending; grouping a shuffled list is undefined")
    return np.concatenate(([0], np.cumsum(np.diff(times) > window_s))).astype(np.int64)

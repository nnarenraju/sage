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

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sage.search.fingerprint import combine

from sage.search.cluster import cluster_slides
from sage.search.triggers import (
    _BLOCK_DATASET,
    _TRIGGER_GROUP,
    StatHistogram,
    TriggerTable,
    histogram_stats,
    merge_shards,
)

REMOVAL_MODES: Tuple[str, ...] = ("inclusive", "exclusive", "hierarchical")

SECONDS_PER_JULIAN_YEAR: float = 31557600.0


def n_louder(stat: np.ndarray, background_stats: np.ndarray) -> np.ndarray:
    """Background events at or above each ``stat``, the count behind :func:`far_of_stat`."""
    stat = np.asarray(stat, dtype=np.float64)
    ordered = np.sort(np.asarray(background_stats, dtype=np.float64).ravel())
    return (ordered.size - np.searchsorted(ordered, stat, side="left")).astype(np.int64)


def far_of_stat(
    stat: np.ndarray, background_stats: np.ndarray, livetime_s: float
) -> np.ndarray:
    """
    Conservative FAR per second: ``(1 + n_b(>= stat)) / T_b``.

    The ``1 +`` is not a guard against dividing by zero -- the denominator is the
    livetime, which is never zero here. It is the statement that a candidate louder than
    every background event has not been shown to have a rate of zero; the background
    simply ran out. Without it the loudest candidate in any campaign gets an infinite
    IFAR, which is a property of the background's length rather than of the candidate.

    The count is inclusive at ``stat``, so a background event exactly as loud counts.

    Defined here rather than in :mod:`sage.search.far`, which re-exports it, because
    :func:`hierarchical_removal` needs it while the background stage is running -- one
    stage before ``far``. A stage may not import a module scheduled after it, or it
    cannot run when it is scheduled to.

    Parameters
    ----------
    livetime_s : float
        Background livetime, summed from the slide plan. There is no closed form for it
        -- see :attr:`~sage.search.slides.SlidePlan.background_livetime_s`, which excludes
        the zero-lag slide because ``n_b`` here counts slid events only.
    """
    stat = np.asarray(stat, dtype=np.float64)
    background_stats = np.asarray(background_stats, dtype=np.float64).ravel()
    if not np.isfinite(livetime_s) or livetime_s <= 0:
        raise ValueError(
            f"background livetime must be finite and positive, got {livetime_s}"
        )
    if background_stats.size and np.isnan(background_stats).any():
        raise ValueError("background statistics contain NaN, which no comparison counts")
    # The query is checked too, and this is the half that matters. A NaN compares false
    # against everything, so searchsorted places it past the end of the background and it
    # counts as louder than every event there -- collecting the smallest rate the search
    # can assign. A window the network could not rank would then arrive as the most
    # significant candidate in the campaign.
    if np.isnan(stat).any():
        raise ValueError(
            f"{int(np.isnan(stat).sum())} of {stat.size} statistics being assigned a rate "
            "are NaN; a NaN counts as louder than every background event and would be "
            "reported as the most significant candidate rather than as a failure"
        )
    return (1.0 + n_louder(stat, background_stats)) / float(livetime_s)


@dataclass
class BackgroundSet:
    """
    Clustered background statistics with their exact accumulated livetime.

    ``gps`` holds one time per background event, in the reference detector's frame, and
    is what the exclusive and hierarchical sets are built by removing from. It is
    optional because a set restored from a histogram alone can still answer
    :meth:`n_above`, but a set without it cannot have foreground removed from it -- there
    is nothing to decide coincidence against.

    ``tc_gps`` holds each event's decoded merger time, the same quantity the zero-lag
    side is vetoed on. ``gps`` is the analysis window's reference time and is quantised
    to the stride; a coincidence test that read one clock on one side and the other on
    the other would carry a fixed bias of up to the tc prior's width into a window of
    comparable size. A set without it falls back to ``gps`` on both sides, which is
    self-consistent but coarser.

    ``slide_id`` says which slide each event came from, which is what turns its
    reference-frame time into a per-detector time: under slide ``k`` a follower's data is
    read at ``gps + offset_k[detector]``. Removing foreground contamination needs those,
    because a background event is contaminated through the detector data it actually used
    and not through the frame it is recorded in.

    ``foreground_livetime_s`` is the zero-lag exposure that survives the same veto, set
    whenever a removal mode reduced the livetime. A removal takes time out of the
    foreground as well as the background -- the vetoed stretch is gone from both -- so a
    FAR curve drawn in exclusive or hierarchical mode needs this rather than the
    inclusive plan's foreground time, which describes an exposure the removal ended.

    ``removed_gps`` records the zero-lag times whose neighbourhoods were taken out of the
    background: every clustered zero-lag trigger for the exclusive set, and only the
    candidates that passed the removal threshold for the hierarchical one. Keeping the
    times rather than a count is what lets a later stage say which candidate a change in
    the background is attributable to. Both modes record the same clock -- the decoded
    merger time where the zero-lag table carries one -- so the two are comparable and a
    reader does not have to know which mode wrote the dataset to know what is in it.
    """

    stats: np.ndarray
    livetime_s: float
    n_slides: int
    removal: str
    histogram: Optional[StatHistogram] = None
    removed_gps: Optional[np.ndarray] = None
    gps: Optional[np.ndarray] = None
    tc_gps: Optional[np.ndarray] = None
    slide_id: Optional[np.ndarray] = None
    foreground_livetime_s: Optional[float] = None

    def __post_init__(self) -> None:
        """Refuse a set that cannot be counted as a background."""
        self.stats = np.asarray(self.stats, dtype=np.float64).ravel()
        if self.removal not in REMOVAL_MODES:
            raise ValueError(
                f"removal must be one of {REMOVAL_MODES}, got {self.removal!r}"
            )
        if not np.isfinite(self.livetime_s) or self.livetime_s <= 0:
            raise ValueError(
                f"background livetime must be finite and positive, got "
                f"{self.livetime_s}; every rate divides by it"
            )
        if self.slide_id is not None:
            self.slide_id = np.asarray(self.slide_id, dtype=np.int64).ravel()
            if self.slide_id.shape != self.stats.shape:
                raise ValueError(
                    f"the background holds {self.slide_id.size} slide ids against "
                    f"{self.stats.size} statistics; read side by side they would "
                    "attribute events to the wrong slides"
                )
        if self.tc_gps is not None:
            self.tc_gps = np.asarray(self.tc_gps, dtype=np.float64).ravel()
            if self.tc_gps.shape != self.stats.shape:
                raise ValueError(
                    f"the background holds {self.tc_gps.size} merger times against "
                    f"{self.stats.size} statistics; read side by side they would describe "
                    "different events"
                )
        if self.histogram is not None and not self.histogram.clustered:
            raise ValueError(
                "the background histogram is not marked clustered; an unclustered "
                "trigger train counts one event per window of a glitch rather than one "
                "per glitch, which inflates the count several times over and lowers "
                "every FAR taken from it"
            )

    def n_above(self, stat: float) -> int:
        """
        Number of background events at or above ``stat``.

        Inclusive, so a background event exactly as loud as the candidate counts toward
        it: it is evidence that the noise reaches that value.

        Answered from the stored statistics when they are present and from the histogram
        otherwise. The statistics are exact; the histogram resolves only to a bin, and
        over-counts by at most one bin's worth in the conservative direction.
        """
        if self.stats.size:
            return int(np.count_nonzero(self.stats >= stat))
        if self.histogram is None:
            raise ValueError(
                "this background holds neither statistics nor a histogram, so nothing "
                "can be counted from it"
            )
        return self.histogram.n_above(stat)

    def save(self, path: str | Path) -> None:
        """
        Write ``background/bg_<removal>.h5`` for one observing run.

        The statistics are stored, not only the histogram. They are what
        :func:`~sage.search.far.build_far_curve` counts, and a histogram resolves a
        statistic only to a bin -- enough for a rate, not enough to reproduce the exact
        count a published IFAR was quoted from.

        Written under ``atomic_h5``, so a kill mid-write leaves the previous background
        intact rather than a truncated file that reads as a thinner background.
        """
        from sage.utils.atomic_io import atomic_h5

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_h5(target, mode="w") as handle:
            handle.attrs["livetime_s"] = float(self.livetime_s)
            handle.attrs["n_slides"] = int(self.n_slides)
            handle.attrs["removal"] = str(self.removal)
            if self.foreground_livetime_s is not None:
                # The zero-lag exposure left by whatever veto produced this set. Both
                # removals reduce it and both compute the reduction; without it here the
                # number dies at the file boundary and every consumer falls back to the
                # un-vetoed exposure, quoting expected counts and p-values over time the
                # veto had already taken away.
                handle.attrs["foreground_livetime_s"] = float(
                    self.foreground_livetime_s
                )
            handle.create_dataset(
                "stats", data=np.asarray(self.stats, dtype=np.float64)
            )
            for name in ("gps", "removed_gps", "tc_gps", "slide_id"):
                values = getattr(self, name)
                if values is not None:
                    handle.create_dataset(
                        name, data=np.asarray(values, dtype=np.float64)
                    )
            if self.histogram is not None:
                group = handle.create_group("histogram")
                group.create_dataset(
                    "counts",
                    data=np.asarray(self.histogram.counts),
                    compression="gzip",
                )
                group.attrs["underflow"] = int(self.histogram.underflow)
                group.attrs["overflow"] = int(self.histogram.overflow)
                group.attrs["clustered"] = bool(self.histogram.clustered)

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundSet":
        """
        Read a persisted background set.

        Every attribute the counting needs is required by name. A background missing its
        livetime or its removal mode is not a background that can be quoted: the first is
        the denominator of every rate and the second says which of three different
        backgrounds the number came from, and neither has a defensible default.
        """
        import h5py

        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"no background set at {target}")
        with h5py.File(target, "r") as handle:
            for name in ("livetime_s", "n_slides", "removal"):
                if name not in handle.attrs:
                    raise ValueError(
                        f"{target} carries no {name!r} attribute; it is not a complete "
                        "background set and no rate taken from it can be defended"
                    )
            if "stats" not in handle:
                raise ValueError(
                    f"{target} is missing the 'stats' dataset; the file was truncated "
                    "part-way through a write"
                )
            histogram = None
            if "histogram" in handle:
                group = handle["histogram"]
                histogram = StatHistogram(
                    counts=np.asarray(group["counts"]),
                    underflow=int(group.attrs["underflow"]),
                    overflow=int(group.attrs["overflow"]),
                    clustered=bool(group.attrs["clustered"]),
                )
            return cls(
                stats=np.asarray(handle["stats"]),
                livetime_s=float(handle.attrs["livetime_s"]),
                n_slides=int(handle.attrs["n_slides"]),
                removal=str(handle.attrs["removal"]),
                histogram=histogram,
                removed_gps=(
                    np.asarray(handle["removed_gps"])
                    if "removed_gps" in handle
                    else None
                ),
                gps=np.asarray(handle["gps"]) if "gps" in handle else None,
                tc_gps=np.asarray(handle["tc_gps"]) if "tc_gps" in handle else None,
                slide_id=(
                    np.asarray(handle["slide_id"]) if "slide_id" in handle else None
                ),
                # Optional rather than required: the inclusive set has no reduction to
                # record, and a set written before this attribute existed is still a
                # valid background. Absent means "not reduced", which is what None says.
                foreground_livetime_s=(
                    float(handle.attrs["foreground_livetime_s"])
                    if "foreground_livetime_s" in handle.attrs
                    else None
                ),
            )


def _within_window(
    times: np.ndarray, veto_times: np.ndarray, window_s: float
) -> np.ndarray:
    """
    Mask of ``times`` lying within ``window_s`` of any time in ``veto_times``.

    Nearest neighbour either side of the insertion point rather than the pairwise
    comparison: a campaign's background holds millions of events and its zero-lag list
    thousands, and the product of the two is what makes the direct implementation
    unusable rather than merely slow.

    The test is inclusive at the boundary, as
    :func:`~sage.search.cluster.cluster_triggers` is, so "one window apart" means the
    same thing everywhere in the search.
    """
    times = np.asarray(times, dtype=np.float64)
    veto_times = np.asarray(veto_times, dtype=np.float64).ravel()
    if times.size == 0 or veto_times.size == 0:
        return np.zeros(times.shape, dtype=bool)
    ordered = np.sort(veto_times)
    insert = np.searchsorted(ordered, times)
    before = ordered[np.clip(insert - 1, 0, ordered.size - 1)]
    after = ordered[np.clip(insert, 0, ordered.size - 1)]
    return (np.abs(times - before) <= window_s) | (np.abs(times - after) <= window_s)


# Coincidence windows for foreground removal. Both are PyCBC's defaults.
#
# ``VETO_WINDOW_S`` is the exclusive background's, PyCBC's ``--veto-window``
# (``pycbc_coinc_statmap``): the separation within which a slid event is treated as a
# copy of a zero-lag trigger rather than as noise. ``HIERARCHICAL_WINDOW_S`` is the
# removal window, PyCBC's ``--hierarchical-removal-window``, and is wider because it
# bounds the stretch of detector data a confirmed signal occupies rather than a matching
# tolerance between two time estimates.
#
# Constants rather than per-trigger widths. Sage's decoder reports a ``tc_sigma`` and an
# earlier revision of this module scaled the window by it, but measured over all four
# production networks -- 200,000 validation windows each, signal and noise -- the largest
# sigma any of them produces is 0.0896 s, so a four-sigma half-width never exceeded
# 0.507 s and the scaling sat on its own 1 s floor in 800,000 of 800,000 cases. No
# reference scales a removal window by a timing uncertainty: PyCBC's windows are scalars
# passed to ``pycbc.events.coinc.time_coincidence``, and sgwc-1 used a constant
# ``zerolag_time_threshold = 10.0``.
VETO_WINDOW_S: float = 0.1
HIERARCHICAL_WINDOW_S: float = 1.0


def _require_events(background: BackgroundSet, action: str) -> None:
    """Refuse a background that cannot have foreground removed from it."""
    if background.gps is None:
        raise ValueError(
            f"this background carries no event times, so {action} has nothing to decide "
            "coincidence against; collate it with collate_slides, which keeps them"
        )
    if np.asarray(background.gps).shape != np.asarray(background.stats).shape:
        raise ValueError(
            f"the background holds {np.asarray(background.gps).size} times against "
            f"{np.asarray(background.stats).size} statistics; the two would be read side "
            "by side and would describe different events"
        )


def _require_clustered_zerolag(table: TriggerTable, action: str) -> None:
    """Refuse a zero-lag trigger list that was never clustered, or lacks a column."""
    if "clustered" in table.attrs and not bool(table.attrs["clustered"]):
        raise ValueError(
            f"refusing to run {action} against an unclustered zero-lag list: a glitch "
            "contributes one trigger per window rather than one candidate, so it would "
            "veto a stretch of background out of all proportion to the one event it is"
        )
    for name in ("stat", "gps"):
        if name not in table.columns:
            raise ValueError(
                f"the zero-lag table holds {sorted(table.columns)} and {action} needs "
                f"{name!r}"
            )


def _declared_slides(shard_paths: Sequence[str | Path]) -> set:
    """
    Slide ids the shards themselves declare, read from their attributes.

    Read from the attributes rather than inferred from the trigger rows, because the two
    disagree in exactly the case worth catching: a slide whose job ran and found nothing
    above threshold contributes no rows, and is indistinguishable by its rows alone from
    a slide whose job never ran at all. Only the attribute distinguishes "measured, empty"
    from "missing".

    A shard counts only when it is *finished*. The slide id is stamped at creation, before
    a single block has been scored, so a job that died part-way leaves a shard that
    declares its slide exactly as a completed one does. Counting it would let the ladder
    collate with a numerator short of the events that slide would have contributed while
    the plan's full livetime still fills the denominator -- every FAR in the campaign low,
    by a factor nothing in the products records. Two conditions say finished: the
    ``finalised`` flag the writer sets on close, and, when the producer stamped how many
    blocks the slide has, every one of them committed.

    Attributes only -- the trigger data is not touched here, so this costs one header read
    per shard and not a second pass over the campaign.
    """
    import h5py

    covered = set()
    for path in shard_paths:
        target = Path(path)
        with h5py.File(target, "r") as handle:
            if not bool(handle.attrs.get("finalised", False)):
                continue
            expected_blocks = handle.attrs.get("n_blocks")
            if expected_blocks is not None:
                done = handle.get(_BLOCK_DATASET)
                if done is None or done.shape[0] != int(expected_blocks):
                    continue
            if "slide_id" in handle.attrs:
                covered.add(int(handle.attrs["slide_id"]))
                continue
            group = handle.get(_TRIGGER_GROUP)
            if group is not None and "slide_id" in group:
                covered.update(int(v) for v in np.unique(np.asarray(group["slide_id"])))
                continue
            raise ValueError(
                f"{target} declares no 'slide_id' attribute and holds no slide_id "
                "column, so the slides it covers cannot be established. A shard that "
                "cannot be attributed to a slide cannot be checked for against the plan, "
                "which is what keeps a failed slide job from shortening the numerator "
                "while the plan's livetime still fills the denominator"
            )
    return covered


def collate_slides(
    shard_paths: Sequence[str | Path],
    slide_plan,
    cluster_window_s: float,
    linkage: str = "peak",
) -> BackgroundSet:
    """
    Cluster every slide's triggers and accumulate the inclusive background.

    Clustering is done per slide, by :func:`~sage.search.cluster.cluster_slides`. Each
    slide is an independent realisation of the background, so two triggers from different
    slides are not the same event however close in time; letting them suppress one
    another would delete background events that were never coincident, which lowers the
    count and with it every false-alarm rate taken from the count.

    Livetime is the sum of the per-slide livetimes the plan measured. There is no closed
    form for it: per-slide retention falls with lag, so ``n_slides * T_zerolag`` always
    overstates, and an overstated denominator reports every rate too low.
    :attr:`~sage.search.slides.SlidePlan.background_livetime_s` already excludes the
    zero-lag slide.

    Zero-lag triggers found among the shards are dropped rather than counted. The
    denominator excludes the zero-lag slide, so the numerator has to as well -- the two
    have to describe the same time or the ratio is not a rate. A shard directory that
    happens to hold the zero-lag shard beside the slides therefore gives the same
    background as one that does not, instead of a background biased by one slide's worth
    of foreground.

    Parameters
    ----------
    shard_paths : sequence of path
        Unclustered slide shards. Their provenance is checked against one another by
        :func:`~sage.search.triggers.merge_shards`, so shards from two configurations
        cannot be collated into one background.
    cluster_window_s : float
        Clustering window, in seconds.
    linkage : {"peak", "gap"}
        Passed through to the clusterer; ``peak`` is the production rule.

    Returns
    -------
    BackgroundSet
        The inclusive background: every slid event that survived clustering, with the
        plan's summed livetime and a histogram of the survivors.

    Raises
    ------
    ValueError
        A shard carries a slide id the plan does not describe. That slide has no measured
        livetime behind it, so its events would be counted in a numerator whose
        denominator never included them.
    """
    table, _ = merge_shards(shard_paths, require_clustered=False)
    for name in ("stat", "gps", "slide_id"):
        if name not in table.columns:
            raise ValueError(
                f"the shards hold {sorted(table.columns)}; a background needs {name!r}, "
                "since it is clustered per slide and counted in time"
            )
    slide_ids = np.asarray(table["slide_id"], dtype=np.int64)
    known = {int(slide.slide_id) for slide in slide_plan.slides}
    unknown = sorted(set(slide_ids.tolist()) - known)
    if unknown:
        raise ValueError(
            f"the shards carry slides {unknown}, which the plan does not describe; their "
            "events have no measured livetime behind them and would be counted in a rate "
            "whose denominator never included them"
        )
    expected = {int(slide.slide_id) for slide in slide_plan.slides if slide.slide_id != 0}
    covered = _declared_slides(shard_paths)
    absent = sorted(expected - covered)
    if absent:
        raise ValueError(
            f"the plan describes {len(expected)} slid slides but the shards cover only "
            f"{len(expected) - len(absent)}; slides {absent[:10]}"
            f"{' and more' if len(absent) > 10 else ''} produced no shard at all. Their "
            "livetime is still summed into the denominator by the plan, so collating "
            "here would divide a numerator short by those slides' events into the full "
            "background time and report every rate low -- in the direction that makes "
            "candidates look more significant. A slide that ran and found nothing writes "
            "an empty shard and is not affected by this"
        )
    slid = slide_ids != 0
    # The clusterer carries the decoded merger time through on the surviving
    # representatives, so foreground removal can compare merger time against merger time
    # on both sides. Recovering it afterwards would mean matching representatives back to
    # input rows by time, which is exactly the ambiguity clustering exists to remove.
    payload = {"slide_id": slide_ids[slid].astype(np.float64)}
    if "tc_gps" in table.columns:
        payload["tc_gps"] = np.asarray(table["tc_gps"], dtype=np.float64)[slid]
    result = cluster_slides(
        np.asarray(table["gps"], dtype=np.float64)[slid],
        np.asarray(table["stat"], dtype=np.float64)[slid],
        slide_ids[slid],
        window_s=float(cluster_window_s),
        linkage=linkage,
        payload=payload,
    )
    return BackgroundSet(
        stats=result.stats,
        livetime_s=slide_plan.background_livetime_s,
        n_slides=sum(1 for slide in slide_plan.slides if slide.slide_id != 0),
        removal="inclusive",
        histogram=histogram_stats(result.stats, clustered=True),
        gps=result.times,
        tc_gps=result.columns.get("tc_gps"),
        slide_id=np.asarray(result.columns["slide_id"]).astype(np.int64),
    )


def _veto_times(table: TriggerTable) -> np.ndarray:
    """
    The times a veto is centred on: the decoded merger time where one exists.

    ``tc_gps`` is the network's estimate of when the merger happened; ``gps`` is the
    analysis window's reference time, which is quantised to the stride and offset from
    the merger by up to the tc prior's width. The background side reads the same column
    through :func:`_detector_times`, so both sides of every coincidence test are on one
    clock -- sgwc-1 compares ``gps_tc`` against ``gps_tc``, and mixing the two would put
    a fixed bias of up to 0.1 s into a window of 0.1 s.
    """
    name = "tc_gps" if "tc_gps" in table.columns else "gps"
    return np.asarray(table[name], dtype=np.float64).ravel()


def _veto_intervals(
    times: np.ndarray, half_width_s: float, detectors: Sequence[str]
) -> dict:
    """
    Veto spans per detector, one ``(t - w, t + w)`` per removed candidate.

    The same spans in every detector. A zero-lag candidate's arrival times differ between
    detectors by at most the light travel across the network -- 27 ms for HLV -- which is
    inside the window itself, so resolving them per detector would be arithmetic below
    the resolution of the thing being computed. PyCBC subtracts one scalar veto time from
    every detector for the same reason.
    """
    w = float(half_width_s)
    spans = [(float(t) - w, float(t) + w) for t in np.ravel(times)]
    return {detector: list(spans) for detector in detectors}


def exclusive_background(
    background: BackgroundSet,
    zerolag_clustered: TriggerTable,
    slide_plan,
    geometry,
    segments_by_detector: dict,
    window_s: float = VETO_WINDOW_S,
) -> BackgroundSet:
    """
    Drop background events coincident with any zero-lag trigger, and the time with them.

    The exclusive background answers "what would the noise look like if the foreground
    were not in it": every slid event that shares a detector time with a zero-lag
    candidate is removed, on the grounds that it may be a slid copy of that candidate
    rather than noise. It is the lower bound of the three backgrounds -- it removes the
    most -- and therefore assigns the highest significance.

    **The test is made in every detector's frame, not the reference frame alone.** A real
    signal in Hanford is re-paired against a different stretch of Livingston in every
    slide, so its contaminating copies sit at a *different* reference time in each slide
    and only one of them is visible from the reference frame. PyCBC applies its veto to
    each detector's own trigger time for exactly this reason
    (``pycbc_coinc_statmap``: ``for ifo in ifos: veto.indices_within_times(...)``). In HL
    a reference-frame test would catch half the contamination; in HLV, one frame of three.

    **The livetime is reduced to match.** Thinning the numerator while the denominator
    still describes the time those events were counted in reports every rate low by the
    vetoed fraction, which is the direction that makes candidates look more significant.
    The reduction is measured, not scaled: :func:`~sage.search.slides.remeasure_livetimes`
    rebuilds every slide's lattice with the vetoed stretches removed, because a veto in
    one detector costs each slide a different amount of coincident time depending on what
    the other detector was doing when it pairs there. This follows PyCBC, which reduces
    its own ``background_time_exc`` rather than leaving it at the inclusive value.

    It is reported beside the inclusive set rather than instead of it. Removing every
    zero-lag trigger's neighbourhood removes genuine noise as well as any signal, so the
    exclusive set understates the noise by an amount nothing measures; quoting it alone
    would present that as a measurement.

    Parameters
    ----------
    slide_plan, geometry, segments_by_detector
        What the livetime is re-measured through, and what places each background event
        in its detectors' frames. The plan must be the ladder the background was collated
        on -- checked, because a mismatched plan would place events at lags they were
        never built from and silently veto the wrong stretches.
    window_s : float
        Half-width of the coincidence test, PyCBC's ``--veto-window``.
    """
    _require_events(background, "exclusive_background")
    _require_clustered_zerolag(zerolag_clustered, "exclusive_background")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"window_s must be finite and positive, got {window_s}")
    veto_times = _veto_times(zerolag_clustered)

    detector_times = _detector_times(background, slide_plan)
    times = np.asarray(background.gps, dtype=np.float64)
    hit = np.zeros(times.shape, dtype=bool)
    for values in detector_times.values():
        hit |= _within_window(values, veto_times, float(window_s))
    keep = ~hit

    from sage.search.slides import remeasure_livetimes

    reduced = remeasure_livetimes(
        slide_plan,
        geometry,
        segments_by_detector,
        _veto_intervals(veto_times, float(window_s), sorted(segments_by_detector)),
    )
    stats = np.asarray(background.stats, dtype=np.float64)[keep]
    return BackgroundSet(
        stats=stats,
        livetime_s=reduced.background_livetime_s,
        n_slides=background.n_slides,
        removal="exclusive",
        histogram=histogram_stats(stats, clustered=True),
        removed_gps=np.sort(veto_times),
        gps=times[keep],
        tc_gps=None if background.tc_gps is None else background.tc_gps[keep],
        slide_id=None if background.slide_id is None else background.slide_id[keep],
        foreground_livetime_s=reduced.foreground_livetime_s,
    )


def _detector_times(background: BackgroundSet, slide_plan) -> dict:
    """
    Each background event's time in every detector's own frame.

    A slid event is recorded at a reference-frame time, but it was built from data the
    detectors supplied at different moments: under a slide the follower is read at
    ``gps + offset``. Contamination reaches the background through those moments -- a real
    signal in Hanford is re-paired against a different stretch of Livingston in every
    slide, and every one of those pairings sits at the same Hanford time -- so a test made
    on the reference frame alone would miss the copies in every slide but one.

    Built from ``tc_gps`` where the background carries it, so the comparison is merger
    time against merger time; ``gps`` otherwise, which is self-consistent as long as the
    zero-lag side falls back with it.
    """
    if background.slide_id is None:
        raise ValueError(
            "this background carries no slide ids, so its events cannot be placed in "
            "each detector's own frame; collate it with collate_slides, which keeps them"
        )
    offsets = {int(slide.slide_id): dict(slide.offsets_s) for slide in slide_plan}
    ids = np.asarray(background.slide_id, dtype=np.int64)
    unknown = sorted(set(ids.tolist()) - set(offsets))
    if unknown:
        raise ValueError(
            f"the background holds slides {unknown} the plan does not describe, so their "
            "events have no lag and cannot be placed in any detector's frame"
        )
    declared = sum(1 for slide in slide_plan if int(slide.slide_id) != 0)
    if declared != int(background.n_slides):
        raise ValueError(
            f"this background was collated on {background.n_slides} slid slides but the "
            f"plan given here describes {declared}; a plan that is not the ladder the "
            "background was built on supplies lags those events were never slid by, and "
            "the veto would be applied to the wrong stretches of every detector"
        )
    base = (
        np.asarray(background.gps, dtype=np.float64)
        if background.tc_gps is None
        else np.asarray(background.tc_gps, dtype=np.float64)
    )
    detectors = sorted({name for value in offsets.values() for name in value})
    shifts = {
        detector: np.array(
            [offsets[int(k)].get(detector, 0.0) for k in ids], dtype=np.float64
        )
        for detector in detectors
    }
    return {detector: base + shift for detector, shift in shifts.items()}


def _inside_intervals(times: np.ndarray, intervals: Sequence[Tuple[float, float]]):
    """Mask of ``times`` lying inside any of a sorted, merged interval list."""
    times = np.asarray(times, dtype=np.float64)
    if not intervals:
        return np.zeros(times.shape, dtype=bool)
    starts = np.array([lo for lo, _ in intervals], dtype=np.float64)
    ends = np.array([hi for _, hi in intervals], dtype=np.float64)
    index = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, starts.size - 1)
    return (times >= starts[index]) & (times <= ends[index])


# How the hierarchical walk decides it is finished.
#
# "significance" is PyCBC's (``pycbc_coinc_statmap``: ``while numpy.any(ifar_foreground
# >= background_time)``): keep removing while any zero-lag candidate is louder than every
# surviving background event, which under counting FAR is the same as its IFAR reaching
# the background livetime. It has no free parameter.
#
# "counted" is sgwc-1's (``search.ipynb`` cell 254): keep walking past background events
# that have nothing louder on them, and stop only after ``ignore_limit`` of them in a row.
# sgwc-1 used 200, annotated as arbitrary.
STOP_RULES: Tuple[str, ...] = ("significance", "counted")


def hierarchical_removal(
    background: BackgroundSet,
    zerolag_clustered: TriggerTable,
    slide_plan,
    geometry,
    segments_by_detector: dict,
    min_background_livetime_s: float = 0.0,
    window_s: float = HIERARCHICAL_WINDOW_S,
    max_iterations: int = 100,
    stop_rule: str = "significance",
    ignore_limit: int = 200,
) -> BackgroundSet:
    """
    Remove foreground contamination from the background, loudest background event first.

    Walk the background downward in ranking statistic. For the loudest event still in it,
    ask whether any zero-lag candidate **louder than that event** shares one of its
    detector times. If one does, the background event is not noise: it is a slid copy of a
    real signal, and the whole stretch of detector data that signal occupies is removed
    from every slide at once, along with the livetime that stretch contributed. Then ask
    again of whatever is now loudest.

    The "louder than" gate is what makes this different from
    :func:`exclusive_background`, which vetoes on every zero-lag trigger regardless of
    significance. A zero-lag candidate quieter than the background event it coincides with
    is no evidence that the background event is contaminated -- the noise reached higher
    there than the candidate did.

    **Stopping.** ``stop_rule`` selects between the two published rules; see
    :data:`STOP_RULES`. Neither stops at the first background event that survives its
    check, which an earlier revision of this function did: the walk starts at the loudest
    background event, that event is usually clean, and stopping there collapsed the
    hierarchical set onto the inclusive one in 40 of 40 randomised backgrounds carrying a
    genuine contaminant. The reasoning behind that stop was also wrong in its own terms --
    as the walk descends, the set of candidates louder than the current event grows, so
    more can be established further down, not less.

    An event whose detector times fall outside analysed zero-lag data is skipped rather
    than counted either way: it could not have been checked, so it is evidence of nothing.
    sgwc-1 prints ``Trigger present in non-coincident data! Ignoring`` and continues.

    **The livetime is reduced with the events**, which follows the ruling that removal
    must cost background time. Note this goes beyond both references: PyCBC recomputes
    ``background_time`` from an attribute its removal does not touch, and states inside
    its own loop that the correction is expected to be negligible
    (``pycbc_add_statmap``); sgwc-1 never re-measures livetime at all. Measuring it
    exactly is strictly better than assuming it negligible, but it is Sage's choice.

    Parameters
    ----------
    min_background_livetime_s : float
        A floor on the surviving background livetime. Zero by default -- no floor unless
        the caller asks for one, which is what both references do. A proposed removal that
        would breach it is **declined and the walk continues** to the next event, rather
        than ending the walk: aborting on the first breach would make the floor a veto on
        every later removal too, and with the floor set near the campaign's own background
        livetime that turns the whole function into a no-op returning the inclusive set
        under a hierarchical label.
    window_s : float
        Half-width of the coincidence and removal window, PyCBC's
        ``--hierarchical-removal-window``.
    max_iterations : int
        Hard bound on removals, PyCBC's ``--max-hierarchical-removal``. The walk already
        terminates -- each pass either stops, retires one unusable event, declines one
        removal, or removes at least one candidate -- and this bounds it independently of
        the inputs. Stopping early leaves the remaining candidates assessed against a
        larger background, which reports them as less significant, not more.
    stop_rule : str
        ``"significance"`` for PyCBC's rule, ``"counted"`` for sgwc-1's.
    ignore_limit : int
        Under ``"counted"``, how many consecutive checkable-but-clean background events
        end the walk. Ignored under ``"significance"``.

    Returns
    -------
    BackgroundSet
        The surviving background with its re-measured livetime, and ``removed_gps`` naming
        the zero-lag candidates whose data was taken out.
    """
    _require_events(background, "hierarchical_removal")
    _require_clustered_zerolag(zerolag_clustered, "hierarchical_removal")
    if not np.isfinite(min_background_livetime_s) or min_background_livetime_s < 0:
        raise ValueError(
            "min_background_livetime_s must be finite and non-negative, got "
            f"{min_background_livetime_s}"
        )
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"window_s must be finite and positive, got {window_s}")
    if max_iterations < 0:
        raise ValueError(f"max_iterations must not be negative, got {max_iterations}")
    if stop_rule not in STOP_RULES:
        raise ValueError(f"stop_rule must be one of {STOP_RULES}, got {stop_rule!r}")
    if ignore_limit < 0:
        raise ValueError(f"ignore_limit must not be negative, got {ignore_limit}")

    from sage.search.segments import coincident_intervals
    from sage.search.slides import remeasure_livetimes

    stats = np.asarray(background.stats, dtype=np.float64)
    times = np.asarray(background.gps, dtype=np.float64)
    detector_times = _detector_times(background, slide_plan)
    analysed = coincident_intervals(segments_by_detector)
    checkable = np.ones(stats.size, dtype=bool)
    for values in detector_times.values():
        checkable &= _inside_intervals(values, analysed)

    candidate_stats = np.asarray(zerolag_clustered["stat"], dtype=np.float64).ravel()
    candidate_times = _veto_times(zerolag_clustered)

    alive = np.ones(stats.size, dtype=bool)
    retired = np.zeros(stats.size, dtype=bool)
    available = np.ones(candidate_stats.size, dtype=bool)
    removed: List[int] = []
    livetime = float(background.livetime_s)
    foreground = None
    detectors = sorted(segments_by_detector)
    half = float(window_s)
    clean_run = 0

    while len(removed) < int(max_iterations):
        pool = alive & ~retired
        if not pool.any():
            break
        index = int(np.flatnonzero(pool)[np.argmax(stats[pool])])

        louder = available & (candidate_stats >= stats[index])
        hit = np.zeros(candidate_stats.size, dtype=bool)
        if louder.any():
            where = np.flatnonzero(louder)
            for detector in detectors:
                separation = np.abs(
                    detector_times[detector][index] - candidate_times[where]
                )
                hit[where] |= separation <= half
        if not hit.any():
            if not checkable[index]:
                # Outside analysed zero-lag data: unanswerable, so neither a stop nor a
                # clean event. sgwc-1 prints and continues.
                retired[index] = True
                continue
            if stop_rule == "significance":
                # PyCBC: the loudest surviving background event has no louder candidate
                # on it, so no candidate outranks the whole background and the condition
                # `any(ifar_foreground >= background_time)` is false.
                break
            clean_run += 1
            retired[index] = True
            if clean_run > int(ignore_limit):
                break
            continue
        clean_run = 0

        proposed = removed + np.flatnonzero(hit).tolist()
        reduced = remeasure_livetimes(
            slide_plan,
            geometry,
            segments_by_detector,
            _veto_intervals(candidate_times[proposed], half, detectors),
        )
        if reduced.background_livetime_s < float(min_background_livetime_s):
            # The floor: decline this removal and keep walking. The event stays in the
            # background, so it must not be offered again.
            retired[index] = True
            continue

        removed = proposed
        livetime = float(reduced.background_livetime_s)
        foreground = float(reduced.foreground_livetime_s)
        available[hit] = False
        for detector in detectors:
            alive &= ~_within_window(
                detector_times[detector], candidate_times[np.flatnonzero(hit)], half
            )

    surviving = np.asarray(background.stats, dtype=np.float64)[alive]
    return BackgroundSet(
        stats=surviving,
        livetime_s=livetime,
        n_slides=background.n_slides,
        removal="hierarchical",
        histogram=histogram_stats(surviving, clustered=True),
        removed_gps=np.sort(candidate_times[removed]),
        gps=times[alive],
        tc_gps=None if background.tc_gps is None else background.tc_gps[alive],
        slide_id=None if background.slide_id is None else background.slide_id[alive],
        foreground_livetime_s=foreground,
    )


# Significance at which :func:`overdispersion_lrt` calls a background over-dispersed.
# Reported alongside the p-value so a caller can apply its own instead.
OVERDISPERSION_LEVEL: float = 0.05

# Range of dispersions the fit searches. The lower end stands in for the Poisson limit;
# it can be this small because the likelihood ratio below is evaluated without ever
# forming a large intermediate, so the arithmetic does not degrade as alpha shrinks.
_MIN_ALPHA: float = 1e-12
_MAX_ALPHA: float = 1e6


def overdispersion_lrt(counts: np.ndarray) -> dict:
    """
    Poisson vs negative-binomial likelihood-ratio test on binned trigger counts.

    Reports whether the background is over-dispersed relative to Poisson, which is
    the condition under which simple order-statistic counting of FAR is valid.

    The alternative is the NB2 parameterisation, ``Var = mu + alpha * mu**2``, which
    contains the Poisson model at ``alpha = 0``. The profile is exact in ``mu``: the
    score equation for the mean gives ``mu = mean(counts)`` whatever ``alpha`` is, so
    only the dispersion is searched for, over one dimension, where a bounded search
    cannot land on the wrong local maximum.

    ``alpha = 0`` is on the boundary of the parameter space, so the null distribution of
    the statistic is not chi-square with one degree of freedom but the equal mixture of
    that and a point mass at zero (Cameron and Trivedi). The reported p-value is
    therefore half the chi-square one. Using the naive chi-square tail would report every
    background as twice as Poisson-like as it is -- in the direction that lets an
    over-dispersed background pass unremarked, which is the failure this test exists to
    catch.

    Under-dispersed counts leave the maximum at the boundary. The fit is then reported as
    ``alpha = 0`` with a statistic of exactly zero rather than the small negative value
    the arithmetic produces, since a likelihood ratio in favour of the nested model is a
    rounding artefact and not a result.

    Parameters
    ----------
    counts : ndarray
        Events per bin, over bins of equal exposure. Equal exposure is the caller's
        responsibility and cannot be checked from the counts alone; unequal bins look
        over-dispersed however Poisson the process is.

    Returns
    -------
    dict
        ``statistic``, ``p_value`` and ``overdispersed`` at ``OVERDISPERSION_LEVEL``,
        the fitted ``alpha`` and ``mean``, the two log-likelihoods, and the observed
        ``index_of_dispersion`` (variance over mean, which is one for a Poisson process)
        as an independent description of the same data.
    """
    from scipy import optimize, special, stats as _stats

    counts = np.asarray(counts)
    if counts.ndim != 1:
        raise ValueError(
            f"counts must be one-dimensional, got shape {tuple(counts.shape)}"
        )
    if counts.size < 2:
        raise ValueError(
            f"a dispersion needs at least two bins to be measured, got {counts.size}"
        )
    values = np.asarray(counts, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("counts contain a non-finite value, which no bin can hold")
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError(
            "counts must be non-negative whole numbers; a fractional count is a rate or "
            "a weight, and neither has a Poisson likelihood"
        )
    mean = float(values.mean())
    if mean == 0.0:
        raise ValueError(
            "every bin is empty, so there is no dispersion to measure and no rate to "
            "compare against"
        )

    loglik_poisson = float(
        np.sum(values * np.log(mean) - mean - special.gammaln(values + 1.0))
    )
    integers = values.astype(np.int64)
    ladder = np.arange(int(integers.max()), dtype=np.float64)

    def loglik_ratio(log_alpha: float) -> float:
        """
        NB2 log-likelihood minus the Poisson one, at the profile mean.

        Written as the ratio rather than as a difference of two likelihoods, and with
        every term evaluated near zero. The textbook form needs
        ``gammaln(y + 1/alpha) - gammaln(1/alpha)``, whose two halves are of order 1e9
        by the time the dispersion is small enough to matter, so the quantity the fit
        depends on is lost to cancellation exactly where the two models are being told
        apart. Because the counts are whole numbers that difference is the finite sum
        ``sum_k log(1/alpha + k)``, which is exact, and pairing each term against the
        Poisson one turns it into ``log1p`` of a small argument.

        Costs one pass over the distinct counts up to the largest, plus one over the
        bins, per evaluation.
        """
        size = np.exp(-log_alpha)
        steps = np.log1p((ladder - mean) / (size + mean))
        cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        return float(
            cumulative[integers].sum()
            - values.size * (size * np.log1p(mean / size) - mean)
        )

    fit = optimize.minimize_scalar(
        lambda log_alpha: -loglik_ratio(log_alpha),
        bounds=(np.log(_MIN_ALPHA), np.log(_MAX_ALPHA)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    ratio = float(-fit.fun)
    if ratio <= 0.0:
        # The maximum sits at the Poisson boundary: the counts are at most as dispersed
        # as Poisson, and a likelihood ratio in favour of the nested model is arithmetic
        # noise rather than a result.
        alpha, loglik_negbin, statistic = 0.0, loglik_poisson, 0.0
    else:
        alpha = float(np.exp(fit.x))
        loglik_negbin = loglik_poisson + ratio
        statistic = 2.0 * ratio
    # Half the chi-square tail: the null puts half its mass at zero because alpha cannot
    # be negative.
    p_value = 0.5 * float(_stats.chi2.sf(statistic, 1)) if statistic > 0.0 else 1.0

    return {
        "n_bins": int(values.size),
        "mean": mean,
        "variance": float(values.var(ddof=1)),
        "index_of_dispersion": float(values.var(ddof=1) / mean),
        "alpha": alpha,
        "loglik_poisson": loglik_poisson,
        "loglik_negbin": loglik_negbin,
        "statistic": statistic,
        "p_value": p_value,
        "level": OVERDISPERSION_LEVEL,
        "overdispersed": bool(p_value < OVERDISPERSION_LEVEL),
    }


def cluster_zerolag(
    table: TriggerTable, window_s: float, linkage: str = "peak"
) -> TriggerTable:
    """
    Reduce a zero-lag trigger train to one representative per event.

    Foreground removal must be given candidates, not windows. A glitch produces one
    trigger per window of the lattice, so an unclustered list opens a veto window around
    each of them and takes out a stretch of background out of all proportion to the single
    event it is -- which is why :func:`exclusive_background` refuses a table not marked
    clustered.

    Every schema column present is carried through on the surviving representatives, so
    the decoded merger time travels with the candidate that was kept.
    """
    from sage.search.cluster import cluster_triggers

    payload = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in table.columns.items()
        if name not in ("stat", "gps")
    }
    result = cluster_triggers(
        np.asarray(table["gps"], dtype=np.float64),
        np.asarray(table["stat"], dtype=np.float64),
        window_s=float(window_s),
        linkage=str(linkage),
        payload=payload or None,
    )
    columns = {"stat": result.stats, "gps": result.times}
    columns.update({name: np.asarray(v) for name, v in result.columns.items()})
    return TriggerTable(columns=columns, attrs={"clustered": True})

#: Where the frozen keep threshold lives, once, for the whole campaign.
KEEP_THRESHOLD_FILE = ("background", "keep_threshold.json")


def freeze_keep_threshold(spec, plan) -> float:
    """
    Freeze the campaign's keep threshold, from the complete zero-lag histogram.

    The threshold is read from **every** window of the observing run, counted before any
    slide job starts, and returned as a bin edge -- exactly representable, so every slide
    thresholds on the identical number.

    Calibrating on a subsample is the failure this exists to prevent. O3a is
    non-stationary, so "the first one per cent of blocks" is not representative of the
    run, and a per-slide threshold would let each slide keep a different fraction of its
    own tail, which changes the background count without changing anything that looks
    wrong.

    Written to its own small file, created exclusively, and **not** into ``slide_plan.h5``.
    The plan is what the whole background array reads, and stamping the threshold into it
    made the freeze a read-modify-write over an HDF5 file that ``SlidePlan.save`` opens
    with mode ``"w"`` -- it truncates in place. Measured on Lustre with ten array tasks
    released together, nine of ten died, with ``BlockingIOError`` under HDF5's own locking
    and with truncated-file and bad-B-tree-signature errors when locking is disabled, as
    it commonly is on a shared filesystem. It also gave the plan two owners, so re-running
    ``slides`` silently wiped a threshold the already-scored rungs had used.

    ``O_CREAT | O_EXCL`` makes the race safe rather than avoiding it: the first task to
    arrive writes the value, and every other task reads what it wrote. There is no
    ordering requirement between array tasks, and no task ever sees a partial file --
    the content is written to a temporary and renamed, which is atomic within a directory.

    A plan carrying an explicit ``keep_threshold`` wins. That is a threshold pinned by the
    configuration rather than derived from this run's foreground, and reproducing a
    published campaign is exactly what it is for.
    """
    import json
    import os
    import tempfile

    from sage.search.triggers import read_shard

    target = spec.path(*KEEP_THRESHOLD_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return float(json.loads(target.read_text())["keep_threshold"])

    if plan is not None and plan.keep_threshold is not None:
        threshold = float(plan.keep_threshold)
        source = "pinned in the slide plan"
    else:
        shard = spec.path("zerolag", "zerolag_slide0000.h5")
        if not Path(shard).is_file():
            raise FileNotFoundError(
                f"no zero-lag shard at {shard}; the keep threshold is frozen from the "
                "complete zero-lag histogram, so the foreground must be scored before "
                "any slide is"
            )
        _, histogram = read_shard(shard)
        threshold = float(
            histogram.quantile_threshold(float(spec.significance.keep_rate))
        )
        source = "quantile of the complete zero-lag histogram"

    payload = json.dumps(
        {
            "keep_threshold": threshold,
            "keep_rate": float(spec.significance.keep_rate),
            "source": source,
            "spec_hash": str(spec.hash()),
        },
        indent=2,
        sort_keys=True,
    )
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".json")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(payload)
        try:
            # Fails if another task got here first, which is the point: its value is
            # already frozen and this one adopts it rather than replacing it.
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        os.unlink(temporary)
    return float(json.loads(target.read_text())["keep_threshold"])


def run(spec, slides=None, **kwargs) -> dict:
    """
    Stage driver: score the lag ladder, collate it, and build every removal mode.

    ``slides`` selects which rungs to score, for a SLURM array where each task owns a few.
    ``None`` scores every slide the plan describes, which is what a single-process run
    does. Collation happens only once every shard exists, so an array task that finishes
    early does not collate a partial ladder -- a background short by a slide divides a
    deficient count into the full livetime and reports every rate low, in the direction
    that makes candidates look more significant.

    The keep threshold is frozen here rather than in ``slides``: it comes from the
    complete zero-lag histogram, and this is the first stage that depends on both. It goes
    to its own file, not into the plan -- see :func:`freeze_keep_threshold`.
    """
    from sage.search.engine import run_search
    from sage.search.slides import SlidePlan

    plan_path = spec.path("slides", "slide_plan.h5")
    plan = SlidePlan.load(plan_path)
    threshold = freeze_keep_threshold(spec, plan)

    wanted = (
        [int(s.slide_id) for s in plan if int(s.slide_id) != 0]
        if slides is None
        else [int(s) for s in slides]
    )
    lags = {int(s.slide_id): dict(s.offsets_s) for s in plan}
    shifts = {
        int(s.slide_id): (dict(s.window_shift) if s.window_shift else None) for s in plan
    }
    scored = []
    for slide_id in wanted:
        if slide_id not in lags:
            raise ValueError(
                f"slide {slide_id} is not in the stored ladder at {plan_path}"
            )
        scored.append(
            run_search(
                spec,
                stage="background",
                slide_id=slide_id,
                offsets_s=lags[slide_id],
                window_shift=shifts.get(slide_id),
                keep_threshold=threshold,
            )
        )

    shards = sorted(spec.path("background").glob("background_slide*.h5"))
    covered = _declared_slides(shards)
    expected = {int(s.slide_id) for s in plan if int(s.slide_id) != 0}
    if not expected <= covered:
        # An array task that ran its share and stopped. Report, do not collate.
        return {
            "scored_slides": wanted,
            "n_scored": len(scored),
            "keep_threshold": threshold,
            "collated": False,
            "missing_slides": sorted(expected - covered),
            "fingerprint": f"partial:{sorted(covered)}",
        }

    inclusive = collate_slides(
        [str(path) for path in shards],
        plan,
        cluster_window_s=float(spec.cluster.window_s),
        linkage=spec.cluster.linkage,
    )
    return _build_removal_modes(spec, plan, inclusive, threshold, wanted, len(scored))


def _mode_digest(background) -> str:
    """
    Content summary of one background: its counts, its livetimes and its events.

    The statistics and merger times are hashed rather than summarised by size, because a
    removal that vetoes a different set of the same size leaves every scalar unchanged.
    ``foreground_livetime_s`` is included because a removal reduces zero-lag exposure as
    well as background events, and that reduction is the denominator of the mode's own
    expected counts.
    """
    import hashlib

    stats = np.ascontiguousarray(np.asarray(background.stats, dtype=np.float64))
    times = (
        np.ascontiguousarray(np.asarray(background.tc_gps, dtype=np.float64)).tobytes()
        if background.tc_gps is not None
        else b"no-tc"
    )
    digest = hashlib.sha256(stats.tobytes() + times).hexdigest()[:16]
    foreground = background.foreground_livetime_s
    return (
        f"{stats.size}:{float(background.livetime_s):.6f}:"
        f"{'none' if foreground is None else format(float(foreground), '.6f')}:{digest}"
    )


def _build_removal_modes(spec, plan, inclusive, threshold, wanted, n_scored) -> dict:
    """
    Save the inclusive background and whichever removed sets the campaign asked for.

    The removed sets are written beside the inclusive one, never instead of it. Each
    removes genuine noise along with any signal, by an amount nothing measures, so
    quoting one alone would present that as a measurement.
    """
    from sage.search.segments import load_segments
    from sage.search.triggers import read_shard

    geometry = spec.geometry_object()
    segments = {
        detector: load_segments(
            Path(spec.data.release_dir)
            / f"data_{detector}_{spec.data.observing_run}_segments.json"
        )
        for detector in spec.data.detectors
    }
    written = {}
    digests = {}
    inclusive_path = spec.path("background", "bg_inclusive.h5")
    inclusive.save(inclusive_path)
    written["inclusive"] = str(inclusive_path)
    digests["inclusive"] = _mode_digest(inclusive)

    removed = [m for m in spec.significance.removal_modes if m != "inclusive"]
    if removed:
        zerolag, _ = read_shard(spec.path("zerolag", "zerolag_slide0000.h5"))
        clustered = cluster_zerolag(
            zerolag,
            window_s=float(spec.cluster.window_s),
            linkage=spec.cluster.linkage,
        )
        for mode in removed:
            if mode == "exclusive":
                result = exclusive_background(
                    inclusive, clustered, plan, geometry, segments,
                    window_s=float(spec.significance.veto_window_s),
                )
            else:
                result = hierarchical_removal(
                    inclusive, clustered, plan, geometry, segments,
                    window_s=float(spec.significance.hierarchical_window_s),
                    min_background_livetime_s=float(
                        spec.significance.min_background_livetime_s
                    ),
                    max_iterations=int(spec.significance.max_removals),
                    stop_rule=spec.significance.stop_rule,
                    ignore_limit=int(spec.significance.ignore_limit),
                )
            path = spec.path("background", f"bg_{mode}.h5")
            result.save(path)
            written[mode] = str(path)
            digests[mode] = _mode_digest(result)

    return {
        "scored_slides": wanted,
        "n_scored": n_scored,
        "keep_threshold": threshold,
        "collated": True,
        "n_background_events": int(inclusive.stats.size),
        "background_livetime_s": float(inclusive.livetime_s),
        "background_livetime_yr": float(
            inclusive.livetime_s / SECONDS_PER_JULIAN_YEAR
        ),
        "modes": written,
        # Per-mode contents, not per-mode names. A removal that takes out a different set
        # of the same size leaves every scalar summary exactly where it was while moving
        # the loudest background event -- which is the number every candidate's FAR is
        # read against. The names alone would report "nothing moved" for a corrected stop
        # rule, a corrected veto window or a corrected clustering.
        "fingerprint": combine(
            f"{threshold:.9g}",
            *(f"{mode}={digests[mode]}" for mode in sorted(digests)),
        ),
    }

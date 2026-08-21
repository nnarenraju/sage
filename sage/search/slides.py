#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : slides.py
Description   : Time-slide ladder generation and exact per-slide livetime.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Lags are stratified over ``[min_separation, tau_max]`` rather than packed against the
minimum. Packing them re-pairs one loud glitch against nearly the same stretch of the
other detector on every slide, at precisely the lag scale where detector noise is most
correlated, which inflates the effective number of independent background samples.

A slide is a lag **per detector relative to a reference**, so a network of ``D`` detectors
has ``D - 1`` independent lags and the ladder is a lattice in that many dimensions. Two
consequences follow, and the second is easy to miss:

* more detectors give many more distinct slides for the same ``tau_max``, so a
  three-detector background is cheaper per year of background than a two-detector one,
  even though each slide retains less livetime;
* the minimum separation has to hold for **every pair**, which for three detectors
  includes the difference between two lagged detectors. An implementation written for two
  detectors only ever checks lags against the reference and will happily emit a slide in
  which Livingston and Virgo sit within a light-travel time of each other, quietly
  admitting genuine coincidences into the background.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sage.search.fingerprint import combine, digest_values
from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid
from sage.search.segments import (
    coincident_intervals,
    hostable_intervals,
    merge_intervals,
    subtract_intervals,
)

# A lag vector that violates the pairwise floor is redrawn inside its own strata rather
# than replaced, so the ladder stays stratified. The bound exists only to turn an
# unsatisfiable request into an error instead of a hang.
_MAX_PLACEMENT_ATTEMPTS: int = 64


@dataclass(frozen=True)
class Slide:
    """
    One lag assignment. ``slide_id == 0`` is zero-lag.

    ``offsets_s`` maps every detector in the network to its lag, including the reference
    at zero, so a slide is self-describing and no caller has to know which detector was
    held fixed. The mapping is a read-only view rather than the dict it was built from:
    ``livetime_s`` was measured for these offsets and no others, so a slide whose lags
    could be edited afterwards would report livetime belonging to a different slide.
    """

    slide_id: int
    offsets_s: Mapping[str, float]
    n_windows: int
    livetime_s: float

    def __post_init__(self) -> None:
        """Freeze the offsets, which ``frozen=True`` does not reach into."""
        object.__setattr__(self, "offsets_s", MappingProxyType(dict(self.offsets_s)))

    def __hash__(self) -> int:
        """Hashable, so slides can be collected in sets and keyed on."""
        return hash(
            (
                self.slide_id,
                tuple(sorted(self.offsets_s.items())),
                self.n_windows,
                self.livetime_s,
            )
        )


@dataclass
class SlidePlan:
    """
    The full ladder for one run, with per-slide livetime measured, not derived.

    ``background_livetime_s`` is the sum over the **slid** slides; the zero-lag slide is
    carried in ``slides`` as the foreground and reported separately. Per-slide retention
    falls with lag, so ``n_slides * T_zerolag`` is never a valid substitute for either.

    ``keep_threshold`` is the statistic above which a background trigger is written. It
    is computed once from the complete zero-lag histogram, before any slide job starts,
    and frozen here so every slide is thresholded on the same number -- a threshold
    re-derived per job would vary with whatever that job happened to see.
    """

    slides: List[Slide]
    reference_detector: str
    seed: int
    min_separation_s: float
    tau_max_s: float
    keep_threshold: Optional[float] = None

    @classmethod
    def build(
        cls,
        geometry: SearchGeometry,
        segments_by_detector: dict,
        n_slides: int,
        reference_detector: str = "H1",
        min_separation_s: float = 20.0,
        tau_max_s: float = 8192.0,
        guard_s: float = 4.0,
        seed: int = 0,
        keep_threshold: Optional[float] = None,
    ) -> "SlidePlan":
        """
        Draw stratified lags and measure each slide's coincident livetime.

        ``n_slides`` counts slid slides; the zero-lag slide is always present as
        ``slide_id == 0`` in addition to them, so the plan describes the foreground and
        the background it will be compared against in one object.

        ``min_separation_s`` is a request for a wider floor than the network's physical
        one. The physical floor always applies, and the value the ladder was actually
        drawn against is what the plan records, so a configuration that asked for less
        cannot be mistaken later for what was used.
        """
        if not segments_by_detector:
            raise ValueError("no detectors given")
        detectors = list(segments_by_detector)
        if reference_detector not in segments_by_detector:
            raise ValueError(
                f"reference detector {reference_detector!r} is not in the network "
                f"{detectors}; slides would be measured against a detector the search "
                "does not read"
            )
        if n_slides < 0:
            raise ValueError(f"n_slides must not be negative, got {n_slides}")
        if n_slides > 0 and len(detectors) < 2:
            raise ValueError(
                f"the network {detectors} has nothing to slide against "
                f"{reference_detector!r}, so it has no background, but {n_slides} "
                "slides were requested; a single-detector search needs a background "
                "estimated some other way, not a ladder of zero-lag copies"
            )

        separation_s = max(
            float(min_separation_s),
            minimum_separation_s(geometry, detectors, guard_s),
        )
        if tau_max_s <= separation_s:
            raise ValueError(
                f"tau_max_s ({tau_max_s}) must exceed the minimum separation "
                f"({separation_s} s, the wider of the requested {min_separation_s} s and "
                "the network's physical floor), or no lag is admissible"
            )

        # Reference first, so the offsets and the coincidence are read in its frame.
        ordered = {reference_detector: segments_by_detector[reference_detector]}
        ordered.update(
            {d: s for d, s in segments_by_detector.items() if d != reference_detector}
        )
        zero_lag = coincident_intervals(ordered)
        if not zero_lag:
            raise ValueError(
                f"the network {detectors} has no coincident data, so no slide can be "
                "measured"
            )
        span_s = zero_lag[-1][1] - zero_lag[0][0]
        if separation_s >= span_s:
            raise ValueError(
                f"a minimum separation of {separation_s} s is not shorter than the "
                f"{span_s} s the network spans, so every slide would retain nothing"
            )

        # Each detector's hostable set and the reference's own union depend on the
        # segments alone, not on the lag, so they are built once here rather than
        # rebuilt inside every slide. On O3a that is 47k segments swept once instead of
        # once per slide.
        context = _MeasurementContext.of(geometry, ordered, reference_detector)

        slides = [
            context.measure(0, {d: 0.0 for d in ordered}),
        ]
        followers = [d for d in ordered if d != reference_detector]
        lags = stratified_lags(
            n_slides,
            len(followers),
            separation_s,
            float(tau_max_s),
            geometry.stride_samples,
            geometry.sample_rate,
            seed,
        )
        for index, row in enumerate(lags):
            offsets = {d: 0.0 for d in ordered}
            offsets.update({d: float(tau) for d, tau in zip(followers, row)})
            slides.append(context.measure(index + 1, offsets))

        if n_slides > 0 and not any(s.livetime_s > 0.0 for s in slides[1:]):
            raise ValueError(
                f"every one of the {n_slides} slides retains no livetime; the lags "
                f"drawn over [{separation_s}, {tau_max_s}] s exceed what the "
                f"{span_s} s of data can support"
            )
        return cls(
            slides=slides,
            reference_detector=reference_detector,
            seed=int(seed),
            min_separation_s=separation_s,
            tau_max_s=float(tau_max_s),
            keep_threshold=None if keep_threshold is None else float(keep_threshold),
        )

    @property
    def background_livetime_s(self) -> float:
        """
        Exact ``T_b``: the summed livetime of the **slid** slides.

        The zero-lag slide is deliberately excluded. It is the foreground -- the time
        candidates are actually drawn from -- so counting it here would make the same
        seconds both the exposure a candidate was found in and part of the exposure its
        false-alarm rate is divided by. ``far.py`` counts ``n_b`` over slid triggers
        alone, and the numerator and denominator of ``(1 + n_b) / T_b`` have to describe
        the same time; a ``T_b`` inflated by one zero-lag slide biases every rate low by
        ``1 / (n + 1)``, which is the direction that makes a candidate look more
        significant than it is. That is 1.67 per cent at 82 slides and a factor of two
        at one.
        """
        return float(
            sum(slide.livetime_s for slide in self.slides if slide.slide_id != 0)
        )

    @property
    def foreground_livetime_s(self) -> float:
        """Zero-lag livetime: the exposure the search reports candidates from."""
        return float(
            sum(slide.livetime_s for slide in self.slides if slide.slide_id == 0)
        )

    def __iter__(self) -> Iterator[Slide]:
        """Iterate slides in id order."""
        return iter(sorted(self.slides, key=lambda slide: slide.slide_id))

    def save(self, path: str | Path) -> None:
        """
        Write ``slides/slide_plan.h5``, including the frozen keep threshold.

        Per-slide livetimes are what the file holds; the total is written beside them as
        a checksum over those numbers, not as the quantity of record. Storing the
        measured per-slide values keeps a reloaded plan reproducing the background
        livetime the campaign was run with rather than a fresh estimate of it, and lets
        :meth:`load` refuse a file whose slide list was truncated part-way through a
        write.

        ``n_slides`` is written with the meaning :meth:`build` gives it -- the number of
        *slid* slides -- and the stored row count, which is one larger, goes separately
        under ``n_records``.
        """
        import h5py

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        detectors = _detector_order(self.reference_detector, self.slides)
        ordered = list(self)
        with h5py.File(target, "w") as handle:
            handle.attrs.update(
                {
                    "reference_detector": self.reference_detector,
                    "detectors": list(detectors),
                    "seed": int(self.seed),
                    "min_separation_s": float(self.min_separation_s),
                    "tau_max_s": float(self.tau_max_s),
                    "n_slides": sum(1 for s in ordered if s.slide_id != 0),
                    "n_records": len(ordered),
                    "background_livetime_s": self.background_livetime_s,
                    "foreground_livetime_s": self.foreground_livetime_s,
                    # NaN rather than a missing attribute: a campaign that has not yet
                    # frozen a threshold is a state the file has to be able to express,
                    # and an absent attribute is indistinguishable from an old file.
                    "keep_threshold": (
                        float("nan")
                        if self.keep_threshold is None
                        else float(self.keep_threshold)
                    ),
                }
            )
            handle.create_dataset(
                "slide_id", data=np.array([s.slide_id for s in ordered], dtype=np.int64)
            )
            handle.create_dataset(
                "n_windows", data=np.array([s.n_windows for s in ordered], dtype=np.int64)
            )
            handle.create_dataset(
                "livetime_s",
                data=np.array([s.livetime_s for s in ordered], dtype=np.float64),
            )
            handle.create_dataset(
                "offsets_s",
                data=np.array(
                    [[s.offsets_s[d] for d in detectors] for s in ordered],
                    dtype=np.float64,
                ).reshape(len(ordered), len(detectors)),
            )

    @classmethod
    def load(cls, path: str | Path) -> "SlidePlan":
        """
        Read a persisted slide plan.

        The stored total is checked against the sum of the stored slides rather than
        replacing it, so a file truncated part-way through a write is refused instead of
        supplying a background livetime no slide list supports. A missing attribute or
        dataset is refused the same way and by name: half a plan read as a whole one
        would put a background livetime into a false-alarm rate that no slide measured.

        The comparison carries a tolerance scaled to the total, because the sum is
        floating point and its exact value depends on the order the slides are added in,
        which is not a property of the file.
        """
        import h5py

        target = Path(path)
        with h5py.File(target, "r") as handle:
            for name in ("reference_detector", "detectors", "seed", "min_separation_s",
                         "tau_max_s", "background_livetime_s"):
                if name not in handle.attrs:
                    raise ValueError(
                        f"{target} carries no {name!r} attribute; it is not a complete "
                        "slide plan and the background it describes cannot be trusted"
                    )
            for name in ("slide_id", "n_windows", "livetime_s", "offsets_s"):
                if name not in handle:
                    raise ValueError(
                        f"{target} is missing the {name!r} dataset; the file was "
                        "truncated part-way through a write"
                    )
            detectors = [str(d) for d in handle.attrs["detectors"]]
            slide_id = np.asarray(handle["slide_id"])
            n_windows = np.asarray(handle["n_windows"])
            livetime_s = np.asarray(handle["livetime_s"])
            offsets_s = np.asarray(handle["offsets_s"])
            sizes = {
                "n_windows": n_windows.shape[0],
                "livetime_s": livetime_s.shape[0],
                "offsets_s": offsets_s.shape[0],
            }
            ragged = {k: v for k, v in sizes.items() if v != slide_id.size}
            if ragged:
                raise ValueError(
                    f"{target} holds {slide_id.size} slide ids against {ragged}; the "
                    "file was truncated part-way through a write"
                )
            threshold = float(handle.attrs.get("keep_threshold", float("nan")))
            plan = cls(
                slides=[
                    Slide(
                        slide_id=int(slide_id[i]),
                        offsets_s={
                            d: float(offsets_s[i, j]) for j, d in enumerate(detectors)
                        },
                        n_windows=int(n_windows[i]),
                        livetime_s=float(livetime_s[i]),
                    )
                    for i in range(slide_id.size)
                ],
                reference_detector=str(handle.attrs["reference_detector"]),
                seed=int(handle.attrs["seed"]),
                min_separation_s=float(handle.attrs["min_separation_s"]),
                tau_max_s=float(handle.attrs["tau_max_s"]),
                keep_threshold=None if np.isnan(threshold) else threshold,
            )
            stored = float(handle.attrs["background_livetime_s"])
        measured = plan.background_livetime_s
        if abs(measured - stored) > 1e-9 * max(1.0, abs(stored)):
            raise ValueError(
                f"{target} records a background livetime of {stored} s against "
                f"{measured} s of stored slides; the file is incomplete"
            )
        return plan


def remeasure_livetimes(
    plan: "SlidePlan",
    geometry: SearchGeometry,
    segments_by_detector: dict,
    vetoes_by_detector: dict,
) -> "SlidePlan":
    """
    Re-measure every slide's livetime with stretches of detector time vetoed out.

    This is what makes an exclusive or hierarchically-removed background a *rate*. Taking
    events out of the numerator while the denominator still describes the time they were
    taken from reports every rate low by the vetoed fraction, in the direction that makes
    candidates look more significant. PyCBC reduces its livetime for the same reason.

    Measured, not scaled. A veto in one detector removes a different amount of coincident
    time from each slide, because under each lag the vetoed stretch pairs with different
    data in the other detector, which may or may not be live there. Subtracting a single
    scalar from the total -- the closed-form route -- assumes that loss is uniform across
    the ladder, and it is not. Every slide is rebuilt through the same lattice it was
    measured on originally, so the reduced livetime is the time the slide would actually
    contribute.

    A window is removed if it *overlaps* a vetoed stretch, not merely if it starts inside
    one: a window carries ``geometry.window_s`` of data, so a start one window-length
    before the veto still reads vetoed samples. Each veto is therefore widened backwards
    by a full window before being subtracted from the hostable set. Failing to widen would
    leave up to one window of vetoed data in the background per veto edge.

    Parameters
    ----------
    vetoes_by_detector : dict
        Detector name to a list of ``(start, end)`` GPS intervals to remove. Detectors
        absent from the mapping are left untouched. The intervals are merged before use,
        so overlapping vetoes -- which a cluster of nearby events produces -- cost their
        union rather than their sum.

    Returns
    -------
    SlidePlan
        A new plan with the same lags, seed and keep threshold, and re-measured
        ``n_windows`` and ``livetime_s`` on every slide including the zero-lag one.

    Notes
    -----
    Costs one lattice rebuild per slide, the same work the original ladder cost -- about
    22 s for the full 82-slide O3a ladder. That is the price of not having a closed form,
    and it is paid once per removal mode.
    """
    for detector in vetoes_by_detector:
        if detector not in segments_by_detector:
            raise ValueError(
                f"vetoes were given for {detector!r}, which is not in the network "
                f"{sorted(segments_by_detector)}; a veto on a detector the search does "
                "not read would silently remove nothing"
            )
    context = _MeasurementContext.of(
        geometry, segments_by_detector, plan.reference_detector
    )
    window_s = float(geometry.window_s)
    hostable = dict(context.hostable_by_detector)
    for detector, intervals in vetoes_by_detector.items():
        merged = merge_intervals(list(intervals))
        if not merged:
            continue
        widened = [(start - window_s, end) for start, end in merged]
        hostable[detector] = subtract_intervals(hostable[detector], widened)
    context = _MeasurementContext(
        geometry=geometry,
        segments_by_detector=segments_by_detector,
        reference_detector=plan.reference_detector,
        hostable_by_detector=hostable,
        reference_union=context.reference_union,
    )
    return SlidePlan(
        slides=[
            context.measure(slide.slide_id, dict(slide.offsets_s)) for slide in plan
        ],
        reference_detector=plan.reference_detector,
        seed=plan.seed,
        min_separation_s=plan.min_separation_s,
        tau_max_s=plan.tau_max_s,
        keep_threshold=plan.keep_threshold,
    )


def stratified_lags(
    n_slides: int,
    n_lagged_detectors: int,
    min_separation_s: float,
    tau_max_s: float,
    stride_samples: int,
    sample_rate: float,
    seed: int,
) -> np.ndarray:
    """
    Draw ``n_slides`` lag vectors stratified over ``[min_separation, tau_max]``.

    Parameters
    ----------
    n_lagged_detectors : int
        One fewer than the network size: the reference detector is never slid.

    Returns
    -------
    ndarray
        ``(n_slides, n_lagged_detectors)`` of lags in seconds.

    Notes
    -----
    Lags are multiples of the stride, so a slid window lands on the same lattice and no
    resampling is implied. Zero lag is excluded, and every drawn vector satisfies
    :func:`pairwise_separations_ok`, which is the constraint that distinguishes a network
    of three from a network of two.

    The admissible stride multiples are partitioned into ``n_slides`` contiguous blocks
    and one multiple is drawn from each, independently per detector. Partitioning the
    lattice rather than the interval keeps the drawn lags distinct however coarse the
    stride is, and spending one slide per block is what spreads the ladder over the whole
    range instead of packing it against either end.
    """
    if n_slides < 0:
        raise ValueError(f"n_slides must not be negative, got {n_slides}")
    if n_lagged_detectors < 0:
        raise ValueError(
            f"n_lagged_detectors must not be negative, got {n_lagged_detectors}"
        )
    if min_separation_s <= 0:
        raise ValueError(
            f"min_separation_s must be positive, got {min_separation_s}; the floor is "
            "physical -- window content plus the longest baseline plus the guard -- and "
            "a non-positive one admits a zero lag, which is the foreground"
        )
    if tau_max_s <= min_separation_s:
        raise ValueError(
            f"tau_max_s ({tau_max_s}) must exceed min_separation_s "
            f"({min_separation_s}), or no lag is admissible"
        )
    if n_slides == 0 or n_lagged_detectors == 0:
        return np.zeros((n_slides, n_lagged_detectors), dtype=float)
    # Sorted lags in one vector are separated pairwise, so the largest needed is
    # n_lagged * min_separation; below that the network cannot be slid at all.
    if tau_max_s < n_lagged_detectors * min_separation_s:
        raise ValueError(
            f"a network with {n_lagged_detectors} slid detectors needs lags up to "
            f"{n_lagged_detectors * min_separation_s} s to keep every pair "
            f"{min_separation_s} s apart, which tau_max_s ({tau_max_s}) does not allow"
        )

    stride_s = stride_samples / sample_rate
    k_min = int(np.ceil(min_separation_s / stride_s))
    while k_min * stride_s < min_separation_s:
        k_min += 1
    k_max = int(np.floor(tau_max_s / stride_s))
    while k_max * stride_s > tau_max_s:
        k_max -= 1
    n_points = k_max - k_min + 1
    if n_points < n_slides:
        raise ValueError(
            f"[{min_separation_s}, {tau_max_s}] s holds {max(n_points, 0)} multiples of "
            f"the {stride_s} s stride, which cannot supply {n_slides} distinct lags"
        )

    rng = np.random.default_rng(seed)
    edges = k_min + (np.arange(n_slides + 1, dtype=np.int64) * n_points) // n_slides
    strata = np.stack(
        [rng.permutation(n_slides) for _ in range(n_lagged_detectors)], axis=1
    )
    multiples = rng.integers(edges[strata], edges[strata + 1])
    lags = multiples * stride_s

    ok = pairwise_separations_ok(lags, min_separation_s)
    for attempt in range(_MAX_PLACEMENT_ATTEMPTS):
        rows = np.flatnonzero(~ok)
        if rows.size == 0:
            break
        if n_slides > 1 and attempt % 2 == 1:
            # Two detectors drawn from neighbouring blocks stay close however they are
            # redrawn, so alternate attempts exchange a block with another slide instead.
            # Exchanging keeps one slide per block, which redrawing a block would not.
            partners = rng.integers(0, n_slides, size=rows.size)
            columns = rng.integers(0, n_lagged_detectors, size=rows.size)
            for row, partner, column in zip(rows, partners, columns):
                strata[[row, partner], column] = strata[[partner, row], column]
            rows = np.union1d(rows, partners)
        block = strata[rows]
        multiples[rows] = rng.integers(edges[block], edges[block + 1])
        lags[rows] = multiples[rows] * stride_s
        ok = pairwise_separations_ok(lags, min_separation_s)
    if not ok.all():
        raise ValueError(
            f"{int((~ok).sum())} of {n_slides} lag vectors still place two detectors "
            f"within {min_separation_s} s after {_MAX_PLACEMENT_ATTEMPTS} attempts; the "
            f"stratified pairing could not place {n_slides} vectors over "
            f"[{min_separation_s}, {tau_max_s}] s under the all-pairs floor for "
            f"{n_lagged_detectors} slid detectors. Raise tau_max_s or ask for fewer "
            "slides -- the request may still be satisfiable by a ladder that is not "
            "one-per-stratum, which this is"
        )
    return lags


def pairwise_separations_ok(lags: np.ndarray, min_separation_s: float) -> np.ndarray:
    """
    Whether each lag vector keeps every detector pair apart.

    Checks the lags themselves, which separate each slid detector from the reference, and
    every difference between them, which separates the slid detectors from each other. A
    vector failing either would place two detectors close enough in slid time for a real
    coincidence to survive into the background.

    Parameters
    ----------
    lags : ndarray
        ``(n_slides, n_lagged_detectors)``, or ``(n_lagged_detectors,)`` for one slide.

    Returns
    -------
    ndarray
        Boolean, one entry per slide.
    """
    array = np.atleast_2d(np.asarray(lags, dtype=float))
    if array.ndim != 2:
        raise ValueError(
            f"lags must be one vector or an array of them, got shape {array.shape}"
        )
    n_slides, n_lagged = array.shape
    ok = np.ones(n_slides, dtype=bool)
    if n_lagged == 0:
        return ok
    ok &= np.all(np.abs(array) >= min_separation_s, axis=1)
    for i in range(n_lagged):
        for j in range(i + 1, n_lagged):
            ok &= np.abs(array[:, i] - array[:, j]) >= min_separation_s
    return ok


def minimum_separation_s(
    geometry: SearchGeometry, detectors: Sequence[str], guard_s: float
) -> float:
    """
    Smallest admissible lag: window content + light travel + guard.

    The light-travel term is the maximum over every detector pair, so adding Virgo to a
    two-detector network raises the floor from about 10 ms to about 27 ms.

    The window term is the analysis content, not the padded window: a lag shorter than
    the content would leave a slid window analysing part of the stretch its own zero-lag
    window analysed, so the same second of data would appear on both sides of the
    coincidence.
    """
    if guard_s < 0:
        raise ValueError(f"guard_s must not be negative, got {guard_s}")
    return float(
        geometry.signal_length_s + geometry.max_light_travel_s(detectors) + guard_s
    )


def _detector_order(reference_detector: str, slides: Sequence[Slide]) -> Tuple[str, ...]:
    """Network order, reference first, taken from the offsets a slide carries."""
    names = list(slides[0].offsets_s) if slides else [reference_detector]
    if reference_detector not in names:
        raise ValueError(
            f"reference detector {reference_detector!r} carries no offset in the plan's "
            f"slides ({names})"
        )
    return (reference_detector, *[d for d in names if d != reference_detector])


@dataclass(frozen=True)
class _MeasurementContext:
    """
    What every slide of one network shares, computed once.

    A slide's livetime depends on the lag, but the sets it is derived from do not: each
    detector's hostable intervals and the reference detector's own data union are
    properties of the segment lists. Rebuilding them per slide made the ladder scale
    with depth in work that never changed.
    """

    geometry: SearchGeometry
    segments_by_detector: dict
    reference_detector: str
    hostable_by_detector: dict
    reference_union: List[Tuple[float, float]]

    @classmethod
    def of(
        cls,
        geometry: SearchGeometry,
        segments_by_detector: dict,
        reference_detector: str,
    ) -> "_MeasurementContext":
        """Sweep each detector's segments once."""
        return cls(
            geometry=geometry,
            segments_by_detector=segments_by_detector,
            reference_detector=reference_detector,
            hostable_by_detector={
                detector: hostable_intervals(segments, geometry.window_samples)
                for detector, segments in segments_by_detector.items()
            },
            reference_union=coincident_intervals(
                {reference_detector: segments_by_detector[reference_detector]}
            ),
        )

    def measure(self, slide_id: int, offsets_s: dict) -> Slide:
        """
        Build the window lattice for one lag assignment and read its livetime off it.

        Measured through the same lattice the background will actually be scored on, so
        the livetime a slide reports is the time it contributes and not an estimate of
        it. An offset says where a follower reads relative to the reference, so the
        follower's data is pulled back by ``-offset`` into the reference frame.

        The restriction handed to the lattice is the reference detector's own data,
        rather than the slid coincidence: the lattice intersects every detector's
        hostable set through its own offset anyway, and a hostable set is contained in
        the data it came from, so computing the slid coincidence first only recomputed a
        bound the lattice was about to impose.

        The coverage decomposition is not requested. A slide needs its window count and
        nothing else, and on the O3a lattice that decomposition is 374 s and 11.8 GB per
        slide against 0.06 s for the algebra -- at 82 slides, the difference between
        eight hours and a few seconds.
        """
        grid = AnalysisGrid.build(
            self.geometry,
            self.segments_by_detector,
            self.reference_union,
            offsets_s=offsets_s,
            slide_id=slide_id,
            reference_detector=self.reference_detector,
            coverage=False,
            hostable_by_detector=self.hostable_by_detector,
        )
        return Slide(
            slide_id=slide_id,
            offsets_s=dict(offsets_s),
            n_windows=len(grid),
            livetime_s=grid.livetime_s,
        )


# Local, so that reading a slide plan does not import the significance layer.
_SECONDS_PER_JULIAN_YEAR: float = 31557600.0

def _same_ladder(left: "SlidePlan", right: "SlidePlan") -> bool:
    """
    Whether two plans assign the same lags to the same slide ids.

    Compared slide by slide rather than by summed livetime: the sum is invariant under
    reordering the ladder, and a reordered ladder is precisely the case where carrying a
    frozen threshold forward would attach it to the wrong slides.
    """
    if len(left.slides) != len(right.slides):
        return False
    return all(
        a.slide_id == b.slide_id and dict(a.offsets_s) == dict(b.offsets_s)
        for a, b in zip(left.slides, right.slides)
    )


def run(spec, **kwargs) -> dict:
    """
    Stage driver: draw the stratified lag ladder and measure each slide's livetime.

    Writes ``slides/slide_plan.h5``. The plan is persisted rather than rebuilt because it
    is the *denominator*: every false-alarm rate divides by the livetime summed from it,
    and a ladder redrawn from a seed on a machine with a different numpy would give a
    background time nothing could reproduce.

    ``keep_threshold`` is deliberately left unset here. It must be frozen from the
    **complete** zero-lag histogram, and ``zerolag`` is a sibling of this stage rather than
    an input to it -- both depend only on ``grid``, so they can be submitted together. The
    ``background`` stage depends on both and is where the threshold is stamped into the
    plan, immediately before any slide is scored against it.
    """
    from sage.search.segments import load_segments

    geometry = spec.geometry_object()
    segments = {
        detector: load_segments(
            Path(spec.data.release_dir)
            / f"data_{detector}_{spec.data.observing_run}_segments.json"
        )
        for detector in spec.data.detectors
    }
    plan = SlidePlan.build(
        geometry,
        segments,
        n_slides=int(spec.slides.n_slides),
        reference_detector=spec.slides.reference_detector,
        min_separation_s=float(spec.slides.min_separation_s),
        tau_max_s=float(spec.slides.tau_max_s),
        guard_s=float(spec.slides.guard_s),
        seed=int(spec.slides.seed),
    )
    target = spec.path("slides", "slide_plan.h5")
    target.parent.mkdir(parents=True, exist_ok=True)

    # The keep threshold lives in this file but does not belong to this stage: background
    # freezes it here from the complete zero-lag histogram, after slides has run. A
    # re-run therefore has to carry it across, and only when the ladder it was frozen
    # against is the same ladder -- a threshold is a property of one lattice, and one
    # rebuilt over different lags is a number from a different campaign. Overwriting it
    # with None was silent: nothing downstream re-reads the plan until a slide is scored,
    # and that slide would then be thresholded against whatever background decides next,
    # while the slides already on disk were thresholded against the value now gone.
    if target.is_file():
        previous = SlidePlan.load(target)
        if previous.keep_threshold is not None and _same_ladder(previous, plan):
            plan = dataclasses.replace(
                plan, keep_threshold=float(previous.keep_threshold)
            )
    plan.save(target)

    background_s = float(plan.background_livetime_s)
    foreground_s = float(plan.foreground_livetime_s)
    retention = (
        background_s / (int(spec.slides.n_slides) * foreground_s)
        if foreground_s > 0 and spec.slides.n_slides
        else float("nan")
    )
    return {
        "plan": str(target),
        "n_slides": int(spec.slides.n_slides),
        "background_livetime_s": background_s,
        "background_livetime_yr": background_s / _SECONDS_PER_JULIAN_YEAR,
        "foreground_livetime_s": foreground_s,
        # Measured, never n * T_zerolag. Retention falls as the ladder lengthens because a
        # lag moves a detector's data off the far end of the run, and the closed form
        # assumes it does not.
        "mean_slide_retention": retention,
        # Per-slide lags and per-slide livetimes, in slide_id order. The summed
        # background livetime is invariant under reordering the ladder, so a plan whose
        # slide_id -> lag map changed would keep the same fingerprint while every
        # background shard on disk, each named by slide_id, now describes a different
        # lag. The keep threshold is deliberately absent: background freezes it into this
        # same file after slides has run, and digesting it would move this stage's
        # fingerprint as a consequence of its own consumer's write.
        "fingerprint": combine(
            f"{background_s:.6f}",
            spec.slides.n_slides,
            spec.slides.seed,
            digest_values(
                {
                    "slide_id": np.asarray(
                        [slide.slide_id for slide in plan.slides], dtype=np.int64
                    ),
                    "offsets_s": {
                        detector: np.asarray(
                            [
                                slide.offsets_s.get(detector, 0.0)
                                for slide in plan.slides
                            ],
                            dtype=np.float64,
                        )
                        for detector in sorted(spec.data.detectors)
                    },
                    "livetime_s": np.asarray(
                        [slide.livetime_s for slide in plan.slides], dtype=np.float64
                    ),
                    "n_windows": np.asarray(
                        [slide.n_windows for slide in plan.slides], dtype=np.int64
                    ),
                }
            ),
        ),
    }

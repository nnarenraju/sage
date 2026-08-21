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

    Two pairings are expressible, and which one is in use is told by which field is set.
    ``offsets_s`` maps every detector to a **GPS lag**, including the reference at zero,
    so a slide is self-describing and no caller has to know which detector was held
    fixed. ``window_shift`` instead maps each follower to an integer **shift along the
    analysed lattice**: follower ordinal ``(i + k) mod N`` is paired with reference
    ordinal ``i``. Every lattice ordinal is hostable in every detector by construction,
    so a shifted pairing loses no livetime at all, where a GPS lag drops whatever it
    pushes into a gap. The mapping is a read-only view rather than the dict it was built from:
    ``livetime_s`` was measured for these offsets and no others, so a slide whose lags
    could be edited afterwards would report livetime belonging to a different slide.
    """

    slide_id: int
    offsets_s: Mapping[str, float]
    n_windows: int
    livetime_s: float
    window_shift: Optional[Mapping[str, int]] = None

    def __post_init__(self) -> None:
        """Freeze the offsets, which ``frozen=True`` does not reach into."""
        object.__setattr__(self, "offsets_s", MappingProxyType(dict(self.offsets_s)))
        if self.window_shift is not None:
            object.__setattr__(
                self, "window_shift", MappingProxyType(dict(self.window_shift))
            )

    def __hash__(self) -> int:
        """Hashable, so slides can be collected in sets and keyed on."""
        return hash(
            (
                self.slide_id,
                tuple(sorted(self.offsets_s.items())),
                tuple(sorted((self.window_shift or {}).items())),
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
    method: str = "ladder"

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
        method: str = "ladder",
        spacing: str = "even",
        max_shift_s: Optional[float] = None,
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

        ``method`` picks how the background is paired.

        ``"ladder"``
            A stratified ladder of GPS lags bounded by ``tau_max_s``. Retention falls
            with lag, because a lag pushes some of the follower's time into a gap.

        ``"roll"``
            Shifts along the analysed lattice, as sgwc-1's background does. Every lattice
            ordinal is hostable in every detector by construction, so **no livetime is
            lost at any shift** -- and because a shift of ``N/(K+1)`` pairs stretches a
            large fraction of the run apart, the slides are far closer to independent
            than a ladder bounded at hours can be. ``tau_max_s`` does not apply.
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
        if method not in SLIDE_METHODS:
            raise ValueError(
                f"unknown slide method {method!r}; known methods are "
                f"{list(SLIDE_METHODS)}"
            )
        if method == "ladder" and tau_max_s <= separation_s:
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
        if method == "roll":
            slides.extend(
                _rolled_slides(
                    geometry,
                    ordered,
                    reference_detector,
                    followers,
                    slides[0],
                    n_slides,
                    separation_s,
                    seed,
                    spacing,
                    max_shift_s,
                )
            )
        else:
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
            method=str(method),
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
                    "method": str(self.method),
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
            # Written for every plan, zero for a ladder: a reader must not have to
            # decide which pairing a file describes from which datasets are present.
            handle.create_dataset(
                "window_shift",
                data=np.array(
                    [
                        [int((s.window_shift or {}).get(d, 0)) for d in detectors]
                        for s in ordered
                    ],
                    dtype=np.int64,
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
            # Absent in a plan written before the roll pairing existed, which is a ladder.
            shifts = (
                np.asarray(handle["window_shift"])
                if "window_shift" in handle
                else np.zeros_like(offsets_s, dtype=np.int64)
            )
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
                        window_shift=(
                            {d: int(shifts[i, j]) for j, d in enumerate(detectors)}
                            if shifts[i].any()
                            else None
                        ),
                    )
                    for i in range(slide_id.size)
                ],
                reference_detector=str(handle.attrs["reference_detector"]),
                seed=int(handle.attrs["seed"]),
                min_separation_s=float(handle.attrs["min_separation_s"]),
                tau_max_s=float(handle.attrs["tau_max_s"]),
                keep_threshold=None if np.isnan(threshold) else threshold,
                method=str(handle.attrs.get("method", "ladder")),
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


SLIDE_METHODS: Tuple[str, ...] = ("ladder", "roll")


def cache_bounded_shift(
    budget_bytes: float,
    n_detectors: int,
    bytes_per_window_per_detector: int,
    stride_s: float,
) -> Tuple[int, float]:
    """
    Largest shift a frontend cache of this size can serve, in windows and in seconds.

    The cache holds one block plus the halo every slide reaches into. A lag ladder's halo
    is its largest lag, which is small. A roll's is its largest *shift*, and an unbounded
    roll -- shifts of ``N/(K+1)``, which is what makes it decorrelate so well -- reaches
    across the whole run, so the working set is the whole run's features and no cache can
    hold it. Measured on the production network: features are 20.5 KB per window per
    detector in bfloat16, so the full O3a lattice is **3.9 TB** and even the 14.5 d
    development lattice is 526 GB.

    Bounding the shift is what lets the two coexist. Every slide's shift must lie inside
    ``[-M, +M]`` so that serving reference block ``b`` needs followers only from
    ``[b - M, b + M]``; the halo is therefore ``2M`` and the affordable ``M`` is half of
    what the budget holds.

    Returns
    -------
    (max_shift_windows, max_shift_s)
        Zero when the budget cannot hold a useful halo, which is the honest answer rather
        than a shift of one window.
    """
    per_window = float(bytes_per_window_per_detector) * int(n_detectors)
    if per_window <= 0:
        raise ValueError("features must occupy a positive number of bytes per window")
    halo = float(budget_bytes) / per_window
    max_shift = int(halo // 2)
    return max_shift, max_shift * float(stride_s)


def rolled_shifts(
    n_slides: int,
    n_followers: int,
    n_windows: int,
    min_shift: int,
    seed: int = 0,
    spacing: str = "even",
    max_shift: Optional[int] = None,
) -> np.ndarray:
    """
    Integer shifts along the analysed lattice, one row per slide.

    The generalisation of sgwc-1's background (``make_background_samples.py``), which
    circularly rolls the follower detector's window list by ``N//2`` and ``N//3`` and so
    takes exactly two slides. The same construction supports any number: ``K`` shifts
    spread over the list give ``K`` slides, each still pairing stretches of data that are
    a large fraction of the run apart.

    ``spacing`` selects how the shifts are placed.

    ``"even"``
        Evenly over the admissible range: ``low + j * (high - low) / (K + 1)`` for
        ``j = 1 .. K``. With no floor and no bound that is ``j * N / (K + 1)``, which is
        sgwc-1's construction; the floor and the cache bound narrow the range rather than
        change the shape. Deterministic, maximally spread, and every shift distinct.
    ``"random"``
        Drawn uniformly from the admissible range without replacement. Available because
        an even ladder is a regular structure and a regular structure can in principle
        resonate with a periodic artefact; there is no evidence of one here, so it is not
        the default.

    ``min_shift`` is the floor in windows. A shift of ``k`` separates the paired windows
    by at least ``k`` strides -- exactly that inside a contiguous stretch, and more
    wherever a gap falls between them -- so this is what enforces the minimum time
    separation. The caller measures what was achieved regardless, because the bound holds
    only for a sorted lattice.

    Returns
    -------
    ndarray of int
        ``(n_slides, n_followers)``. Each follower gets its own shift, so a
        three-detector network does not roll two detectors together and re-pair them at
        zero lag with each other.
    """
    if n_slides < 0:
        raise ValueError(f"n_slides must not be negative, got {n_slides}")
    if n_followers < 1 and n_slides > 0:
        raise ValueError("a roll needs at least one follower detector to shift")
    if n_slides == 0:
        return np.zeros((0, max(n_followers, 0)), dtype=np.int64)
    if spacing not in ("even", "random"):
        raise ValueError(f"spacing must be 'even' or 'random', got {spacing!r}")

    low, high = int(min_shift), int(n_windows) - int(min_shift)
    if max_shift is not None:
        # Bounded so a frontend cache can hold the halo every slide reaches into. Without
        # a bound the shifts span the run and no cache is possible; see
        # :func:`cache_bounded_shift`.
        high = min(high, int(min_shift) + int(max_shift))
    if high <= low:
        raise ValueError(
            f"a lattice of {n_windows} windows admits no shift at least {min_shift} "
            "windows from either end; the minimum separation is too large for the data"
        )
    available = high - low
    if n_slides > available:
        raise ValueError(
            f"{n_slides} distinct shifts were asked for but only {available} are "
            f"admissible on a lattice of {n_windows} windows with a floor of "
            f"{min_shift}"
        )

    rng = np.random.default_rng(seed)
    rows = []
    for follower in range(max(n_followers, 1)):
        if spacing == "even":
            # Spread over the admissible range, which is the whole lattice when nothing
            # bounds it and the cache halo when something does.
            shifts = np.round(
                low + np.arange(1, n_slides + 1) * available / (n_slides + 1)
            ).astype(np.int64)
            # Each follower is offset by a further stride of the same ladder, so two
            # followers are never rolled by the same amount and left at zero lag with
            # each other.
            shifts = low + (shifts - low + follower) % available
        else:
            shifts = rng.choice(available, size=n_slides, replace=False) + low
        rows.append(np.sort(shifts.astype(np.int64)))
    return np.column_stack(rows)


def shifted_separations_s(
    reference_gps: np.ndarray, shift: int, stride_s: float
) -> float:
    """
    Smallest time separation a shift actually produces, in seconds.

    The lattice is sorted, so a shift of ``k`` windows separates a pair by at least
    ``k * stride`` and by more wherever a gap falls between them: skipping a gap can only
    push the paired times further apart, never closer. The floor is therefore guaranteed
    by the shift alone, and this measures it rather than deriving it, because the
    guarantee rests on the lattice being sorted -- a property of how the span list was
    assembled, not something enforced here. sgwc-1 checks the same thing the same way
    (``|t_H1 - t_L1| >= 1.0`` s, ``make_background_samples.py``).

    The wrapped pairs are included. Those are the ones the shift says nothing about: at
    the wrap a pair is separated by the whole run minus ``k`` strides, which is large,
    but it is large for a different reason and is worth having in the same number.
    """
    times = np.asarray(reference_gps, dtype=np.float64)
    if times.size == 0:
        raise ValueError("an empty lattice has no separations to measure")
    rolled = np.roll(times, -int(shift))
    return float(np.abs(rolled - times).min())


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


def slides_for_background(
    target_yr: float, foreground_s: float, retention: float = 1.0
) -> int:
    """
    Slides needed to reach a target background depth, given this campaign's foreground.

    The number a campaign should be described by is the depth, not the count: an O3a HL
    search analyses 106.75 days and an O3a HLV one 80.81, so the same slide count gives
    them different backgrounds and their false-alarm rates are not comparable at the same
    ladder length. Asking for years gives every search the same floor.

    ``retention`` is the fraction of the foreground a slide keeps. Exactly one for a roll,
    which loses nothing; below one for a lag ladder, which needs proportionally more
    slides for the same depth. Rounded **up**, because a campaign that asked for ten years
    and got nine and a half has a false-alarm floor it did not ask for.
    """
    import math

    if target_yr <= 0:
        raise ValueError(f"target_yr must be positive, got {target_yr}")
    if foreground_s <= 0:
        raise ValueError(
            "the campaign has no foreground livetime, so no depth of background can be "
            "expressed as a multiple of it"
        )
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must lie in (0, 1], got {retention}")
    per_slide_yr = foreground_s * retention / _SECONDS_PER_JULIAN_YEAR
    return max(1, math.ceil(float(target_yr) / per_slide_yr))


def _rolled_slides(
    geometry,
    ordered: dict,
    reference_detector: str,
    followers: Sequence[str],
    zero_lag: Slide,
    n_slides: int,
    separation_s: float,
    seed: int,
    spacing: str,
    max_shift_s: Optional[float] = None,
) -> List[Slide]:
    """
    Slides paired by shifting along the analysed lattice rather than in GPS.

    The livetime of every rolled slide equals the zero-lag livetime **exactly**, and that
    is a fact about the construction rather than an assumption worth measuring around:
    the lattice admits only ordinals every detector can host, so re-pairing ordinal ``i``
    with ordinal ``(i + k) mod N`` reads real data in every detector for every ``i``.
    Nothing falls into a gap because nothing is moved in time. That is the whole
    advantage over a lag ladder, whose retention falls with lag.

    What *is* measured is the separation each shift achieves. It is bounded below by
    ``k * stride`` on a sorted lattice, so the check confirms the floor rather than
    discovering it -- but it confirms it over every pair, which is what sgwc-1 does too,
    and it is the assertion that would catch a lattice assembled out of order.
    """
    if n_slides == 0:
        return []

    grid = AnalysisGrid.build(
        geometry,
        ordered,
        coincident_intervals(ordered),
        slide_id=0,
        reference_detector=reference_detector,
        coverage=False,
    )
    starts = np.concatenate(
        [span.starts_gps() for span in grid.reference_spans]
    )
    n_windows = starts.size
    min_shift = int(np.ceil(separation_s / geometry.stride_s))
    max_shift = (
        None if max_shift_s is None
        else max(int(float(max_shift_s) / geometry.stride_s), min_shift + 1)
    )
    shifts = rolled_shifts(
        n_slides, len(followers), n_windows, min_shift, seed=seed, spacing=spacing,
        max_shift=max_shift,
    )

    out: List[Slide] = []
    for index, row in enumerate(shifts):
        # The reference is carried at zero, exactly as ``offsets_s`` carries it, so a
        # slide is self-describing and a round trip through the file changes nothing.
        shift_by_detector = {d: 0 for d in ordered}
        shift_by_detector.update({d: int(k) for d, k in zip(followers, row)})
        for detector, shift in shift_by_detector.items():
            if not shift:
                continue
            achieved = shifted_separations_s(starts, shift, geometry.stride_s)
            if achieved < separation_s:
                raise ValueError(
                    f"shifting {detector} by {shift} windows leaves a pair only "
                    f"{achieved:.3f} s apart, inside the {separation_s} s floor. The "
                    "lattice skips the gaps between segments, so a shift is not a fixed "
                    "time separation; raise the floor or draw a different set"
                )
        out.append(
            Slide(
                slide_id=index + 1,
                offsets_s={d: 0.0 for d in ordered},
                n_windows=int(zero_lag.n_windows),
                livetime_s=float(zero_lag.livetime_s),
                window_shift=shift_by_detector,
            )
        )
    return out


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


def _requested_slides(spec, geometry, segments_by_detector) -> int:
    """
    How many slides this campaign wants: its explicit count, or its target depth.

    The depth is turned into a count here rather than in :meth:`SlidePlan.build`, because
    it needs the foreground livetime and that is a property of this campaign's data. A
    roll keeps all of it and a ladder does not, so the two need different counts for the
    same years -- the retention assumed is the method's own.
    """
    if spec.slides.n_slides is not None:
        return int(spec.slides.n_slides)

    zero_lag = AnalysisGrid.build(
        geometry,
        segments_by_detector,
        coincident_intervals(segments_by_detector),
        slide_id=0,
        reference_detector=spec.slides.reference_detector,
        coverage=False,
    )
    method = str(getattr(spec.slides, "method", "ladder"))
    # A roll loses nothing. A ladder's retention depends on its lags and is not known
    # until they are drawn, so 0.9 stands in -- deliberately pessimistic, because
    # overshooting the target costs compute and undershooting it costs a floor the
    # campaign asked for. The plan records what was actually achieved either way.
    retention = 1.0 if method == "roll" else 0.9
    return slides_for_background(
        float(spec.slides.target_background_yr), zero_lag.livetime_s, retention
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
        n_slides=_requested_slides(spec, geometry, segments),
        reference_detector=spec.slides.reference_detector,
        min_separation_s=float(spec.slides.min_separation_s),
        tau_max_s=float(spec.slides.tau_max_s),
        guard_s=float(spec.slides.guard_s),
        method=str(getattr(spec.slides, "method", "ladder")),
        spacing=str(getattr(spec.slides, "spacing", "even")),
        max_shift_s=getattr(spec.slides, "max_shift_s", None),
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
    # Counted off the plan rather than read back from the specification: with a target
    # depth the count is derived, and the number of record is what was actually built.
    n_background = sum(1 for slide in plan.slides if slide.slide_id != 0)
    retention = (
        background_s / (n_background * foreground_s)
        if foreground_s > 0 and n_background
        else float("nan")
    )
    return {
        "plan": str(target),
        "n_slides": n_background,
        "target_background_yr": spec.slides.target_background_yr,
        "background_livetime_s": background_s,
        "background_livetime_yr": background_s / _SECONDS_PER_JULIAN_YEAR,
        "foreground_livetime_s": foreground_s,
        # Measured, never n * T_zerolag. For a ladder the two genuinely differ: retention
        # falls as the ladder lengthens, because a lag moves a detector's data off the far
        # end of the run. For a roll they agree exactly, and that is a property of the
        # construction rather than the closed form being used -- the number still comes
        # from summing the plan.
        "mean_slide_retention": retention,
        # Per-slide pairings and per-slide livetimes, in slide_id order. The summed
        # background livetime is invariant under reordering, so a plan whose
        # slide_id -> pairing map changed would keep the same fingerprint while every
        # background shard on disk, each named by slide_id, now describes a different
        # pairing. For a roll that is the whole of it: every rolled slide reports the
        # same livetime and window count, so the shifts are the only thing left to tell
        # two plans apart. The keep threshold is deliberately absent: background freezes it into this
        # same file after slides has run, and digesting it would move this stage's
        # fingerprint as a consequence of its own consumer's write.
        "fingerprint": combine(
            f"{background_s:.6f}",
            n_background,
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
                    # Carried for the same reason as the lags, and it is the *only* thing
                    # that distinguishes two rolled plans: every rolled slide has the same
                    # livetime and the same window count, so a reassignment of shifts to
                    # slide_ids moves nothing else in this digest while every shard on
                    # disk, named by slide_id, now describes a different pairing.
                    "window_shift": {
                        detector: np.asarray(
                            [
                                (slide.window_shift or {}).get(detector, 0)
                                for slide in plan.slides
                            ],
                            dtype=np.int64,
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

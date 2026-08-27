#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : segments.py
Description   : Segment sidecar ingest, interval algebra and the ownership sweep.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The ``.bin`` release is a concatenation of independent, overlapping chunks:

* ``sample_start_idx`` is contiguous across the whole file (no index gaps), but
  segments are NOT ordered by GPS in the sidecar, and consecutive-in-GPS segments
  overlap by ~15.5994 s (512 s chunks advancing 496.4006 s).
* A GPS instant inside an overlap appears in two segments with DIFFERENT sample
  values, because each chunk was resampled and filtered on its own boundaries.

Consequently GPS is a per-segment coordinate, not a global function of file index.
Reads must never cross a segment boundary, and each GPS instant must be assigned to
exactly one owning segment or it is analysed twice. Because the overlap (15.5994 s)
is slightly smaller than a padded window (16 s), each boundary leaves a ~0.4006 s
band that can host no window start; ``window_hosts`` reports this explicitly rather
than absorbing it silently.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sage.search.fingerprint import combine, digest_values

Interval = Tuple[float, float]


@dataclass(frozen=True)
class Segment:
    """One contiguous chunk of conditioned strain inside a ``.bin`` release."""

    segment_index: int
    detector: str
    observing_run: str
    gps_start: float
    gps_end: float
    sample_rate: float
    nsamples: int
    sample_start_idx: int
    dyn_range_fac: float
    noise_low_freq_cutoff: float
    #: HDF5 path to this segment's samples, for a search-grade release. ``None`` for the
    #: flat ``.bin`` training releases, where ``sample_start_idx`` locates the segment in
    #: one contiguous stream instead. The two layouts are read by different code paths and
    #: this is what distinguishes them, so it is carried rather than inferred from the
    #: directory's contents.
    dataset: Optional[str] = None

    @property
    def duration_s(self) -> float:
        """``gps_end - gps_start``."""
        return self.gps_end - self.gps_start

    def gps_of_local(self, i: int) -> float:
        """GPS time of segment-local sample ``i``."""
        return self.gps_start + i / self.sample_rate

    def local_of_gps(self, gps: float) -> int:
        """
        Segment-local sample index at or after ``gps``.

        Refuses a time outside this segment rather than clamping. GPS is a per-segment
        coordinate here: a time inside an overlap exists in two segments at two different
        local indices, holding two different sample values, so a silently clamped index
        would read the wrong data from the right-looking segment.
        """
        if not (self.gps_start <= gps <= self.gps_end):
            raise ValueError(
                f"{gps} lies outside segment {self.segment_index} "
                f"[{self.gps_start}, {self.gps_end}] of {self.detector}"
            )
        return int(np.ceil((gps - self.gps_start) * self.sample_rate - 1e-9))

    def global_index(self, i: int) -> int:
        """Absolute ``.bin`` sample index of segment-local sample ``i``."""
        return self.sample_start_idx + int(i)


@dataclass(frozen=True)
class HostSpan:
    """A run of window starts hosted entirely inside one segment."""

    segment: Segment
    first_local: int
    n_windows: int
    stride_samples: int

    @property
    def first_gps(self) -> float:
        """GPS of the first window start in the span."""
        return self.segment.gps_of_local(self.first_local)

    def starts_local(self) -> np.ndarray:
        """Segment-local sample indices of every window start."""
        return self.first_local + self.stride_samples * np.arange(
            self.n_windows, dtype=np.int64
        )

    def starts_gps(self) -> np.ndarray:
        """
        GPS times of every window start.

        Built from the integer sample indices, so the times are exact rather than
        accumulated from repeated additions of a float stride.
        """
        return self.segment.gps_start + self.starts_local() / self.segment.sample_rate


@dataclass(frozen=True)
class CoverageReport:
    """Decomposition of analysed time, emitted by :func:`window_hosts`."""

    union_s: float
    hosted_s: float
    n_windows: int
    lost_window_fit_s: float
    lost_boundary_holes_s: float
    lost_phase_restart_s: float
    n_holes: int

    def as_dict(self) -> dict:
        """Flat dict for manifest attrs."""
        return {
            "union_s": self.union_s,
            "hosted_s": self.hosted_s,
            "n_windows": self.n_windows,
            "lost_window_fit_s": self.lost_window_fit_s,
            "lost_boundary_holes_s": self.lost_boundary_holes_s,
            "lost_phase_restart_s": self.lost_phase_restart_s,
            "n_holes": self.n_holes,
        }


def load_segments(path: str | Path) -> List[Segment]:
    """
    Read a ``*_segments.json`` sidecar into :class:`Segment` records.

    Returned in file order, which is *not* time order. Call :func:`sort_by_gps` where
    time order is wanted; the file order is retained because ``sample_start_idx`` is
    assigned along it and the two must not be conflated.
    """
    import json

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    detectors = {r["detector"] for r in records}
    if len(detectors) > 1:
        raise ValueError(
            f"{path} names more than one detector ({sorted(detectors)}); a sidecar "
            "describes one detector's release"
        )
    runs = {r["observing_run"] for r in records}
    if len(runs) > 1:
        raise ValueError(f"{path} names more than one observing run ({sorted(runs)})")
    return [
        Segment(
            segment_index=int(r["segment_index"]),
            detector=r["detector"],
            observing_run=r["observing_run"],
            gps_start=float(r["gps_start"]),
            gps_end=float(r["gps_end"]),
            sample_rate=float(r["sample_rate"]),
            nsamples=int(r["nsamples"]),
            sample_start_idx=int(r["sample_start_idx"]),
            dyn_range_fac=float(r["dyn_range_fac"]),
            dataset=(str(r["dataset"]) if r.get("dataset") else None),
            noise_low_freq_cutoff=float(r["noise_low_freq_cutoff"]),
        )
        for r in records
    ]


def sort_by_gps(segments: Sequence[Segment]) -> List[Segment]:
    """Return ``segments`` ordered by ``gps_start`` (the sidecar is unordered)."""
    return sorted(segments, key=lambda s: (s.gps_start, s.gps_end))


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Merge overlapping/abutting intervals into a disjoint, sorted list."""
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: List[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(
    a: Sequence[Interval], b: Sequence[Interval], shift_b_s: float = 0.0
) -> List[Interval]:
    """Intersect two disjoint interval lists, optionally shifting ``b`` in time."""
    left = merge_intervals(a)
    right = merge_intervals((s + shift_b_s, e + shift_b_s) for s, e in b)
    out: List[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            out.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return out


def coincident_intervals(
    segments_by_detector: dict, shifts_s: dict | None = None
) -> List[Interval]:
    """
    GPS intervals where every detector has data, under optional per-detector shifts.

    Generic over the network size: two detectors and three are the same intersection
    taken over more lists.
    """
    if not segments_by_detector:
        raise ValueError("no detectors given")
    shifts_s = shifts_s or {}

    def as_intervals(value) -> List[Interval]:
        items = list(value)
        if items and isinstance(items[0], Segment):
            return merge_intervals((s.gps_start, s.gps_end) for s in items)
        return merge_intervals(items)

    detectors = list(segments_by_detector)
    accumulated = as_intervals(segments_by_detector[detectors[0]])
    accumulated = merge_intervals(
        (s + shifts_s.get(detectors[0], 0.0), e + shifts_s.get(detectors[0], 0.0))
        for s, e in accumulated
    )
    for detector in detectors[1:]:
        accumulated = intersect_intervals(
            accumulated,
            as_intervals(segments_by_detector[detector]),
            shift_b_s=shifts_s.get(detector, 0.0),
        )
    return accumulated


def hostable_intervals(
    segments: Sequence[Segment], window_samples: int
) -> List[Interval]:
    """
    Times at which a window start can sit entirely inside a single segment.

    Not the same as the times the detector has data. A window needs a whole window of
    contiguous samples from *one* chunk, and consecutive chunks overlap by less than a
    window, so each boundary leaves a band where data exists but no chunk can hold a
    whole window.

    This matters most for coincidence. Two detectors have their boundaries at unrelated
    times, so a window can be hostable in one and not the other; intersecting raw data
    presence would admit windows a follower detector cannot supply, which surfaces much
    later as a reader running off the end of a segment.
    """
    if not segments:
        return []
    rate = segments[0].sample_rate
    window_s = window_samples / rate
    return merge_intervals(
        (s.gps_start, s.gps_end - window_s)
        for s in segments
        if s.nsamples >= window_samples
    )


def subtract_intervals(
    a: Sequence[Interval], b: Sequence[Interval]
) -> List[Interval]:
    """Parts of ``a`` not covered by ``b``, both treated as interval lists."""
    left = merge_intervals(a)
    right = merge_intervals(b)
    out: List[Interval] = []
    for start, end in left:
        cursor = start
        for r0, r1 in right:
            if r1 <= cursor:
                continue
            if r0 >= end:
                break
            if r0 > cursor:
                out.append((cursor, min(r0, end)))
            cursor = max(cursor, r1)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return [(s, e) for s, e in out if e > s]


def window_hosts(
    segments: Sequence[Segment],
    window_samples: int,
    stride_samples: int,
    restrict_to: Sequence[Interval] | None = None,
    coverage: bool = True,
) -> Tuple[List[HostSpan], Optional[CoverageReport]]:
    """
    Assign window starts to owning segments so no GPS instant is analysed twice.

    Segments are swept in GPS order; each owns the part of its span not already
    covered, and hosts only windows that fit entirely inside it. The returned
    :class:`CoverageReport` separates time lost to window fit, to the per-boundary
    holes described in the module docstring, and to stride phase restarts.

    Parameters
    ----------
    coverage : bool
        Whether to decompose what the lattice did not reach. The decomposition walks
        every window start individually and costs far more than the spans do -- on the
        O3a lattice it is essentially the whole cost of this function -- so a caller
        that only needs the spans, such as a slide measuring its own livetime, asks for
        ``False`` and gets ``None`` in its place. The spans are identical either way.

    Notes
    -----
    Window starts sit on each segment's *own* sample lattice, at multiples of the stride
    from that segment's first sample. They cannot share one global lattice: segments
    begin at unrelated GPS times, so a global lattice would land between samples in all
    but one of them.

    A window is hosted only if it fits entirely inside its segment. Reading across a
    boundary would splice two chunks whose overlapping samples differ, which injects a
    discontinuity at exactly the scale the search is looking for.
    """
    if not segments:
        return [], (CoverageReport(0.0, 0.0, 0, 0.0, 0.0, 0.0, 0) if coverage else None)

    ordered = sort_by_gps(segments)
    rate = ordered[0].sample_rate
    stride_s = stride_samples / rate
    window_s = window_samples / rate

    union = merge_intervals((s.gps_start, s.gps_end) for s in ordered)
    restriction = None if restrict_to is None else merge_intervals(restrict_to)
    if restriction is not None:
        union = intersect_intervals(union, restriction)
    union_s = sum(e - s for s, e in union)

    spans: List[HostSpan] = []
    covered_until = -np.inf
    # The restriction is merged once and then walked with a cursor rather than
    # intersected per segment. Both lists are in GPS order, so a restriction interval
    # that ends before this segment begins can never be wanted again -- segments only
    # move forward. Intersecting per segment re-sorts the whole restriction each time,
    # which on the O3a lattice is 22,874 sorts of 37,000 intervals and by far the
    # dominant cost of building a lattice.
    cursor = 0

    for segment in ordered:
        if restriction is None:
            allowed: List[Interval] = [(segment.gps_start, segment.gps_end)]
        else:
            while (
                cursor < len(restriction)
                and restriction[cursor][1] <= segment.gps_start
            ):
                cursor += 1
            allowed = []
            index = cursor
            while index < len(restriction) and restriction[index][0] < segment.gps_end:
                lo = max(segment.gps_start, restriction[index][0])
                hi = min(segment.gps_end, restriction[index][1])
                if hi > lo:
                    allowed.append((lo, hi))
                index += 1
        # Nothing already assigned may be assigned again.
        allowed = [
            (max(lo, covered_until), hi)
            for lo, hi in allowed
            if hi > max(lo, covered_until)
        ]

        for lo, hi in allowed:
            # A start must lie at or after `lo`, its whole window must fit inside the
            # segment, and the stride of searchable time it claims must fit inside the
            # allowed interval -- otherwise a window at the very end would claim time
            # that was excluded, and the analysed total would exceed the region.
            upper_start = min(hi - stride_s, segment.gps_end - window_s)
            if upper_start < lo:
                continue
            needed = max(0.0, (lo - segment.gps_start) * rate)
            first_local = int(np.ceil(needed / stride_samples - 1e-9)) * stride_samples
            last_allowed = min(
                (upper_start - segment.gps_start) * rate,
                float(segment.nsamples - window_samples),
            )
            if last_allowed < first_local:
                continue
            last_local = (
                int(np.floor((last_allowed - first_local) / stride_samples + 1e-9))
                * stride_samples
                + first_local
            )
            n_windows = (last_local - first_local) // stride_samples + 1
            if n_windows <= 0:
                continue
            spans.append(
                HostSpan(
                    segment=segment,
                    first_local=int(first_local),
                    n_windows=int(n_windows),
                    stride_samples=int(stride_samples),
                )
            )
            covered_until = segment.gps_of_local(int(last_local)) + stride_s

    if not coverage:
        return spans, None
    return spans, coverage_report(spans, union, union_s, stride_s)


def coverage_report(
    spans: Sequence[HostSpan],
    union: Sequence[Interval],
    union_s: float,
    stride_s: float,
) -> CoverageReport:
    """
    Attribute every second of ``union`` that the lattice did not reach.

    Split out from :func:`window_hosts` because it is the expensive half and not every
    caller needs it: it walks each window start in turn, where the spans that produced
    them are counted in closed form.

    The decomposition is required to close exactly -- hosted plus the three losses must
    equal the union -- which is what makes it a check on the sweep rather than a summary
    of it. A start assigned outside the union, or two starts claiming the same second,
    shows up here as a failure to balance.
    """
    n_windows = sum(s.n_windows for s in spans)
    hosted_s = n_windows * stride_s

    # Attribute everything in the union that no window start claims. Each hosted start
    # claims one stride of searchable time; what is left over has three causes.
    claimed = merge_intervals(
        (float(g), float(g) + stride_s) for span in spans for g in span.starts_gps()
    )
    unclaimed = subtract_intervals(union, claimed)

    union_ends = {round(e, 9) for _, e in union}
    lost_window_fit_s = 0.0
    lost_boundary_holes_s = 0.0
    lost_phase_restart_s = 0.0
    n_holes = 0
    for lo, hi in unclaimed:
        length = hi - lo
        if round(hi, 9) in union_ends:
            # The tail of a contiguous stretch of data: a window needs a full window of
            # samples, so the last window_s of any stretch can host no start.
            lost_window_fit_s += length
        elif length > stride_s:
            # Structural: consecutive chunks overlap by less than one window, so a band
            # at each boundary can be reached by neither.
            lost_boundary_holes_s += length
            n_holes += 1
        else:
            # Sub-stride remainder from each segment restarting the stride phase.
            lost_phase_restart_s += length

    total = (
        hosted_s + lost_window_fit_s + lost_boundary_holes_s + lost_phase_restart_s
    )
    if abs(total - union_s) > 1e-6 * max(1.0, union_s):
        raise AssertionError(
            f"coverage decomposition does not close: {total} accounted against "
            f"{union_s} of union. This means a window start was assigned outside the "
            "union, or two starts claimed the same time."
        )
    return CoverageReport(
        union_s=union_s,
        hosted_s=hosted_s,
        n_windows=int(n_windows),
        lost_window_fit_s=lost_window_fit_s,
        lost_boundary_holes_s=lost_boundary_holes_s,
        lost_phase_restart_s=lost_phase_restart_s,
        n_holes=n_holes,
    )


def load_veto_segments(
    run: str, detector: str, category: int = 1, cache_dir: str | Path | None = None
) -> List[Interval]:
    """
    Fetch CAT-N veto intervals (cached on /work; never /tmp).

    Through :func:`pycbc.dq.query_flag`, which is where the awkward part already lives.
    GWOSC publishes a veto flag with the opposite sense to the LIGO convention -- its
    ``*_CAT n_VETO`` timeline marks the time that is *good* -- and PyCBC handles the
    inversion explicitly (``pycbc/dq.py:161``, "Special cases as the GWOSC convention is
    backwards from normal LIGO / Virgo operation"). Reimplementing that here would be one
    more place for the sense to be got backwards, and a veto applied with the wrong sense
    removes exactly the time it was meant to keep.

    Returned as intervals to subtract, not as the time to keep, so it composes with
    :func:`subtract_intervals` the way every other exclusion in this module does.

    Parameters
    ----------
    category : int
        Veto category. 1 is the mandatory pre-analysis removal; higher categories are
        trigger-level and are not a data cut.
    cache_dir : path, optional
        Where the fetched timeline is kept. Never under the system temporary directory:
        a campaign is re-run months later and a veto list that has silently vanished
        would be re-fetched from a service that may by then answer differently.
    """
    import os

    from pycbc.dq import query_flag

    from sage.search.dataprep import run_span

    span = run_span(run)
    flag = f"CBC_CAT{int(category)}_VETO"
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if str(cache_dir) == "/tmp" or "/tmp/" in f"{cache_dir}/":
            raise ValueError(
                f"cache_dir must not be under /tmp, got {cache_dir}; a veto list that "
                "vanishes between runs makes the livetime irreproducible"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    vetoed = query_flag(
        detector, flag, int(span[0]), int(span[1]), cache=cache_dir is not None
    )
    return merge_intervals((float(s.start), float(s.end)) for s in vetoed)


def verify_cat1_applied(
    segments: Sequence[Segment], vetoes: Sequence[Interval]
) -> dict:
    """
    Difference the release coverage against the official CAT1 list.

    CAT1 removal is mandatory before analysis; this proves whether the release
    already has it rather than assuming either way.

    The two states are told apart by *how much* vetoed time the release still covers, not
    by whether any is present: a release built from GWOSC's ``DATA`` flag already excludes
    category 1 by construction, so an overlap of a few seconds at a boundary is rounding
    and an overlap of hours is a release that never applied it.
    """
    covered = merge_intervals((s.gps_start, s.gps_end) for s in segments)
    vetoed = merge_intervals(vetoes)
    remaining = intersect_intervals(covered, vetoed)
    covered_s = sum(b - a for a, b in covered)
    vetoed_s = sum(b - a for a, b in vetoed)
    remaining_s = sum(b - a for a, b in remaining)
    return {
        "release_livetime_s": covered_s,
        "veto_livetime_s": vetoed_s,
        "vetoed_time_still_covered_s": remaining_s,
        "fraction_of_release": remaining_s / covered_s if covered_s else 0.0,
        "applied": remaining_s == 0.0,
        "intervals": remaining,
    }



def exact_livetime_s(
    spans: Sequence[HostSpan], stride_samples: int, sample_rate: float
) -> float:
    """
    Analysed time as ``n_windows * stride_s``, summed exactly over spans.

    The window count is summed as an integer and multiplied once, rather than
    accumulating a float stride per span. Over a hundred million windows the two differ.
    """
    total_windows = sum(int(span.n_windows) for span in spans)
    return total_windows * (stride_samples / sample_rate)


def run(spec, **kwargs) -> dict:
    """
    Stage driver: resolve the network's analysable time and record its decomposition.

    Reads each detector's sidecar, intersects the unions into coincident time, sweeps the
    window lattice for ownership, and writes the coverage decomposition into the campaign
    manifest under the observing run.

    The decomposition, not a total. Every rate the search quotes divides by one of these
    numbers, so *which* time was lost to what -- stride-phase restarts, ends of chains,
    genuine gaps -- is the difference between a defensible livetime and an assertion. The
    decomposition is required to close exactly, which is what makes it a check on the
    sweep rather than a summary of it.

    Vetoes are applied when the campaign asks for them and a cached veto list is present.
    ``apply_cat1`` with no cache is refused rather than skipped: proceeding would produce a
    livetime that silently describes unvetoed data while the provenance says otherwise.
    """
    from sage.search.manifest import RunManifest

    geometry = spec.geometry_object()
    run_name = spec.data.observing_run
    release = Path(spec.data.release_dir)
    segments = {
        detector: load_segments(
            release / f"data_{detector}_{run_name}_segments.json"
        )
        for detector in spec.data.detectors
    }

    vetoes: dict = {}
    if spec.data.apply_cat1:
        cache = spec.data.cat1_cache_dir
        if cache is None:
            raise ValueError(
                "data.apply_cat1 is set but no data.cat1_cache_dir is configured, so the "
                "veto intervals cannot be read and no network fetch is attempted here. "
                "Point cat1_cache_dir at a cached veto list, or set apply_cat1=False and "
                "state in the configuration that the release's own flag selection is "
                "being relied on -- what must not happen is a livetime that describes "
                "unvetoed data while the provenance says it was vetoed"
            )
        for detector in spec.data.detectors:
            vetoes[detector] = load_veto_segments(cache, detector, run_name)

    unions = {}
    for detector, records in segments.items():
        union = merge_intervals(
            [(s.gps_start, s.gps_end) for s in records]
        )
        if vetoes.get(detector):
            union = subtract_intervals(union, vetoes[detector])
        unions[detector] = union

    coincident = list(unions[spec.data.detectors[0]])
    for detector in spec.data.detectors[1:]:
        coincident = intersect_intervals(coincident, unions[detector])
    coincident_s = float(sum(hi - lo for lo, hi in coincident))

    # The lattice is carried by the reference detector, so ownership is swept over its
    # segments restricted to coincident time. A follower's own segmentation is resolved
    # per slide, where the lag decides which of its segments hosts each window.
    reference = spec.slides.reference_detector
    if reference not in segments:
        raise ValueError(
            f"slides.reference_detector {reference!r} is not in the network "
            f"{sorted(segments)}; the lattice would be carried by a detector the search "
            "does not read"
        )
    spans, report = window_hosts(
        segments[reference],
        geometry.window_samples,
        geometry.stride_samples,
        restrict_to=coincident,
        coverage=True,
    )

    coverage = {
        "detectors": list(spec.data.detectors),
        "coincident_livetime_s": coincident_s,
        "coincident_intervals": len(coincident),
        # The reference detector's own hostable count, which is an UPPER BOUND on what the
        # network analyses: a window starting inside coincident time still needs a whole
        # window of contiguous data in every follower, and a follower's segment boundaries
        # fall at moments unrelated to the reference's. AnalysisGrid resolves that per
        # detector and its count is the authoritative one -- on the real O3a release the
        # two differ by 11,632 windows, 0.09%. Reported side by side so the difference is
        # a number rather than a discrepancy noticed later.
        "reference_hostable_windows": int(sum(s.n_windows for s in spans)),
        "reference_hostable_livetime_s": float(
            sum(s.n_windows for s in spans) * geometry.stride_s
        ),
        "vetoed": bool(vetoes),
        "reference_detector": reference,
        **(report.as_dict() if report is not None else {}),
        **{
            f"union_livetime_s_{detector}": float(
                sum(hi - lo for lo, hi in union)
            )
            for detector, union in unions.items()
        },
    }
    manifest = RunManifest(path=Path(spec.path("manifest.h5")))
    manifest.record_livetime(run_name, coverage)
    return {
        **coverage,
        # Digest the whole coverage dict, not three of its entries. The loss
        # decomposition is this stage's product as much as the livetime is -- it is what
        # says whether a deficit is a genuine gap or a lattice that restarts its phase --
        # and a re-attribution between its terms leaves every summary scalar alone.
        "fingerprint": combine(
            coverage["reference_hostable_windows"],
            f"{coincident_s:.6f}",
            len(coincident),
            digest_values(
                {
                    **coverage,
                    "coincident_intervals_gps": np.asarray(
                        coincident, dtype=np.float64
                    ).reshape(-1, 2),
                }
            ),
        ),
    }

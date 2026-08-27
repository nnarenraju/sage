#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_segments.py
Description   : Interval algebra, ownership and the analysed-time decomposition.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Runs on synthetic sidecars; needs no data and no GPU.

The release these read is awkward in three specific ways, all measured from the real
sidecars: chunks are numbered in an order anti-correlated with time, consecutive chunks
overlap by 15.5994 s, and the overlapping samples differ between the two chunks holding
them. Every test here is shaped by one of those.
"""

import json

import numpy as np
import pytest

from sage.search.segments import (
    Segment,
    coincident_intervals,
    exact_livetime_s,
    intersect_intervals,
    load_segments,
    merge_intervals,
    sort_by_gps,
    window_hosts,
)

SAMPLE_RATE = 2048.0
WINDOW_SAMPLES = 32768          # 16 s
STRIDE_SAMPLES = 205            # 0.100097656250 s
OVERLAP_S = 15.5994             # measured on the real O3a and O3b sidecars
CHUNK_S = 512.0


def _chain(n_chunks=4, gps0=1238166018.0, chunk_s=CHUNK_S, overlap_s=OVERLAP_S,
           reverse_index=True):
    """Segments overlapping like the real release, numbered against time order."""
    step = chunk_s - overlap_s
    spans = [(gps0 + k * step, gps0 + k * step + chunk_s) for k in range(n_chunks)]
    order = list(range(n_chunks))[::-1] if reverse_index else list(range(n_chunks))
    nsamples = int(round(chunk_s * SAMPLE_RATE))
    out = []
    for position, chunk in enumerate(order):
        out.append(
            Segment(
                segment_index=position,
                detector="H1",
                observing_run="O3a",
                gps_start=spans[chunk][0],
                gps_end=spans[chunk][1],
                sample_rate=SAMPLE_RATE,
                nsamples=nsamples,
                sample_start_idx=position * nsamples,
                dyn_range_fac=5.902958103587057e20,
                noise_low_freq_cutoff=15.0,
            )
        )
    return out


class TestIntervalAlgebra:
    """Merging and intersection against cases with a known answer."""

    def test_merge_disjoint_is_identity(self):
        """Disjoint intervals are returned unchanged and sorted."""
        given = [(30.0, 40.0), (10.0, 20.0)]
        assert merge_intervals(given) == [(10.0, 20.0), (30.0, 40.0)]

    def test_merge_overlapping(self):
        """Overlapping and exactly touching intervals combine into one."""
        assert merge_intervals([(0.0, 10.0), (5.0, 15.0)]) == [(0.0, 15.0)]
        assert merge_intervals([(0.0, 10.0), (10.0, 20.0)]) == [(0.0, 20.0)]
        assert merge_intervals([(0.0, 10.0), (10.5, 20.0)]) == [
            (0.0, 10.0),
            (10.5, 20.0),
        ]

    def test_merge_empty(self):
        assert merge_intervals([]) == []

    def test_intersect_nested_and_empty(self):
        """Nested, partial and disjoint intersections."""
        assert intersect_intervals([(0.0, 100.0)], [(10.0, 20.0)]) == [(10.0, 20.0)]
        assert intersect_intervals([(0.0, 15.0)], [(10.0, 20.0)]) == [(10.0, 15.0)]
        assert intersect_intervals([(0.0, 5.0)], [(10.0, 20.0)]) == []
        assert intersect_intervals([], [(10.0, 20.0)]) == []

    def test_intersect_with_shift(self):
        """Shifting one side moves the intersection by the same amount."""
        a = [(0.0, 100.0)]
        b = [(10.0, 20.0)]
        assert intersect_intervals(a, b, shift_b_s=5.0) == [(15.0, 25.0)]
        assert intersect_intervals(a, b, shift_b_s=-5.0) == [(5.0, 15.0)]

    def test_coincident_three_detectors(self):
        """Coincidence generalises past two detectors."""
        got = coincident_intervals(
            {
                "H1": [(0.0, 100.0)],
                "L1": [(10.0, 90.0)],
                "V1": [(20.0, 80.0)],
            }
        )
        assert got == [(20.0, 80.0)]

    def test_coincident_applies_shifts(self):
        """A time slide shifts one detector's segments before intersecting."""
        got = coincident_intervals(
            {"H1": [(0.0, 100.0)], "L1": [(0.0, 100.0)]}, shifts_s={"L1": 50.0}
        )
        assert got == [(50.0, 100.0)]


class TestOwnership:
    """Each instant is analysed once, and losses are attributable."""

    def test_no_start_hosted_twice(self):
        """Overlapping chunks never both host the same start time."""
        spans, _ = window_hosts(_chain(), WINDOW_SAMPLES, STRIDE_SAMPLES)
        starts = np.concatenate([s.starts_gps() for s in spans])
        assert starts.size == np.unique(starts).size
        assert np.all(np.diff(np.sort(starts)) > 0)

    def test_window_within_one_chunk(self):
        """Every hosted window fits entirely inside its owning chunk."""
        window_s = WINDOW_SAMPLES / SAMPLE_RATE
        spans, _ = window_hosts(_chain(), WINDOW_SAMPLES, STRIDE_SAMPLES)
        for span in spans:
            starts = span.starts_gps()
            assert starts.min() >= span.segment.gps_start
            assert starts.max() + window_s <= span.segment.gps_end + 1e-9

    def test_hole_is_window_minus_overlap(self):
        """
        The gap at a boundary is the window length minus the chunk overlap.

        The stored release overlaps consecutive chunks by slightly less than one window,
        so each boundary leaves a narrow band hosting no start. The size follows from
        the geometry and is asserted rather than measured after the fact.
        """
        window_s = WINDOW_SAMPLES / SAMPLE_RATE
        stride_s = STRIDE_SAMPLES / SAMPLE_RATE
        segments = _chain(n_chunks=4)
        _, report = window_hosts(segments, WINDOW_SAMPLES, STRIDE_SAMPLES)

        expected_per_hole = window_s - OVERLAP_S
        assert report.n_holes == 3
        mean_hole = report.lost_boundary_holes_s / report.n_holes
        assert mean_hole == pytest.approx(expected_per_hole, abs=stride_s)

    def test_wider_overlap_closes_hole(self):
        """
        Overlapping by more than a window leaves no unreachable band.

        This is the fix the search-grade release applies: request an overlap of one
        window plus both trims instead of one window minus them.
        """
        window_s = WINDOW_SAMPLES / SAMPLE_RATE
        segments = _chain(n_chunks=4, overlap_s=window_s + 0.5)
        _, report = window_hosts(segments, WINDOW_SAMPLES, STRIDE_SAMPLES)
        assert report.lost_boundary_holes_s == pytest.approx(0.0, abs=1e-9)
        assert report.n_holes == 0

    def test_coverage_decomposition_closes(self):
        """Union coverage equals hosted time plus every itemised loss."""
        _, report = window_hosts(_chain(), WINDOW_SAMPLES, STRIDE_SAMPLES)
        total = (
            report.hosted_s
            + report.lost_window_fit_s
            + report.lost_boundary_holes_s
            + report.lost_phase_restart_s
        )
        assert total == pytest.approx(report.union_s, abs=1e-6)

    def test_loss_terms_non_negative(self):
        """A decomposition that closes by letting a term go negative is not one."""
        _, report = window_hosts(_chain(), WINDOW_SAMPLES, STRIDE_SAMPLES)
        assert report.lost_window_fit_s >= 0.0
        assert report.lost_boundary_holes_s >= 0.0
        assert report.lost_phase_restart_s >= 0.0
        assert report.hosted_s >= 0.0

    def test_livetime_is_windows_times_stride(self):
        """Analysed time is a count times the stride, with no accumulated error."""
        spans, report = window_hosts(_chain(), WINDOW_SAMPLES, STRIDE_SAMPLES)
        stride_s = STRIDE_SAMPLES / SAMPLE_RATE
        livetime = exact_livetime_s(spans, STRIDE_SAMPLES, SAMPLE_RATE)
        assert livetime == report.n_windows * stride_s
        assert livetime == report.hosted_s

    def test_short_segment_hosts_nothing(self):
        """A chunk too short for one window contributes no starts, and says so."""
        short = _chain(n_chunks=1, chunk_s=8.0)
        spans, report = window_hosts(short, WINDOW_SAMPLES, STRIDE_SAMPLES)
        assert spans == []
        assert report.n_windows == 0
        assert report.lost_window_fit_s == pytest.approx(8.0, abs=1e-9)

    def test_restriction_limits_hosting(self):
        """Restricting to an interval list keeps only starts inside it."""
        segments = _chain(n_chunks=2)
        lo = segments[-1].gps_start + 100.0
        hi = lo + 50.0
        spans, _ = window_hosts(
            segments, WINDOW_SAMPLES, STRIDE_SAMPLES, restrict_to=[(lo, hi)]
        )
        starts = np.concatenate([s.starts_gps() for s in spans])
        assert starts.min() >= lo
        assert starts.max() <= hi

    def test_gap_not_hosted(self):
        """Genuine missing time hosts nothing and is reported as lost."""
        a = _chain(n_chunks=1)[0]
        b = Segment(
            segment_index=1, detector="H1", observing_run="O3a",
            gps_start=a.gps_end + 1000.0, gps_end=a.gps_end + 1000.0 + CHUNK_S,
            sample_rate=SAMPLE_RATE, nsamples=int(CHUNK_S * SAMPLE_RATE),
            sample_start_idx=int(CHUNK_S * SAMPLE_RATE),
            dyn_range_fac=1.0, noise_low_freq_cutoff=15.0,
        )
        spans, report = window_hosts([a, b], WINDOW_SAMPLES, STRIDE_SAMPLES)
        starts = np.concatenate([s.starts_gps() for s in spans])
        assert not np.any((starts > a.gps_end) & (starts < b.gps_start))
        # The union excludes the gap, so it is not charged as a loss.
        assert report.union_s == pytest.approx(2 * CHUNK_S, abs=1e-6)

    def test_no_coverage_same_lattice(self):
        """
        ``coverage=False`` changes what is reported, never which windows are hosted.

        The decomposition walks every window start and is by far the more expensive half
        -- on the O3a lattice, 374 s and 11.8 GB against 0.06 s for the algebra -- so a
        slide ladder measuring its own livetime declines it. That is only safe if the
        spans are bit-for-bit what the full call produces, because the livetime the
        ladder reports is read straight off them.
        """
        segments = _chain(n_chunks=3)
        restrict = [(segments[0].gps_start + 60.0, segments[-1].gps_end - 60.0)]
        for restriction in (None, restrict):
            full, report = window_hosts(
                segments, WINDOW_SAMPLES, STRIDE_SAMPLES, restrict_to=restriction
            )
            fast, none = window_hosts(
                segments,
                WINDOW_SAMPLES,
                STRIDE_SAMPLES,
                restrict_to=restriction,
                coverage=False,
            )
            assert none is None
            assert report is not None
            assert len(fast) == len(full)
            for quick, whole in zip(fast, full):
                assert quick.segment.segment_index == whole.segment.segment_index
                assert quick.first_local == whole.first_local
                assert quick.n_windows == whole.n_windows
                assert quick.stride_samples == whole.stride_samples
            assert sum(s.n_windows for s in fast) == report.n_windows

    def test_no_coverage_empty_input(self):
        """The empty case answers in the same shape as the populated one."""
        spans, report = window_hosts([], WINDOW_SAMPLES, STRIDE_SAMPLES)
        assert spans == [] and report.n_windows == 0
        spans, none = window_hosts([], WINDOW_SAMPLES, STRIDE_SAMPLES, coverage=False)
        assert spans == [] and none is None

    @pytest.mark.parametrize("overlap_s", [0.0, OVERLAP_S, 510.0])
    @pytest.mark.parametrize("fragments", [1, 7, 40])
    def test_cursor_matches_per_segment(
        self, overlap_s, fragments
    ):
        """
        Walking the restriction with a cursor gives what per-segment intersection gave.

        The restriction is merged once and then walked forward, because intersecting it
        against each segment separately re-sorts the whole list every time -- at run
        scale, 22,874 sorts of 37,000 intervals, and by far the dominant cost of building
        a lattice. That rewrite is only sound if the cursor never passes an interval a
        later segment still needs, so it is checked here against a direct transcription
        of the algorithm it replaced.

        Parametrised over the overlap because that is what makes the cursor's monotonicity
        non-obvious: segments are ordered by start, and at the 510 s overlap measured on
        the real release consecutive chunks cover nearly the same span, so their order by
        start says little about their order by end. Parametrised over the number of
        restriction fragments because a single interval spanning everything exercises no
        cursor movement at all.
        """
        segments = _chain(n_chunks=6, overlap_s=overlap_s)
        lo, hi = segments[0].gps_start, segments[-1].gps_end
        step = (hi - lo) / fragments
        restriction = [(lo + k * step, lo + (k + 0.7) * step) for k in range(fragments)]

        def reference(segs):
            """The pre-cursor sweep: intersect the whole restriction, per segment."""
            out = []
            covered_until = -np.inf
            stride_s = STRIDE_SAMPLES / SAMPLE_RATE
            window_s = WINDOW_SAMPLES / SAMPLE_RATE
            for segment in sort_by_gps(segs):
                allowed = intersect_intervals(
                    [(segment.gps_start, segment.gps_end)], restriction
                )
                allowed = [
                    (max(a, covered_until), b)
                    for a, b in allowed
                    if b > max(a, covered_until)
                ]
                for a, b in allowed:
                    upper = min(b - stride_s, segment.gps_end - window_s)
                    if upper < a:
                        continue
                    needed = max(0.0, (a - segment.gps_start) * SAMPLE_RATE)
                    first = int(np.ceil(needed / STRIDE_SAMPLES - 1e-9)) * STRIDE_SAMPLES
                    last_allowed = min(
                        (upper - segment.gps_start) * SAMPLE_RATE,
                        float(segment.nsamples - WINDOW_SAMPLES),
                    )
                    if last_allowed < first:
                        continue
                    last = (
                        int(np.floor((last_allowed - first) / STRIDE_SAMPLES + 1e-9))
                        * STRIDE_SAMPLES
                        + first
                    )
                    n = (last - first) // STRIDE_SAMPLES + 1
                    if n <= 0:
                        continue
                    out.append((segment.segment_index, int(first), int(n)))
                    covered_until = segment.gps_of_local(int(last)) + stride_s
            return out

        walked, _ = window_hosts(
            segments, WINDOW_SAMPLES, STRIDE_SAMPLES, restrict_to=restriction
        )
        assert [
            (s.segment.segment_index, s.first_local, s.n_windows) for s in walked
        ] == reference(segments)


class TestSidecarIngest:
    """Reading the stored layout."""

    def test_index_layout_is_contiguous(self, synthetic_release):
        """Sample indices run continuously across chunks."""
        segments = load_segments(synthetic_release / "data_H1_O3a_segments.json")
        for a, b in zip(segments, segments[1:]):
            assert a.sample_start_idx + a.nsamples == b.sample_start_idx

    def test_gps_order_not_assumed(self, synthetic_release):
        """Chunks are sorted by time on load; file order carries no meaning."""
        segments = load_segments(synthetic_release / "data_H1_O3a_segments.json")
        # The fixture numbers records against time order, as the real release does.
        assert [s.gps_start for s in segments] != sorted(s.gps_start for s in segments)
        ordered = sort_by_gps(segments)
        assert [s.gps_start for s in ordered] == sorted(s.gps_start for s in ordered)
        # Sorting must not disturb the index bookkeeping.
        assert {s.sample_start_idx for s in ordered} == {
            s.sample_start_idx for s in segments
        }

    def test_gps_maps_within_chunk(self):
        """Converting a time to an index requires naming the chunk."""
        segments = sort_by_gps(_chain())
        first, second = segments[0], segments[1]
        # A time inside the overlap exists in both chunks, at different local indices.
        shared = second.gps_start + 1.0
        i0 = first.local_of_gps(shared)
        i1 = second.local_of_gps(shared)
        assert i0 != i1
        assert first.gps_of_local(i0) == pytest.approx(shared, abs=1.0 / SAMPLE_RATE)
        assert second.gps_of_local(i1) == pytest.approx(shared, abs=1.0 / SAMPLE_RATE)
        # And the two map to different absolute positions in the file.
        assert first.global_index(i0) != second.global_index(i1)

    def test_time_outside_chunk_refused(self):
        """Asking for an index outside the chunk is an error, not a clamp."""
        segment = sort_by_gps(_chain())[0]
        with pytest.raises(ValueError):
            segment.local_of_gps(segment.gps_end + 10.0)
        with pytest.raises(ValueError):
            segment.local_of_gps(segment.gps_start - 10.0)

    def test_duration_matches_samples(self):
        for segment in _chain():
            assert segment.duration_s == pytest.approx(
                segment.nsamples / segment.sample_rate, abs=1e-9
            )

    def test_mixed_detector_sidecar_refused(self, tmp_path):
        """A sidecar naming two detectors is a corrupted release, not a merge."""
        records = [
            {
                "segment_index": 0, "detector": "H1", "observing_run": "O3a",
                "gps_start": 0.0, "gps_end": 512.0, "sample_rate": SAMPLE_RATE,
                "nsamples": 1048576, "sample_start_idx": 0,
                "dyn_range_fac": 1.0, "noise_low_freq_cutoff": 15.0,
            },
            {
                "segment_index": 1, "detector": "L1", "observing_run": "O3a",
                "gps_start": 512.0, "gps_end": 1024.0, "sample_rate": SAMPLE_RATE,
                "nsamples": 1048576, "sample_start_idx": 1048576,
                "dyn_range_fac": 1.0, "noise_low_freq_cutoff": 15.0,
            },
        ]
        path = tmp_path / "mixed_segments.json"
        path.write_text(json.dumps(records))
        with pytest.raises(ValueError, match="detector"):
            load_segments(path)


class TestVetoVerification:
    """Whether a release already had category 1 removed, measured rather than assumed."""

    def _release(self, spans):
        from sage.search.segments import Segment

        return [
            Segment(
                segment_index=index,
                detector="H1",
                observing_run="O3a",
                gps_start=float(start),
                gps_end=float(end),
                sample_rate=2048.0,
                nsamples=int((end - start) * 2048),
                sample_start_idx=0,
                dyn_range_fac=5.9e20,
                noise_low_freq_cutoff=15.0,
            )
            for index, (start, end) in enumerate(spans)
        ]

    def test_disjoint_veto_reads_as_applied(self):
        """
        A release built on GWOSC's ``DATA`` flag already excludes category 1, so the two
        sets do not meet and the check has to say so rather than reporting an absence of
        evidence.
        """
        from sage.search.segments import verify_cat1_applied

        report = verify_cat1_applied(self._release([(0.0, 1000.0)]), [(2000.0, 3000.0)])

        assert report["applied"]
        assert report["vetoed_time_still_covered_s"] == 0.0

    def test_overlap_is_reported_as_time_not_a_flag(self):
        """
        How much vetoed time a release still covers is what separates rounding at a
        boundary from a release that never applied the veto. A boolean alone cannot.
        """
        from sage.search.segments import verify_cat1_applied

        report = verify_cat1_applied(self._release([(0.0, 1000.0)]), [(100.0, 200.0)])

        assert not report["applied"]
        assert report["vetoed_time_still_covered_s"] == pytest.approx(100.0)
        assert report["fraction_of_release"] == pytest.approx(0.1)
        assert report["intervals"] == [(100.0, 200.0)]

    def test_counts_only_the_overlap(self):
        """
        A veto reaching outside the release is not time the release covers, so it must
        not inflate the overlap -- which would report an applied veto as unapplied.
        """
        from sage.search.segments import verify_cat1_applied

        report = verify_cat1_applied(
            self._release([(0.0, 100.0)]), [(50.0, 5000.0)]
        )

        assert report["veto_livetime_s"] == pytest.approx(4950.0)
        assert report["vetoed_time_still_covered_s"] == pytest.approx(50.0)

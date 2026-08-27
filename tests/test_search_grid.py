#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_grid.py
Description   : The window lattice, block partitioning and cross-detector alignment.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One detector carries the lattice and the others follow it to the nearest sample, because
segment start times are multiples of half a sample and the two grids coincide only about
half the time. The residual is bounded by half a sample and is asserted here rather than
assumed, since it is a timing systematic and the network resolves arrival-time
differences.

Runs on synthetic segments; needs no data and no GPU.
"""

import numpy as np
import pytest

from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid
from sage.search.segments import Segment, coincident_intervals

RATE = 2048.0
CHUNK_S = 512.0
OVERLAP_S = 15.5994

GEOMETRY = SearchGeometry(
    sample_rate=RATE,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=205,
    tc_lower_s=5.0,
    tc_upper_s=7.0,
)


def _segments(detector, gps0, n_chunks=3, sub_sample_offset=0.0):
    """Overlapping chunks, optionally shifted off the reference sample grid."""
    step = CHUNK_S - OVERLAP_S
    nsamples = int(round(CHUNK_S * RATE))
    out = []
    for k in range(n_chunks):
        start = gps0 + k * step + sub_sample_offset
        out.append(
            Segment(
                segment_index=k, detector=detector, observing_run="O3a",
                gps_start=start, gps_end=start + CHUNK_S, sample_rate=RATE,
                nsamples=nsamples, sample_start_idx=k * nsamples,
                dyn_range_fac=1.0, noise_low_freq_cutoff=15.0,
            )
        )
    return out


def _network(sub_sample_offset=0.0, detectors=("H1", "L1")):
    gps0 = 1238166018.0
    segs = {detectors[0]: _segments(detectors[0], gps0)}
    for extra in detectors[1:]:
        segs[extra] = _segments(extra, gps0, sub_sample_offset=sub_sample_offset)
    return segs


def _grid(sub_sample_offset=0.0, detectors=("H1", "L1"), offsets_s=None):
    segs = _network(sub_sample_offset, detectors)
    coincident = coincident_intervals(segs, shifts_s=offsets_s)
    return AnalysisGrid.build(
        GEOMETRY, segs, coincident, offsets_s=offsets_s, reference_detector=detectors[0]
    )


class TestConstruction:
    """The lattice is built on one detector and covers coincident time."""

    def test_default_reference_is_first(self):
        assert _grid().reference_detector == "H1"

    def test_unknown_reference_is_refused(self):
        segs = _network()
        with pytest.raises(ValueError, match="reference"):
            AnalysisGrid.build(
                GEOMETRY, segs, coincident_intervals(segs), reference_detector="V1"
            )

    def test_reference_offset_refused(self):
        """Slides are measured relative to the reference, so its own lag must be zero."""
        segs = _network()
        with pytest.raises(ValueError, match="reference"):
            AnalysisGrid.build(
                GEOMETRY, segs, coincident_intervals(segs), offsets_s={"H1": 5.0}
            )

    def test_length_and_livetime_agree(self):
        grid = _grid()
        assert grid.livetime_s == len(grid) * GEOMETRY.stride_s

    def test_coverage_report_is_kept(self):
        """The decomposition that produced the lattice travels with it."""
        grid = _grid()
        assert grid.coverage is not None
        assert grid.coverage.n_windows == len(grid)

    def test_three_detector_network(self):
        grid = _grid(detectors=("H1", "L1", "V1"))
        assert grid.detectors == ("H1", "L1", "V1")
        assert len(grid) > 0


class TestBlocks:
    """Work partitioning never splits a run."""

    def test_blocks_cover_every_span_exactly_once(self):
        grid = _grid()
        blocks = grid.blocks(60.0)
        covered = []
        for block in blocks:
            covered.extend(range(*block.span_slice))
        assert covered == list(range(len(grid.reference_spans)))

    def test_block_counts_sum_to_lattice(self):
        grid = _grid()
        total = sum(
            n for block in grid.blocks(60.0) for _, _, n in grid.iter_block(block)
        )
        assert total == len(grid)

    def test_block_gps_ordered(self):
        grid = _grid()
        for block in grid.blocks(60.0):
            times = grid.gps(block)
            assert np.all(np.diff(times) > 0)
            assert times.size == sum(n for _, _, n in grid.iter_block(block))

    def test_non_positive_block_length_is_refused(self):
        with pytest.raises(ValueError):
            _grid().blocks(0.0)

    def test_single_huge_block(self):
        grid = _grid()
        blocks = grid.blocks(1e9)
        assert len(blocks) == 1
        assert blocks[0].span_slice == (0, len(grid.reference_spans))


class TestCrossDetectorAlignment:
    """The part that has to be right for a coincident search."""

    def test_aligned_detectors_have_zero_residual(self):
        grid = _grid(sub_sample_offset=0.0)
        for block in grid.blocks(120.0):
            assert grid.alignment_residuals(block)["L1"] == pytest.approx(0.0, abs=1e-9)

    def test_half_sample_residual(self):
        """
        The measured worst case on the real release: exactly half a sample.

        Segment starts are multiples of half a sample, so two detectors are either
        aligned or offset by this much, and nothing in between.
        """
        grid = _grid(sub_sample_offset=0.5 / RATE)
        for block in grid.blocks(120.0):
            assert grid.alignment_residuals(block)["L1"] == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.parametrize("offset", [0.0, 0.25 / RATE, 0.5 / RATE])
    def test_residual_never_exceeds_half_a_sample(self, offset):
        """
        Rounding to the nearest sample bounds the error for any real offset.

        Real offsets are 0 or exactly half a sample: the strain was conditioned at twice
        the analysis rate, so a segment start is a multiple of half a sample and nothing
        between occurs.
        """
        grid = _grid(sub_sample_offset=offset)
        for block in grid.blocks(120.0):
            for detector, residual in grid.alignment_residuals(block).items():
                assert residual <= 0.5 + 1e-9, detector

    @pytest.mark.parametrize("offset", [0.0, 0.25 / RATE, 0.5 / RATE, 0.9 / RATE])
    def test_all_detectors_supply_window(self, offset):
        """
        Whatever the grid offset, no window is hosted that a follower cannot supply.

        Found on the real O3a release rather than in a fixture: chunk boundaries fall at
        unrelated times in each detector, so a window can sit inside a single H1 chunk
        while straddling an L1 boundary. Intersecting raw data presence admits those; the
        lattice restricts on *hostable* time instead, so the resolver never runs off the
        end of a segment.
        """
        grid = _grid(sub_sample_offset=offset)
        for block in grid.blocks(120.0):
            expected = sum(n for _, _, n in grid.iter_block(block))
            for detector in grid.detectors:
                supplied = sum(
                    r.n_windows for r in grid.iter_block_detector(block, detector)
                )
                assert supplied == expected, detector

    def test_hostable_restriction_costs_little(self):
        """
        Excluding each detector's boundary bands removes a small, bounded fraction.

        Measured on O3a it is 0.31 per cent of coincident time. A large loss here would
        mean the chunk geometry, not the coincidence, is the problem.
        """
        segs = _network(sub_sample_offset=0.5 / RATE)
        coincident = coincident_intervals(segs)
        grid = AnalysisGrid.build(GEOMETRY, segs, coincident)
        coincident_s = sum(e - s for s, e in coincident)
        assert 0.90 < grid.livetime_s / coincident_s <= 1.0

    def test_detector_counts_agree(self):
        """
        A coincident window exists in every detector or in none.

        A count mismatch means one detector silently dropped windows the others kept,
        which would misalign the whole batch.
        """
        grid = _grid(sub_sample_offset=0.5 / RATE, detectors=("H1", "L1", "V1"))
        for block in grid.blocks(90.0):
            counts = {
                det: sum(r.n_windows for r in grid.iter_block_detector(block, det))
                for det in grid.detectors
            }
            assert len(set(counts.values())) == 1, counts

    def test_runs_never_cross_a_segment_boundary(self):
        """Every run fits inside one segment of its own detector."""
        grid = _grid(sub_sample_offset=0.5 / RATE)
        for block in grid.blocks(90.0):
            for detector in grid.detectors:
                for run in grid.iter_block_detector(block, detector):
                    last = run.first_local + (run.n_windows - 1) * run.stride_samples
                    assert run.first_local >= 0
                    assert last + GEOMETRY.window_samples <= run.segment.nsamples

    def test_slide_offset_shifts_the_followers(self):
        """A slid detector reads later data for the same reference window."""
        offsets = {"L1": 100.0}
        grid = _grid(offsets_s=offsets)
        block = grid.blocks(1e9)[0]
        reference = next(iter(grid.iter_block_detector(block, "H1")))
        follower = next(iter(grid.iter_block_detector(block, "L1")))
        t_ref = reference.segment.gps_of_local(reference.first_local)
        t_fol = follower.segment.gps_of_local(follower.first_local)
        assert t_fol - t_ref == pytest.approx(100.0, abs=1.0 / RATE)

    def test_unknown_detector_is_refused(self):
        grid = _grid()
        block = grid.blocks(1e9)[0]
        with pytest.raises(ValueError, match="unknown detector"):
            list(grid.iter_block_detector(block, "K1"))

    def test_whole_lattice_runs_match_blockwise(self):
        """
        ``runs_for_detector`` yields every block's runs, for followers too.

        The follower path is the one that matters: only the reference detector has stored
        spans, so a caller needing a follower's analysed windows has nothing to read and
        must derive them here.
        """
        grid = _grid(sub_sample_offset=0.5 / RATE, detectors=("H1", "L1", "V1"))
        for detector in grid.detectors:
            whole = list(grid.runs_for_detector(detector))
            blockwise = [
                run
                for block in grid.blocks(90.0)
                for run in grid.iter_block_detector(block, detector)
            ]
            assert sum(r.n_windows for r in whole) == len(grid)
            assert [
                (r.segment.segment_index, r.first_local, r.n_windows) for r in whole
            ] == [
                (r.segment.segment_index, r.first_local, r.n_windows) for r in blockwise
            ]


class TestBuildCallSites:
    """
    Every lattice in the package is built the same way.

    Not a behavioural test: the call sites live in stage drivers that need a checkpoint
    and a GPU, so the property is checked where it is actually decided. Both halves were
    violated -- the injection campaign and the trials stage each took the reference
    detector from whichever detector happened to be first in the segment dict, and four
    sites paid for a coverage decomposition nobody reads.
    """

    @staticmethod
    def _calls():
        """Every ``AnalysisGrid.build`` call in the package, as an AST node."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "sage" / "search"
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "build"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "AnalysisGrid"
                ):
                    yield path, node

    def test_reference_detector_is_always_given(self):
        """
        Left to the default it is whichever detector the segment dict lists first.

        The lattice is defined in the reference detector's frame, so two stages that
        disagree about it describe different windows while reporting the same livetime and
        the same count.
        """
        offenders = [
            f"{path.name}:{node.lineno}"
            for path, node in self._calls()
            if "reference_detector" not in {kw.arg for kw in node.keywords}
        ]
        assert not offenders, offenders

    def test_coverage_is_requested_only_where_it_is_read(self):
        """Only the ``grid`` stage reports the decomposition; it costs 374 s on O3a."""
        offenders = [
            f"{path.name}:{node.lineno}"
            for path, node in self._calls()
            if path.name != "grid.py"
            and "coverage" not in {kw.arg for kw in node.keywords}
        ]
        assert not offenders, offenders

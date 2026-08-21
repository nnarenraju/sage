#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_reader.py
Description   : Window unfolding and segment-bounded reads over the memmap release.

Created on 2026-08-13

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Runs on the synthetic release; needs no data and no GPU.

The property that matters is that a read never crosses a chunk boundary, and a contiguous
release satisfies a weak version of that by accident. The fixture is filled so that every
sample of a chunk carries that chunk's own constant, which turns the invariant into a
single assertion: a window that stayed inside one chunk has zero peak-to-peak, and a
spliced one cannot. :class:`TestSegmentBoundaryInvariant` also exercises the oracle
against a deliberately spliced read, since an oracle that never fires proves nothing.

The streaming reader itself is layer 4 -- it owns the GPU pipeline and the prefetch
thread -- and is left unimplemented, with its expected behaviour recorded as strict xfail.
"""

import dataclasses

import numpy as np
import pytest

from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid
from sage.search.reader import (
    StreamingStrainReader,
    WindowBatch,
    read_segment_span,
    unfold_windows,
)
from sage.search.segments import (
    Segment,
    coincident_intervals,
    load_segments,
    window_hosts,
)

RATE = 2048.0
WINDOW_SAMPLES = 32768          # 16 s
STRIDE_SAMPLES = 205            # 0.100097656250 s
DYN_RANGE_FAC = 5.902958103587057e20

GEOMETRY = SearchGeometry(
    sample_rate=RATE,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=STRIDE_SAMPLES,
    tc_lower_s=5.0,
    tc_upper_s=7.0,
)


@pytest.fixture(scope="module")
def small_release(tmp_path_factory):
    """
    A constant-filled release with the real chunk geometry and shorter chunks.

    Built once for the module, since every test here only reads it. Chunks are 64 s rather
    than 512 s, which changes how many windows a chunk hosts and nothing else: the overlap
    is still 15.5994 s, slightly less than one 16 s window, so the boundary bands and the
    ownership sweep behave exactly as they do on the real release.
    """
    from tests.search_fixtures import make_synthetic_release

    return make_synthetic_release(
        tmp_path_factory.mktemp("reader_release"), detectors=("H1", "L1"), chunk_s=64.0
    )


def _open(release, detector="H1", run="O3a"):
    """Segment records and the flat little-endian float32 stream they index into."""
    segments = load_segments(release / f"data_{detector}_{run}_segments.json")
    mmap = np.memmap(release / f"data_{detector}_{run}.bin", dtype="<f4", mode="r")
    return segments, mmap


def _segment(nsamples=100, sample_start_idx=1000, dyn_range_fac=2.0, index=3):
    """One record, positioned away from the start of the file on purpose."""
    return Segment(
        segment_index=index,
        detector="H1",
        observing_run="O3a",
        gps_start=1238166018.0,
        gps_end=1238166018.0 + nsamples / RATE,
        sample_rate=RATE,
        nsamples=nsamples,
        sample_start_idx=sample_start_idx,
        dyn_range_fac=dyn_range_fac,
        noise_low_freq_cutoff=15.0,
    )


def _file(n=2000):
    """A stand-in release stream whose value encodes its own absolute index."""
    return np.arange(n, dtype=np.float32)


class TestUnfoldCounts:
    """The window count and the position of the last window."""

    @pytest.mark.parametrize(
        "n,window,stride",
        [
            (50, 10, 1),
            (50, 10, 3),
            (50, 10, 10),
            (50, 10, 17),
            (50, 50, 1),
            (131072, WINDOW_SAMPLES, STRIDE_SAMPLES),
        ],
    )
    def test_window_count_closed_form(self, n, window, stride):
        """
        The count is exactly ``1 + (n - window) // stride``.

        The lattice size is bookkeeping the whole search is indexed by: livetime is a
        window count times the stride, and a trigger's identity is its ordinal within a
        run. An off-by-one here misreports analysed time and shifts every trigger.
        """
        windows = unfold_windows(np.arange(n, dtype=np.float32), window, stride)
        assert windows.shape == (1 + (n - window) // stride, window)

    @pytest.mark.parametrize("stride", [1, 3, 10, 17, 64])
    def test_last_window_inside_block(self, stride):
        """
        The last window ends at or before the end of the block, and one more will not fit.

        The block handed in is exactly one segment's worth of samples, so a window running
        past its end is a read across a chunk boundary. Maximality is asserted alongside,
        because refusing to emit a perfectly good final window silently drops livetime.
        """
        n, window = 500, 37
        windows = unfold_windows(np.arange(n, dtype=np.float32), window, stride)
        last_start = (windows.shape[0] - 1) * stride
        assert last_start + window <= n
        assert last_start + stride + window > n

    def test_contents_match_gather(self):
        """
        Every window equals the slice a per-window gather would have produced.

        The strided view is an optimisation of that gather; if the two disagree the
        stride arithmetic, not the loop, is wrong.
        """
        block = np.arange(97, dtype=np.float32)
        window, stride = 13, 5
        windows = unfold_windows(block, window, stride)
        expected = np.stack(
            [block[i * stride : i * stride + window] for i in range(windows.shape[0])]
        )
        assert np.array_equal(windows, expected)

    def test_wide_stride_leaves_gaps(self):
        """
        A stride longer than the window is allowed, and the windows do not overlap.

        Refusing such a stride is :class:`SearchGeometry`'s job, as it is the object that
        knows the gap would be unanalysed time. This is index arithmetic and does not
        re-litigate the decision, so the two cannot disagree about what a lattice is.
        """
        block = np.arange(50, dtype=np.float32)
        windows = unfold_windows(block, 10, 15)
        assert windows.shape == (3, 10)
        assert np.array_equal(windows[1], block[15:25])
        # The block's value is its own index, so the skipped samples are visible directly.
        assert int(windows[1][0] - windows[0][-1]) - 1 == 15 - 10


class TestUnfoldDegenerate:
    """Empty and refused inputs."""

    @pytest.mark.parametrize("n", [0, 1, 9])
    def test_short_block_yields_none(self, n):
        """
        A short block gives an empty result rather than raising.

        Segment tails and restricted intervals routinely leave fewer than a window of
        samples. Raising would force every caller to pre-check the same arithmetic, and a
        caller that forgot would abort a block over a boundary condition that is normal.
        """
        windows = unfold_windows(np.arange(n, dtype=np.float32), 10, 1)
        assert windows.shape == (0, 10)
        assert windows.dtype == np.float32

    def test_exactly_one_window(self):
        """A block the length of a window yields one window, not zero."""
        block = np.arange(10, dtype=np.float32)
        windows = unfold_windows(block, 10, 3)
        assert windows.shape == (1, 10)
        assert np.array_equal(windows[0], block)

    @pytest.mark.parametrize("stride", [0, -1])
    def test_nonpositive_stride_refused(self, stride):
        """
        A stride that does not advance the window start is an error, not an empty result.

        A zero stride would place every window at the same sample and emit an unbounded
        number of identical windows; silently returning one window instead would analyse a
        block once and report it as fully covered.
        """
        with pytest.raises(ValueError, match="stride"):
            unfold_windows(np.arange(50, dtype=np.float32), 10, stride)

    @pytest.mark.parametrize("window", [0, -4])
    def test_nonpositive_window_refused(self, window):
        """A window of no samples has no meaning and divides the block into nothing."""
        with pytest.raises(ValueError, match="window"):
            unfold_windows(np.arange(50, dtype=np.float32), window, 5)

    def test_multidim_block_refused(self):
        """
        The block is one segment's samples, a flat stream.

        A ``(detector, sample)`` array unfolded along the wrong axis would produce windows
        that look right and mix detectors, so the shape is checked instead of guessed.
        """
        with pytest.raises(ValueError, match="one-dimensional"):
            unfold_windows(np.zeros((2, 50), dtype=np.float32), 10, 5)


class TestUnfoldIsAView:
    """The property that makes a hundred-million-window search affordable."""

    @pytest.mark.parametrize("stride", [1, 2, 3, 32, 64, 205, 512, 1024])
    def test_windows_are_views(self, stride):
        """
        No stride copies. The result is a strided view of the block.

        A run unfolds of order 1e8 windows of 32768 float32 samples: copying per window
        would move about 12 TB per detector to produce data the block already holds. The
        strides here run from one sample to twice the window, since the case that would
        force a copy, if one existed, would be a stride that is not a whole number of
        samples -- and the geometry makes the stride an integer for exactly that reason.
        """
        block = np.arange(4096, dtype=np.float32)
        windows = unfold_windows(block, 512, stride)
        assert windows.shape[0] > 1
        assert np.shares_memory(windows, block)

    def test_view_is_read_only(self):
        """
        Windows cannot be written through.

        Consecutive windows share ``window - stride`` samples, so an in-place edit of one
        window would silently alter its neighbours; a whitener or an injection applied
        window-by-window would corrupt the block it is reading from.
        """
        windows = unfold_windows(np.arange(50, dtype=np.float32), 10, 3)
        assert windows.flags.writeable is False
        with pytest.raises(ValueError):
            windows[0, 0] = 1.0

    def test_strided_block_unfolds(self):
        """A block that is itself a view is not silently materialised first."""
        base = np.arange(200, dtype=np.float32)
        block = base[::2]
        windows = unfold_windows(block, 10, 3)
        assert np.shares_memory(windows, base)
        assert np.array_equal(windows[1], block[3:13])


class TestReadSegmentSpan:
    """Reads are addressed through the segment and bounded by it."""

    def test_span_uses_file_offset(self):
        """
        A read is placed by ``sample_start_idx``, not by position in the sidecar.

        Records are numbered in an order anti-correlated with GPS, so an implementation
        that used the record's position, or its GPS rank, would read a plausible-looking
        stretch of the wrong chunk.
        """
        segment = _segment(nsamples=100, sample_start_idx=1000, dyn_range_fac=1.0)
        got = read_segment_span(_file(), segment, 10, 5)
        assert np.array_equal(got, np.arange(1010, 1015, dtype=np.float32))

    def test_dyn_range_divided_out(self, small_release):
        """
        Stored samples are scaled by ``dyn_range_fac``; the reader returns strain.

        The factor is 5.9e20, so leaving it in place changes nothing structurally and
        everything numerically: the whitener would see amplitudes 20 orders of magnitude
        from the PSD it divides by.
        """
        segments, mmap = _open(small_release)
        segment = segments[0]
        raw = np.asarray(mmap[segment.sample_start_idx : segment.sample_start_idx + 8])
        got = read_segment_span(mmap, segment, 0, 8)
        assert segment.dyn_range_fac == pytest.approx(DYN_RANGE_FAC)
        assert np.array_equal(got, raw / segment.dyn_range_fac)
        # The fixture fills chunk k with the constant k+1, recovered after the division.
        assert got == pytest.approx(segment.segment_index + 1, rel=1e-5)

    def test_dtype_preserved(self, small_release):
        """
        Dividing out the factor must not promote float32 to float64.

        A run reads of order 1e8 windows; promoting on the way out doubles the bytes moved
        and the device transfer for information the release does not carry.
        """
        segments, mmap = _open(small_release)
        assert read_segment_span(mmap, segments[0], 0, 16).dtype == np.float32

    def test_result_does_not_alias(self):
        """
        The returned array is independent of the memmap.

        The block is handed to the conditioning pipeline, which may work in place; an
        alias
        would write back through the mapping into the release on disk.
        """
        segment = _segment(dyn_range_fac=2.0)
        mmap = _file()
        got = read_segment_span(mmap, segment, 0, 20)
        assert not np.shares_memory(got, mmap)

    def test_whole_segment_readable(self):
        """The bound is inclusive: a span ending on the last sample is legal."""
        segment = _segment(nsamples=100, sample_start_idx=1000, dyn_range_fac=1.0)
        got = read_segment_span(_file(), segment, 0, 100)
        assert got.shape == (100,)

    def test_zero_length_span(self):
        """An empty request reads nothing rather than raising; it is inside the chunk."""
        got = read_segment_span(_file(), _segment(), 10, 0)
        assert got.shape == (0,)

    @pytest.mark.parametrize(
        "first_local,n_samples",
        [(95, 10), (100, 1), (101, 0), (-1, 5), (0, 101)],
    )
    def test_span_outside_segment_refused(self, first_local, n_samples):
        """
        A span that is not wholly inside the segment is an error, never a clamp.

        GPS is a per-segment coordinate: the sample after a segment's last belongs to a
        chunk ~496 s away, and the overlapping samples of two chunks differ because each
        was resampled and filtered on its own boundaries. Clamping would return the wrong
        data under a right-looking segment index, which no downstream check can catch.
        """
        with pytest.raises(ValueError, match="segment"):
            read_segment_span(_file(), _segment(nsamples=100), first_local, n_samples)

    def test_negative_count_refused(self):
        """A negative count would slice backwards and return an empty array silently."""
        with pytest.raises(ValueError, match="n_samples"):
            read_segment_span(_file(), _segment(), 10, -5)

    def test_segment_past_file_refused(self):
        """
        A sidecar that indexes beyond the ``.bin`` fails loudly.

        Slicing a memmap past its end returns a short array rather than raising, so a
        truncated or mismatched release would otherwise surface as a batch quietly missing
        its final samples.
        """
        segment = _segment(nsamples=100, sample_start_idx=0)
        with pytest.raises(ValueError, match="outside the 50 samples"):
            read_segment_span(_file(n=50), segment, 0, 100)

    def test_bad_dyn_range_refused(self):
        """A zero or negative factor is a corrupt sidecar, not a scale to divide by."""
        with pytest.raises(ValueError, match="dyn_range_fac"):
            read_segment_span(_file(), _segment(dyn_range_fac=0.0), 0, 10)

    def test_multidim_release_refused(self):
        """The ``.bin`` is a flat sample stream; a shaped array would slice by row."""
        with pytest.raises(ValueError, match="flat"):
            read_segment_span(np.zeros((4, 100), dtype=np.float32), _segment(), 0, 10)


class TestSegmentBoundaryInvariant:
    """The oracle: a window that crossed a chunk boundary is not constant."""

    def test_window_holds_one_chunk(self, synthetic_release):
        """
        No window emitted for the real 512 s chunk geometry spans two chunks.

        Each chunk of the fixture is filled with its own constant, so a window that stayed
        inside its owning chunk has zero peak-to-peak and a spliced one cannot. Run on
        the full-size release, since this is the assertion the whole reader exists to
        satisfy.
        """
        segments, mmap = _open(synthetic_release)
        spans, _ = window_hosts(segments, WINDOW_SAMPLES, STRIDE_SAMPLES)
        assert spans
        for span in spans:
            n_samples = (span.n_windows - 1) * span.stride_samples + WINDOW_SAMPLES
            block = read_segment_span(
                mmap, span.segment, span.first_local, n_samples
            )
            windows = unfold_windows(block, WINDOW_SAMPLES, span.stride_samples)
            assert np.all(np.ptp(windows, axis=1) == 0.0)
            assert windows.shape[0] == span.n_windows

    def test_window_carries_owner_value(self, small_release):
        """
        A window's constant identifies the chunk it came from.

        Constancy alone would survive a read of the wrong chunk, since every chunk is
        internally constant. The value pins that the samples came from the segment the
        lattice named, which matters because the fixture numbers records against GPS order
        exactly as the real sidecars do.
        """
        segments, mmap = _open(small_release)
        spans, _ = window_hosts(segments, WINDOW_SAMPLES, STRIDE_SAMPLES)
        assert spans
        for span in spans:
            n_samples = (span.n_windows - 1) * span.stride_samples + WINDOW_SAMPLES
            block = read_segment_span(
                mmap, span.segment, span.first_local, n_samples
            )
            windows = unfold_windows(block, WINDOW_SAMPLES, span.stride_samples)
            assert windows[:, 0] == pytest.approx(
                span.segment.segment_index + 1, rel=1e-5
            )

    def test_oracle_fires_on_splice(self, small_release):
        """
        A read across the join is caught by the same assertion, so the oracle has teeth.

        The negative control for the two tests above: a synthetic release is contiguous in
        file index, so a reader that ignored segment bounds entirely would still return
        data of the right shape and pass any weaker check.
        """
        segments, mmap = _open(small_release)
        first, second = segments[0], segments[1]
        assert first.sample_start_idx + first.nsamples == second.sample_start_idx
        start = second.sample_start_idx - WINDOW_SAMPLES // 2
        spliced = np.asarray(mmap[start : start + WINDOW_SAMPLES + STRIDE_SAMPLES])
        windows = unfold_windows(spliced, WINDOW_SAMPLES, STRIDE_SAMPLES)
        assert np.all(np.ptp(windows, axis=1) > 0.0)

    def test_reader_refuses_splice(self, small_release):
        """
        The spliced span above is refused from either chunk, not clamped into one.

        It is the same file indices in both cases; only the record named differs. Refusing
        both is what makes the boundary a property of the data rather than of which record
        the caller happened to pass.
        """
        segments, mmap = _open(small_release)
        first, second = segments[0], segments[1]
        with pytest.raises(ValueError, match="segment"):
            read_segment_span(
                mmap, first, first.nsamples - WINDOW_SAMPLES // 2, WINDOW_SAMPLES
            )
        with pytest.raises(ValueError, match="segment"):
            read_segment_span(mmap, second, -(WINDOW_SAMPLES // 2), WINDOW_SAMPLES)

    @pytest.mark.parametrize("offsets_s", [None, {"L1": 50.0}])
    def test_lattice_windows_intact(
        self, small_release, offsets_s
    ):
        """
        The invariant holds for the followers, and under a time slide.

        The lattice is carried by one detector and the others follow it across their own,
        unrelated chunk boundaries; a slide moves a follower onto a different chunk again.
        A follower run that ran off the end of its segment would splice two chunks in one
        detector only, which is indistinguishable downstream from a coincident glitch.
        """
        opened = {det: _open(small_release, det) for det in ("H1", "L1")}
        segments = {det: pair[0] for det, pair in opened.items()}
        mmaps = {det: pair[1] for det, pair in opened.items()}
        grid = AnalysisGrid.build(
            GEOMETRY,
            segments,
            coincident_intervals(segments, shifts_s=offsets_s),
            offsets_s=offsets_s,
            reference_detector="H1",
        )
        assert len(grid) > 0
        seen = 0
        for block in grid.blocks(120.0):
            for detector in grid.detectors:
                for run in grid.iter_block_detector(block, detector):
                    n_samples = (
                        run.n_windows - 1
                    ) * run.stride_samples + WINDOW_SAMPLES
                    strain = read_segment_span(
                        mmaps[detector], run.segment, run.first_local, n_samples
                    )
                    windows = unfold_windows(
                        strain, WINDOW_SAMPLES, run.stride_samples
                    )
                    assert np.all(np.ptp(windows, axis=1) == 0.0)
                    assert windows.shape[0] == run.n_windows
                    seen += run.n_windows
        assert seen == 2 * len(grid)


def _grid(release, detectors=("H1", "L1"), **kw):
    """The lattice over a release, built the way a campaign builds it."""
    segments = {
        d: load_segments(release / f"data_{d}_O3a_segments.json") for d in detectors
    }
    return AnalysisGrid.build(
        GEOMETRY, segments, coincident_intervals(segments), **kw
    )


class TestStreamingReader:
    """Construction, refusal, and what a batch carries."""

    def test_reader_opens_release(self, small_release):
        """
        A reader is constructed from a release directory and a lattice.

        Construction is where the memmaps are acquired, so it is also where a release that
        does not match the lattice must be refused.
        """
        reader = StreamingStrainReader(small_release, _grid(small_release), GEOMETRY)
        try:
            assert reader.detectors == ("H1", "L1")
            assert reader.blocks
        finally:
            reader.close()

    def test_missing_stream_refused(self, small_release, tmp_path):
        """
        A sidecar with no stream behind it is refused at construction.

        Discovered at first read instead, it would have written triggers from whatever the
        earlier reads returned, and those are indistinguishable from real ones afterwards.
        """
        bare = tmp_path / "sidecars_only"
        bare.mkdir()
        for name in small_release.glob("*_segments.json"):
            (bare / name.name).write_bytes(name.read_bytes())
        with pytest.raises(FileNotFoundError, match="no strain for"):
            StreamingStrainReader(bare, _grid(small_release), GEOMETRY)

    def test_mismatched_geometry_refused(self, small_release):
        """
        A geometry that is not the lattice's is refused.

        Unfolding at a stride the lattice was not built on shifts every window start after
        the first, and the shift is invisible because both are self-consistent.
        """
        other = dataclasses.replace(GEOMETRY, stride_samples=STRIDE_SAMPLES + 1)
        with pytest.raises(ValueError, match="geometry"):
            StreamingStrainReader(small_release, _grid(small_release), other)

    def test_bad_batch_size_refused(self, small_release):
        """A batch of no windows never advances."""
        for bad in (0, -1):
            with pytest.raises(ValueError, match="batch_size"):
                StreamingStrainReader(
                    small_release, _grid(small_release), GEOMETRY, batch_size=bad
                )

    def test_batch_reports_count(self):
        """
        A batch's length is the number of windows it carries.

        The batch is clamped to the windows left in the owning segment, so its length is
        not the requested batch size and every consumer has to read it from the batch.
        """
        batch = WindowBatch(
            strain=np.zeros((3, 2, 8), dtype=np.float32),
            gps=np.zeros(3),
            segment_index=np.zeros(3, dtype=np.int64),
            local_start=np.zeros(3, dtype=np.int64),
            slide_id=0,
        )

        assert len(batch) == 3


class TestStreamingReaderCoversTheLattice:
    """Every window the lattice declares is read exactly once."""

    def test_window_count_matches_the_lattice(self, small_release):
        """
        The reader emits `len(grid)` windows -- no more, no fewer.

        The lattice is what the livetime is measured from, so a reader that dropped a
        window at every segment tail would search less time than the denominator says and
        report every rate low.
        """
        grid = _grid(small_release)
        reader = StreamingStrainReader(small_release, grid, GEOMETRY, batch_size=64)
        try:
            assert sum(len(batch) for batch in reader) == len(grid)
        finally:
            reader.close()

    def test_gps_matches_the_lattice(self, small_release):
        """Window start times come back in lattice order and are the lattice's own."""
        grid = _grid(small_release)
        reader = StreamingStrainReader(small_release, grid, GEOMETRY, batch_size=37)
        try:
            seen = np.concatenate([batch.gps for batch in reader])
        finally:
            reader.close()
        expected = np.concatenate([grid.gps(block) for block in grid.blocks(32768.0)])

        assert np.array_equal(seen, expected)

    def test_batch_size_does_not_change_the_result(self, small_release):
        """
        Batching is a transport detail, not part of the answer.

        A different batch size splits the same lattice differently and must deliver the
        same windows in the same order, or the triggers a campaign finds would depend on
        a memory setting.
        """
        grid = _grid(small_release)
        collected = []
        for size in (16, 64, 1000):
            reader = StreamingStrainReader(
                small_release, grid, GEOMETRY, batch_size=size, prefetch=0
            )
            try:
                collected.append(
                    np.concatenate([np.asarray(b.strain).copy() for b in reader])
                )
            finally:
                reader.close()

        assert np.array_equal(collected[0], collected[1])
        assert np.array_equal(collected[0], collected[2])

    def test_batch_is_clamped_to_the_segment(self, small_release):
        """
        The effective batch never exceeds what the current segment can supply.

        Requesting more than a whole segment holds must yield short batches rather than a
        read that runs past the boundary into a chunk ~496 s away.
        """
        grid = _grid(small_release)
        reader = StreamingStrainReader(
            small_release, grid, GEOMETRY, batch_size=10**6, prefetch=0
        )
        try:
            sizes = [len(batch) for batch in reader]
        finally:
            reader.close()

        assert len(sizes) > 1
        assert max(sizes) < 10**6


class TestStreamingReaderBoundaries:
    """The invariant the whole reader exists for."""

    def test_no_window_spans_a_chunk(self, small_release):
        """
        Every window carries one chunk's worth of strain, in every detector.

        The release fills chunk *k* with the constant *k*, so a window that spliced two
        chunks has a non-zero range. This is the oracle a contiguous fixture cannot
        provide, and the reason the fixture is built the way it is.
        """
        grid = _grid(small_release)
        reader = StreamingStrainReader(small_release, grid, GEOMETRY, batch_size=64)
        try:
            for batch in reader:
                spans = np.ptp(np.asarray(batch.strain), axis=-1)
                assert not np.any(spans), "a window spliced two chunks"
        finally:
            reader.close()

    def test_strain_views_its_block(self, small_release):
        """
        The batch does not materialise its windows.

        Consecutive windows overlap by `window - stride` samples, so at the default batch
        size the materialised form is 2.1 GB while the samples behind it are 6.8 MB. A
        reader that copied would move that 2.1 GB for every one of ~92 million windows.
        """
        grid = _grid(small_release)
        reader = StreamingStrainReader(small_release, grid, GEOMETRY, batch_size=64)
        try:
            batch = next(iter(reader))
        finally:
            reader.close()

        assert batch.block is not None
        assert np.asarray(batch.strain).base is not None
        assert batch.block.nbytes < np.asarray(batch.strain).nbytes
        assert not np.asarray(batch.strain).flags.writeable

    def test_identity_is_the_reference_detector(self, small_release):
        """
        `segment_index` and `local_start` identify the window in the reference frame.

        That is the persisted identity in the trigger schema. A follower's position is
        this plus the slide offset, so recording it per detector would store something
        already implied and let the two disagree.
        """
        grid = _grid(small_release)
        reader = StreamingStrainReader(small_release, grid, GEOMETRY, batch_size=64)
        try:
            batch = next(iter(reader))
        finally:
            reader.close()

        assert batch.segment_index.shape == (len(batch),)
        assert len(set(batch.segment_index.tolist())) == 1
        assert np.array_equal(
            np.diff(batch.local_start),
            np.full(len(batch) - 1, STRIDE_SAMPLES, dtype=np.int64),
        )


class TestStreamingReaderResume:
    """Seeking, closing and reading ahead."""

    def test_seek_skips_earlier_blocks(self, small_release):
        """A campaign resumes at a completed block boundary, not from the start."""
        grid = _grid(small_release, )
        reader = StreamingStrainReader(
            small_release, grid, GEOMETRY, batch_size=64, block_seconds=8.0
        )
        try:
            assert len(reader.blocks) > 1
            whole = sum(len(batch) for batch in reader)
            reader.seek(reader.blocks[1].block_id)
            resumed = sum(len(batch) for batch in reader)
            first = sum(len(b) for b in reader.iter_block(reader.blocks[0]))
        finally:
            reader.close()

        assert resumed == whole - first

    def test_seek_to_unknown_block_refused(self, small_release):
        """Resuming at a block the lattice does not have would silently read nothing."""
        reader = StreamingStrainReader(small_release, _grid(small_release), GEOMETRY)
        try:
            with pytest.raises(ValueError, match="not in this lattice"):
                reader.seek(9999)
        finally:
            reader.close()

    def test_close_is_idempotent_and_final(self, small_release):
        """
        Closing twice is not an error; reading after it is.

        A caller closing in a `finally` after an exception that already closed it must not
        raise a second exception over the first.
        """
        reader = StreamingStrainReader(small_release, _grid(small_release), GEOMETRY)
        reader.close()
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            list(reader)

    def test_prefetch_does_not_change_the_result(self, small_release):
        """
        Reading ahead is a latency device and must not alter the data or its order.

        The producer runs on another thread, so an ordering bug here would show up as
        triggers attributed to the wrong times, intermittently.
        """
        grid = _grid(small_release)
        out = []
        for depth in (0, 1, 4):
            reader = StreamingStrainReader(
                small_release, grid, GEOMETRY, batch_size=32, prefetch=depth
            )
            try:
                out.append(np.concatenate([batch.gps for batch in reader]))
            finally:
                reader.close()

        assert np.array_equal(out[0], out[1])
        assert np.array_equal(out[0], out[2])

    def test_context_manager_closes(self, small_release):
        """The reader releases its memmaps on the way out of a with-block."""
        with StreamingStrainReader(
            small_release, _grid(small_release), GEOMETRY
        ) as reader:
            assert reader.blocks
        with pytest.raises(ValueError, match="closed"):
            list(reader)

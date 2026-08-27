#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : reader.py
Description   : Segment-ordered streaming strain reader over the memmap release.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Reads are confined to a single segment -- and refused, never clamped, when asked to go
outside one. Sample index ``n+1`` at a segment end belongs to a different chunk whose GPS
start is ~496 s away, so a read spanning the boundary splices two unrelated epochs.
Consecutive windows overlap by ``window - stride`` samples, so blocks are read once and
expanded with ``unfold`` rather than gathered per window.
"""

import numbers
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _samples(name: str, value) -> int:
    """
    Read an argument that counts samples, refusing anything not a whole number of them.

    Truncating instead would put the lattice somewhere the caller did not ask for and
    leave no trace: 0.1 s at 2048 Hz is 204.8 samples, and silently analysing at 204
    shifts every window start after the first.
    :class:`~sage.search.geometry.SearchGeometry` refuses that same value by name, so
    accepting it here would let it in through the back door.
    """
    if isinstance(value, bool) or not isinstance(value, (numbers.Integral, float, np.floating)):
        raise TypeError(
            f"{name} must be a whole number of samples, got {value!r}"
        )
    if not float(value).is_integer():
        raise ValueError(
            f"{name} must be a whole number of samples, got {value}; 0.1 s at 2048 Hz "
            "is 204.8 samples, which is why sample counts are specified in samples and "
            "not in seconds"
        )
    return int(value)

from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid, Block
from sage.search.segments import Segment


@dataclass
class WindowBatch:
    """
    One batch of raw, dyn-range-corrected strain windows.

    ``strain`` is a **read-only strided view** of shape ``(batch, detectors, window)``,
    not an array. Consecutive windows overlap by ``window - stride`` samples, so at the
    default batch size the materialised form is 2.1 GB per batch while the samples behind
    it are 6.8 MB -- a 300x amplification that exists only because the lattice oversamples.
    Copying it would move that 2.1 GB per batch, per detector-pair, for every one of ~92
    million O3a windows.

    ``block`` is what was actually read: a contiguous ``(detectors, samples)`` array that
    ``strain`` views into. A consumer moving data to a device should transfer this and
    unfold there -- ``torch.Tensor.unfold`` is a view on GPU as ``sliding_window_view`` is
    here -- rather than transfer ``strain``, which would force it contiguous first.

    ``gps``, ``segment_index`` and ``local_start`` are the **reference detector's**, one
    per window. That is the persisted identity in the trigger schema, and it is enough:
    a follower's position is this plus the slide's offset, so storing it per detector
    would record something already implied and let the two disagree.
    """

    strain: "np.ndarray"
    gps: np.ndarray
    segment_index: np.ndarray
    local_start: np.ndarray
    slide_id: int
    block: Optional[np.ndarray] = None
    detectors: Tuple[str, ...] = ()

    def __len__(self) -> int:
        """
        Number of windows in the batch.

        Read from the batch rather than assumed: the effective size is clamped to the
        windows left in the current owning segment in *every* detector, so it is normally
        smaller than the requested batch size and is 1 at a segment tail.
        """
        return int(np.asarray(self.strain).shape[0])


def _unfold_rows(block: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    """
    Unfold every row of a ``(detectors, samples)`` block into overlapping windows.

    Returns a read-only ``(windows, detectors, window_samples)`` view. The transpose is
    what makes this free: for a C-contiguous block the detector axis has stride
    ``samples``, so moving it inside the window axis is a stride permutation rather than a
    copy. :func:`unfold_windows` is the one-dimensional case and is what the index
    arithmetic is tested through; this is the same operation applied per row.
    """
    block = np.asarray(block)
    if block.ndim != 2:
        raise ValueError(
            f"block must be (detectors, samples); got shape {tuple(block.shape)}"
        )
    if block.shape[1] < window_samples:
        empty = np.empty((0, block.shape[0], window_samples), dtype=block.dtype)
        empty.flags.writeable = False
        return empty
    view = sliding_window_view(block, window_samples, axis=1)[:, ::stride_samples]
    return np.transpose(view, (1, 0, 2))


class _FlatStream:
    """
    One detector's strain as a single memory-mapped sample stream.

    The training releases' layout: every segment lives at ``sample_start_idx`` inside one
    contiguous ``.bin``, and reading is pure index arithmetic.
    """

    def __init__(self, path: Path, segments: Sequence[Segment]) -> None:
        self.path = path
        self.mmap = np.memmap(path, dtype="<f4", mode="r")
        needed = max(s.sample_start_idx + s.nsamples for s in segments)
        if self.mmap.shape[0] < needed:
            raise ValueError(
                f"{path.name} holds {self.mmap.shape[0]} samples but its sidecar places "
                f"segments out to {needed}; the two describe different releases"
            )

    def read(self, segment: Segment, first_local: int, n_samples: int) -> np.ndarray:
        return read_segment_span(self.mmap, segment, first_local, n_samples)

    def close(self) -> None:
        self.mmap = None


class _SegmentedStream:
    """
    One detector's strain as one HDF5 dataset per segment.

    The search-grade layout. Segments are separate datasets rather than offsets into one
    stream, which is what lets the search release keep the known events the training
    release removed: a segment can be rebuilt without renumbering every sample after it.

    ``sample_start_idx`` is still carried and still describes a virtual concatenation, but
    nothing here indexes through it -- the segment's own dataset is addressed directly, so
    a sidecar whose offsets drifted cannot silently return a neighbour's samples.
    """

    def __init__(self, path: Path, segments: Sequence[Segment]) -> None:
        import h5py

        self.path = path
        self.handle = h5py.File(path, "r")
        missing = sorted(
            {s.dataset for s in segments if s.dataset and s.dataset not in self.handle}
        )
        if missing:
            raise ValueError(
                f"{path.name} is missing {len(missing)} of the datasets its sidecar "
                f"names, starting with {missing[0]!r}; the sidecar and the release "
                "disagree about what was written"
            )

    def read(self, segment: Segment, first_local: int, n_samples: int) -> np.ndarray:
        if segment.dataset is None:
            raise ValueError(
                f"segment {segment.segment_index} of {segment.detector} names no dataset, "
                f"so it cannot be read from {self.path.name}, which stores one dataset per "
                "segment"
            )
        first_local = _samples("first_local", first_local)
        n_samples = _samples("n_samples", n_samples)
        if n_samples < 0:
            raise ValueError(f"n_samples must not be negative, got {n_samples}")
        if first_local < 0 or first_local + n_samples > segment.nsamples:
            raise ValueError(
                f"span [{first_local}, {first_local + n_samples}) is not contained in "
                f"segment {segment.segment_index} of {segment.detector}, which holds "
                f"{segment.nsamples} samples"
            )
        if not np.isfinite(segment.dyn_range_fac) or not segment.dyn_range_fac > 0:
            raise ValueError(
                f"segment {segment.segment_index} of {segment.detector} declares "
                f"dyn_range_fac={segment.dyn_range_fac}, which cannot be divided out"
            )
        block = np.asarray(
            self.handle[segment.dataset][first_local : first_local + n_samples]
        )
        return np.divide(block, segment.dyn_range_fac, dtype=block.dtype)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class StreamingStrainReader:
    """
    Iterate a run's window lattice in segment order, one block at a time.

    Reads are confined to a single segment in every detector at once. The effective batch
    is therefore the smallest number of windows any detector has left in its current
    segment -- the reference detector's segmentation and a follower's do not line up once
    a slide offset is applied, so bounding on the reference alone would splice the
    follower across a boundary.

    Parameters
    ----------
    release_dir : path
        Directory holding ``data_{det}_{run}.bin`` and its sidecars.
    grid : AnalysisGrid
        The lattice to walk, including any slide offsets.
    batch_size : int
        Upper bound only; the effective batch is clamped to the windows remaining
        in the current owning segment.
    block_seconds : float
        Work-block length, matching ``EngineSpec.block_seconds``. A block is a whole
        number of spans, so a block boundary never falls inside a run of windows.
    prefetch : int
        Batches read ahead on a background thread. ``0`` reads synchronously. The read is
        sequential and small -- 820 bytes per window, one stride's worth -- so this hides
        NFS latency rather than bandwidth.
    pin_memory : bool
        Recorded for the consumer that moves batches to a device. Nothing is pinned here:
        pinning is a torch allocation, and this module stays numpy-only so that reading
        strain does not require a GPU stack.
    """

    def __init__(
        self,
        release_dir: str | Path,
        grid: AnalysisGrid,
        geometry: SearchGeometry,
        batch_size: int = 8192,
        prefetch: int = 2,
        pin_memory: bool = True,
        block_seconds: float = 32768.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if prefetch < 0:
            raise ValueError(f"prefetch must not be negative, got {prefetch}")
        if geometry != grid.geometry:
            raise ValueError(
                "the reader's geometry and the lattice's disagree; unfolding at a stride "
                "the lattice was not built on shifts every window start after the first, "
                "and the shift is invisible because both are self-consistent"
            )
        self.release_dir = Path(release_dir)
        self.grid = grid
        self.geometry = geometry
        self.batch_size = int(batch_size)
        # Kept so the engine can walk the same partition rather than infer one. Inferring
        # it from the blocks was wrong in a way nothing surfaced: a block's `duration_s`
        # is its wall span, gaps included, and re-partitioning at the largest of those
        # gave a coarser set of blocks than the reader held.
        self.block_seconds = float(block_seconds)
        self.prefetch = int(prefetch)
        self.pin_memory = bool(pin_memory)
        self.detectors: Tuple[str, ...] = tuple(grid.detectors)
        self._start_block = 0
        self._closed = False
        self._blocks: List[Block] = grid.blocks(float(block_seconds))
        self._streams: Dict[str, object] = {}
        for detector in self.detectors:
            self._streams[detector] = self._open(detector)

    def _open(self, detector: str):
        """Open one detector's strain for this lattice."""
        return open_stream(
            self.release_dir, detector, self.grid.segments_by_detector.get(detector)
        )

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("this reader is closed; its strain files have been released")

    def __iter__(self) -> Iterator[WindowBatch]:
        """Yield batches in lattice order, from the seek position onwards."""
        self._require_open()
        for block in self._blocks[self._start_block :]:
            yield from self.iter_block(block)

    def iter_block(self, block: Block) -> Iterator[WindowBatch]:
        """Yield batches for a single block."""
        self._require_open()
        stream = self._iter_block_sync(block)
        if self.prefetch <= 0:
            yield from stream
            return
        yield from _prefetched(stream, self.prefetch)

    def _iter_block_sync(self, block: Block) -> Iterator[WindowBatch]:
        """
        Walk one block, reading each detector inside whichever segment it is currently in.

        The cursor is per detector because the segmentations differ: under a slide the
        follower's data is read at ``gps + offset``, which crosses its own segment
        boundaries at moments unrelated to the reference's. Each step advances every
        detector by the same number of windows -- they are the same windows -- but each
        may be at a different place in a different segment when it does.
        """
        runs = {
            detector: list(self.grid.iter_block_detector(block, detector))
            for detector in self.detectors
        }
        totals = {
            detector: sum(run.n_windows for run in detector_runs)
            for detector, detector_runs in runs.items()
        }
        reference = self.grid.reference_detector
        total = totals[reference]
        disagreeing = {d: n for d, n in totals.items() if n != total}
        if disagreeing:
            raise ValueError(
                f"block {block.block_id} carries {total} windows in the reference "
                f"detector but {disagreeing} elsewhere; the detectors would be read "
                "past one another and every window after the first shortfall would pair "
                "strain from different moments"
            )
        if total == 0:
            return

        gps = np.asarray(self.grid.gps(block), dtype=np.float64)
        window = self.geometry.window_samples
        stride = self.geometry.stride_samples
        # (run index, windows already taken from it), one cursor per detector.
        position = {detector: [0, 0] for detector in self.detectors}
        cursor = 0
        while cursor < total:
            take = min(self.batch_size, total - cursor)
            for detector in self.detectors:
                index, taken = position[detector]
                take = min(take, runs[detector][index].n_windows - taken)
            if take <= 0:
                raise ValueError(
                    f"block {block.block_id} stalled at window {cursor}; a run reports "
                    "no windows left while the block is not finished"
                )

            n_samples = window + stride * (take - 1)
            raw = np.empty((len(self.detectors), n_samples), dtype=np.float32)
            for row, detector in enumerate(self.detectors):
                index, taken = position[detector]
                run = runs[detector][index]
                raw[row] = self._streams[detector].read(
                    run.segment, run.first_local + stride * taken, n_samples
                )

            index, taken = position[reference]
            reference_run = runs[reference][index]
            yield WindowBatch(
                strain=_unfold_rows(raw, window, stride),
                gps=gps[cursor : cursor + take],
                segment_index=np.full(
                    take, reference_run.segment.segment_index, dtype=np.int64
                ),
                local_start=(
                    reference_run.first_local
                    + stride * (taken + np.arange(take, dtype=np.int64))
                ),
                slide_id=int(self.grid.slide_id),
                block=raw,
                detectors=self.detectors,
            )

            for detector in self.detectors:
                position[detector][1] += take
                if position[detector][1] >= runs[detector][position[detector][0]].n_windows:
                    position[detector][0] += 1
                    position[detector][1] = 0
            cursor += take

    def seek(self, block_id: int) -> None:
        """
        Resume at a block boundary.

        Only at a boundary: a block is a whole number of spans, so resuming at one cannot
        land mid-run, and the shard written for a completed block is the unit the campaign
        records. Resuming mid-block would either duplicate triggers or drop them, and
        which of the two is not visible from the shard afterwards.
        """
        self._require_open()
        known = {block.block_id for block in self._blocks}
        if block_id not in known:
            raise ValueError(
                f"block {block_id} is not in this lattice, which holds "
                f"{len(self._blocks)} blocks numbered "
                f"{min(known) if known else 'none'}..{max(known) if known else 'none'}"
            )
        self._start_block = next(
            index
            for index, block in enumerate(self._blocks)
            if block.block_id == block_id
        )

    @property
    def blocks(self) -> List[Block]:
        """The work blocks this reader walks."""
        return list(self._blocks)

    def close(self) -> None:
        """
        Release the open strain files and the prefetch thread.

        Idempotent, so a caller closing in a ``finally`` after an exception that already
        closed it does not raise a second one over the first.
        """
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        self._closed = True

    def __enter__(self) -> "StreamingStrainReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_stream(release_dir: str | Path, detector: str, segments):
    """
    Open one detector's strain, refusing a release the sidecar disagrees with.

    Two layouts are supported and are told apart by what the sidecar carries, not by what
    the directory happens to contain. A segment naming a ``dataset`` belongs to a
    search-grade HDF5 release, one dataset per segment; one that does not belongs to a
    flat ``.bin`` training release, where ``sample_start_idx`` locates it in a single
    stream. Guessing from the filenames would pick the wrong reader for a directory
    holding both.

    Checked when the stream is opened rather than at first read: a mismatch discovered
    part way through a block has already written triggers from whatever the earlier reads
    returned, and those are indistinguishable from real ones afterwards.

    Module level rather than a reader method because the injection campaign draws its
    noise from the same release and must open it the same way. Reimplementing the choice
    there hardcoded the flat layout, while the search release is the segmented one.
    """
    release_dir = Path(release_dir)
    if not segments:
        raise ValueError(
            f"the lattice names detector {detector!r} but carries no segments for it"
        )
    run = segments[0].observing_run
    mixed = {segment.observing_run for segment in segments}
    if len(mixed) != 1:
        raise ValueError(
            f"{detector} segments span observing runs {sorted(mixed)}; one campaign "
            "covers one run, and the release file is named for it"
        )
    segmented = any(segment.dataset for segment in segments)
    if segmented and not all(segment.dataset for segment in segments):
        raise ValueError(
            f"{detector}'s sidecar names a dataset for some segments and not others, "
            "so the release is half one layout and half the other and neither reader "
            "can address all of it"
        )
    stem = release_dir / f"data_{detector}_{run}"
    path = stem.with_suffix(".h5" if segmented else ".bin")
    if not path.is_file():
        raise FileNotFoundError(
            f"no strain for {detector} at {path}; the lattice was built from "
            f"{stem}_segments.json, so the release is missing the stream its own "
            "sidecar indexes into"
        )
    if segmented:
        return _SegmentedStream(path, segments)
    return _FlatStream(path, segments)


_SENTINEL = object()


def _prefetched(stream: Iterator, depth: int) -> Iterator:
    """
    Read ahead on one background thread, bounded by ``depth``.

    Bounded rather than unbounded: a batch holds its own block, so an unbounded queue
    would read the whole run into memory whenever the consumer is slower than the reader,
    which on a GPU-bound search is always.

    The producer is a daemon and is stopped by the ``stop`` event, so a consumer that
    abandons the iterator part way -- a ``break``, or an exception downstream -- does not
    leave a thread reading a release nobody is consuming.
    """
    buffer: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()

    def produce() -> None:
        try:
            for item in stream:
                while not stop.is_set():
                    try:
                        buffer.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as error:  # re-raised in the consumer
            buffer.put(error)
        else:
            buffer.put(_SENTINEL)

    worker = threading.Thread(target=produce, name="strain-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = buffer.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        worker.join(timeout=5.0)


def read_segment_span(
    mmap: np.ndarray, segment: Segment, first_local: int, n_samples: int
) -> np.ndarray:
    """
    Read ``n_samples`` from one segment, dividing out ``dyn_range_fac``.

    A span that is not wholly inside ``segment`` is refused rather than clamped. Local
    sample ``nsamples`` is the first sample of a different chunk whose GPS start is ~496 s
    away, so a clamped or overrunning read returns unrelated strain from a right-looking
    segment index, and the splice is a discontinuity at the scale the search ranks on.

    The stored dtype is carried through. The release is float32 and a run reads of order
    1e8 windows, so promoting here would double the bytes moved for no added information.
    The division is performed in the stored dtype explicitly rather than left to
    promotion, which under NEP 50 would widen the block to float64 for a
    ``dyn_range_fac`` that happened to be a numpy scalar.

    Raises
    ------
    ValueError
        If ``mmap`` is not one-dimensional; if ``dyn_range_fac`` is not finite and
        positive; if ``n_samples`` is negative; if the span is not wholly inside the
        segment; if the segment's own placement falls outside the file; or if
        ``first_local`` or ``n_samples`` is not a whole number of samples.
    TypeError
        If ``first_local`` or ``n_samples`` is not a number at all.
    """
    if mmap.ndim != 1:
        raise ValueError(
            f"the release is a flat sample stream; got a {mmap.ndim}-D array"
        )
    # Written as a positive assertion so that NaN fails it. `<= 0` is passed by both NaN
    # and +inf, and an infinite factor returns an all-zero block -- which satisfies the
    # constant-fill boundary oracle downstream and is therefore invisible.
    if not np.isfinite(segment.dyn_range_fac) or not segment.dyn_range_fac > 0:
        raise ValueError(
            f"segment {segment.segment_index} of {segment.detector} declares "
            f"dyn_range_fac={segment.dyn_range_fac}, which cannot be divided out"
        )
    first_local = _samples("first_local", first_local)
    n_samples = _samples("n_samples", n_samples)
    if n_samples < 0:
        raise ValueError(f"n_samples must not be negative, got {n_samples}")
    if first_local < 0 or first_local + n_samples > segment.nsamples:
        raise ValueError(
            f"span [{first_local}, {first_local + n_samples}) is not contained in "
            f"segment {segment.segment_index} of {segment.detector}, which holds "
            f"{segment.nsamples} samples"
        )
    start = segment.global_index(first_local)
    stop = start + n_samples
    # Both ends: a negative start is a Python slice from the tail of the file, so a
    # sidecar placing a segment before the start of the release would return a
    # right-shaped block of strain from an unrelated chunk rather than fail.
    if start < 0 or stop > mmap.shape[0]:
        raise ValueError(
            f"segment {segment.segment_index} of {segment.detector} places samples "
            f"[{start}, {stop}) outside the {mmap.shape[0]} samples the file holds; "
            "the sidecar and the release disagree"
        )
    block = np.asarray(mmap[start:stop])
    return np.divide(block, segment.dyn_range_fac, dtype=block.dtype)


def unfold_windows(block: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    """
    Expand a contiguous block into overlapping windows without copying per window.

    Returns a strided, read-only view of ``block`` shaped
    ``(1 + (len(block) - window_samples) // stride_samples, window_samples)``. A run holds
    of order 1e8 windows of 32768 float32 samples; materialising them would move tens of
    terabytes to gain nothing, since the lattice is an affine index map and any positive
    stride expresses it as an array stride, overlapping or not. The view is read-only
    because consecutive windows share ``window_samples - stride_samples`` samples, so an
    in-place write would edit its neighbours as well.

    A block shorter than one window yields no windows rather than raising; that is the
    ordinary state of a segment tail, not an error. The empty result is read-only like
    the populated one, so the contract does not depend on how much data arrived. A stride
    longer than the window is accepted here and leaves gaps between windows; whether that
    is meaningful for a search is decided by
    :class:`~sage.search.geometry.SearchGeometry`, which refuses it.

    ``block`` is not required to be contiguous. A strided view unfolds correctly -- the
    result is a view of whatever samples the view addresses -- and what those samples
    mean is the caller's business, not this function's.

    Raises
    ------
    ValueError
        If ``block`` is not one-dimensional, if either count is not positive, or if
        either is not a whole number of samples.
    TypeError
        If either count is not a number at all.
    """
    window_samples = _samples("window_samples", window_samples)
    stride_samples = _samples("stride_samples", stride_samples)
    block = np.asarray(block)
    if block.ndim != 1:
        raise ValueError(
            f"block must be one-dimensional; got shape {tuple(block.shape)}"
        )
    if window_samples <= 0:
        raise ValueError(f"window_samples must be positive, got {window_samples}")
    if stride_samples <= 0:
        raise ValueError(
            f"stride_samples must be positive, got {stride_samples}; a non-positive "
            "stride does not advance the window start"
        )
    if block.shape[0] < window_samples:
        empty = np.empty((0, window_samples), dtype=block.dtype)
        empty.flags.writeable = False
        return empty
    return sliding_window_view(block, window_samples)[::stride_samples]

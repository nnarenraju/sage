#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : grid.py
Description   : Window lattice, blocks and coincidence bookkeeping.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One detector carries the lattice and the others follow it.

They cannot share one lattice. Segment start times are multiples of half a sample, not of
a whole one, so two detectors' sample grids coincide only about half the time; measured on
the O3a release, overlapping H1 and L1 segments are either exactly aligned or offset by
exactly half a sample. A non-reference detector's window therefore starts at the sample
nearest the reference time, and the residual -- at most half a sample, 0.244 ms, about
2.4 per cent of the H1-L1 light-travel time -- is recorded rather than absorbed.

The residual is constant along a run: both detectors advance by the same integer stride,
so it is fixed by the pair of segments and changes only when one of them ends.
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from sage.search.fingerprint import combine, digest_values
from sage.search.geometry import SearchGeometry
from pathlib import Path

from sage.search.segments import (
    HostSpan,
    Interval,
    Segment,
    hostable_intervals,
    intersect_intervals,
    sort_by_gps,
    window_hosts,
)


@dataclass(frozen=True)
class Block:
    """A contiguous unit of work: one shard, one resume marker, one cache residency."""

    block_id: int
    gps_start: float
    gps_end: float
    span_slice: Tuple[int, int]

    @property
    def duration_s(self) -> float:
        """Block wall span."""
        return self.gps_end - self.gps_start


@dataclass(frozen=True)
class DetectorRun:
    """A run of windows in one detector, aligned to the reference lattice."""

    detector: str
    segment: Segment
    first_local: int
    n_windows: int
    stride_samples: int
    residual_samples: float

    def starts_local(self) -> np.ndarray:
        """Segment-local start indices."""
        return self.first_local + self.stride_samples * np.arange(
            self.n_windows, dtype=np.int64
        )


@dataclass
class AnalysisGrid:
    """
    The window lattice for one run, one detector set and one slide.

    The persisted identity of a window is its owning segment plus its segment-local
    start sample. Ordinals over the concatenated span list are an iteration device
    only and are never written, since they shift whenever the span list changes.
    """

    geometry: SearchGeometry
    spans_by_detector: dict
    slide_id: int = 0
    offsets_s: Optional[dict] = None
    #: Integer shift along this lattice, per follower. When set, reference ordinal ``i``
    #: is paired with follower ordinal ``(i + k) mod N`` instead of with the follower's
    #: data at ``t_i + offset``. Every ordinal is hostable in every detector, so a
    #: shifted pairing cannot fall into a gap and loses no livetime at any shift.
    window_shift: Optional[dict] = None
    reference_detector: str = ""
    segments_by_detector: dict = field(default_factory=dict)
    coverage: Optional[object] = None

    @classmethod
    def build(
        cls,
        geometry: SearchGeometry,
        segments_by_detector: dict,
        coincident: Sequence[Interval],
        offsets_s: Optional[dict] = None,
        slide_id: int = 0,
        reference_detector: Optional[str] = None,
        coverage: bool = True,
        hostable_by_detector: Optional[dict] = None,
        window_shift: Optional[dict] = None,
    ) -> "AnalysisGrid":
        """
        Construct the lattice for the given coincident intervals and slide.

        ``coincident`` is expressed in the reference detector's frame, already accounting
        for any slide offsets, so the lattice covers exactly the time every detector has
        data for under this slide.

        Parameters
        ----------
        coverage : bool
            Whether to decompose the time the lattice did not reach. The decomposition
            costs orders of magnitude more than the lattice itself -- on O3a, 374 s and
            11.8 GB against 0.06 s -- so a caller that only needs window counts, such as
            a slide ladder measuring its own livetime, passes ``False`` and finds
            :attr:`coverage` set to ``None``. The lattice is identical either way.
        hostable_by_detector : dict, optional
            Precomputed :func:`hostable_intervals` per detector. They depend on the
            segments alone and not on the slide, so a ladder building many lattices over
            one network computes them once and passes them in.
        """
        if not segments_by_detector:
            raise ValueError("no detectors given")
        detectors = list(segments_by_detector)
        reference = reference_detector or detectors[0]
        if reference not in segments_by_detector:
            raise ValueError(
                f"reference detector {reference!r} is not in the network {detectors}"
            )
        offsets_s = dict(offsets_s or {})
        if offsets_s.get(reference, 0.0) != 0.0:
            raise ValueError(
                f"the reference detector {reference!r} carries a non-zero slide offset "
                f"({offsets_s[reference]}); slides are measured relative to it"
            )

        # A window must fit inside ONE chunk of EVERY detector. Detectors have their
        # chunk boundaries at unrelated times, so a start can be hostable in the
        # reference and not in a follower; restricting on data presence alone admits
        # windows a follower cannot supply. Each follower's hostable set is pulled back
        # through its slide offset into the reference frame before intersecting.
        precomputed = dict(hostable_by_detector or {})
        restriction = list(coincident)
        for detector, segments in segments_by_detector.items():
            hostable = precomputed.get(detector)
            if hostable is None:
                hostable = hostable_intervals(segments, geometry.window_samples)
            shift = -offsets_s.get(detector, 0.0)
            restriction = intersect_intervals(restriction, hostable, shift_b_s=shift)

        spans, report = window_hosts(
            segments_by_detector[reference],
            geometry.window_samples,
            geometry.stride_samples,
            restrict_to=restriction,
            coverage=coverage,
        )
        ordered = {
            det: sort_by_gps(segs) for det, segs in segments_by_detector.items()
        }
        return cls(
            geometry=geometry,
            spans_by_detector={reference: spans},
            slide_id=slide_id,
            offsets_s=offsets_s,
            window_shift=window_shift,
            reference_detector=reference,
            segments_by_detector=ordered,
            coverage=report,
        )

    @property
    def detectors(self) -> Tuple[str, ...]:
        """Network order, reference first."""
        rest = [d for d in self.segments_by_detector if d != self.reference_detector]
        return (self.reference_detector, *rest)

    @property
    def reference_spans(self) -> List[HostSpan]:
        """The spans carrying the lattice."""
        return self.spans_by_detector[self.reference_detector]

    def __len__(self) -> int:
        """Total number of windows."""
        return sum(span.n_windows for span in self.reference_spans)

    @property
    def livetime_s(self) -> float:
        """``len(self) * geometry.stride_s``."""
        return len(self) * self.geometry.stride_s

    def blocks(self, block_seconds: float) -> List[Block]:
        """
        Partition the lattice into work blocks.

        A block is a whole number of spans, so no block boundary ever falls inside a run
        of windows and the reader never has to resume mid-segment. Block length therefore
        approximates ``block_seconds`` rather than matching it.
        """
        if block_seconds <= 0:
            raise ValueError(f"block_seconds must be positive, got {block_seconds}")
        spans = self.reference_spans
        stride_s = self.geometry.stride_s
        blocks: List[Block] = []
        start_index = 0
        accumulated = 0.0
        for index, span in enumerate(spans):
            accumulated += span.n_windows * stride_s
            is_last = index == len(spans) - 1
            if accumulated >= block_seconds or is_last:
                first, last = spans[start_index], spans[index]
                blocks.append(
                    Block(
                        block_id=len(blocks),
                        gps_start=float(first.starts_gps()[0]),
                        gps_end=float(last.starts_gps()[-1]) + stride_s,
                        span_slice=(start_index, index + 1),
                    )
                )
                start_index = index + 1
                accumulated = 0.0
        return blocks

    def iter_block(self, block: Block) -> Iterator[Tuple[Segment, int, int]]:
        """Yield ``(segment, first_local, n_windows)`` runs, never crossing a segment."""
        for span in self.reference_spans[block.span_slice[0] : block.span_slice[1]]:
            yield span.segment, span.first_local, span.n_windows

    def iter_block_detector(
        self, block: Block, detector: str
    ) -> Iterator[DetectorRun]:
        """
        Yield the runs one detector contributes to a block.

        For the reference detector these are its own spans. For any other, each reference
        run is mapped through the slide offset onto that detector's segments and split
        wherever it would leave one, so a run never crosses a boundary.
        """
        if detector not in self.segments_by_detector:
            raise ValueError(f"unknown detector {detector!r}")
        stride = self.geometry.stride_samples
        if detector == self.reference_detector:
            for span in self.reference_spans[block.span_slice[0] : block.span_slice[1]]:
                yield DetectorRun(
                    detector=detector,
                    segment=span.segment,
                    first_local=span.first_local,
                    n_windows=span.n_windows,
                    stride_samples=stride,
                    residual_samples=0.0,
                )
            return

        offset = self.offsets_s.get(detector, 0.0) if self.offsets_s else 0.0
        shift = (self.window_shift or {}).get(detector)
        window_s = self.geometry.window_s
        stride_s = self.geometry.stride_s
        candidates = self.segments_by_detector[detector]
        lattice = self._lattice_starts() if shift else None
        cursor_ordinal = self._span_ordinals()[block.span_slice[0]] if shift else 0
        for span in self.reference_spans[block.span_slice[0] : block.span_slice[1]]:
            if shift:
                # Paired by position in the analysed lattice, not by time. The ordinals
                # are contiguous, but their *times* are not: the lattice skips every gap
                # between segments, so a rolled span lands wherever it lands and steps
                # by one stride only until it crosses one of those gaps.
                ordinals = (
                    cursor_ordinal + np.arange(span.n_windows, dtype=np.int64) + shift
                ) % lattice.size
                starts = lattice[ordinals]
                cursor_ordinal += span.n_windows
                # A run marches from its first sample by a fixed stride, so it may only
                # cover targets that really are one stride apart. Without this the run
                # keeps marching past a jump and reads a stretch of strain that no
                # window was ever assigned -- silently, because every index stays inside
                # the segment.
                steps = np.diff(starts)
                tolerance = 0.5 / self.geometry.sample_rate
                stops = list(
                    np.flatnonzero(np.abs(steps - stride_s) > tolerance) + 1
                )
            else:
                starts = span.starts_gps() + offset
                stops = []
            stops.append(starts.size)
            cursor = 0
            stop_index = 0
            while cursor < starts.size:
                while stops[stop_index] <= cursor:
                    stop_index += 1
                limit = stops[stop_index]
                target = float(starts[cursor])
                segment = _owning_segment(candidates, target, window_s)
                if segment is None:
                    # No data in this detector at this time under this slide. The
                    # coincidence should have excluded it, so this is worth failing on.
                    raise ValueError(
                        f"{detector} has no segment covering {target} + {window_s} s "
                        f"under slide {self.slide_id}; the coincident intervals and the "
                        "segment list disagree"
                    )
                exact = (target - segment.gps_start) * segment.sample_rate
                first_local = int(round(exact))
                residual = abs(exact - first_local)
                # Rounding may land just outside at a segment edge, which is where the
                # half-sample tolerance in _owning_segment admitted it.
                first_local = min(
                    max(first_local, 0),
                    segment.nsamples - self.geometry.window_samples,
                )
                # How many further windows stay inside this segment, and inside this
                # contiguous chunk of targets.
                room = (segment.nsamples - self.geometry.window_samples) - first_local
                n_here = min(limit - cursor, room // stride + 1) if room >= 0 else 0
                if n_here <= 0:
                    raise ValueError(
                        f"{detector} segment {segment.segment_index} cannot host a "
                        f"window at {target}"
                    )
                yield DetectorRun(
                    detector=detector,
                    segment=segment,
                    first_local=first_local,
                    n_windows=int(n_here),
                    stride_samples=stride,
                    residual_samples=residual,
                )
                cursor += int(n_here)

    def _lattice_starts(self) -> np.ndarray:
        """
        Every reference window start in this lattice, in order.

        Built once and cached: a roll needs to look up an ordinal anywhere in the run,
        and rebuilding the array per block would sweep the whole lattice per block. On
        the O3a lattice it is 100 MB of float64 against ~12.5 million windows.
        """
        if getattr(self, "_lattice_cache", None) is None:
            starts = (
                np.concatenate([span.starts_gps() for span in self.reference_spans])
                if self.reference_spans
                else np.zeros(0, dtype=np.float64)
            )
            setattr(self, "_lattice_cache", starts)
        return self._lattice_cache

    def _span_ordinals(self) -> np.ndarray:
        """Global ordinal of each span's first window; cached with the lattice."""
        if getattr(self, "_ordinal_cache", None) is None:
            counts = [span.n_windows for span in self.reference_spans]
            setattr(
                self,
                "_ordinal_cache",
                np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            )
        return self._ordinal_cache

    def alignment_residuals(self, block: Block) -> dict:
        """
        Worst sub-sample misalignment per detector across a block, in samples.

        Bounded by half a sample by construction. Reported so a timing systematic is a
        number in the provenance rather than an assumption.
        """
        out = {}
        for detector in self.detectors:
            worst = 0.0
            for run in self.iter_block_detector(block, detector):
                worst = max(worst, run.residual_samples)
            out[detector] = worst
        return out

    def gps(self, block: Block) -> np.ndarray:
        """Reference-frame window-start GPS times for a block."""
        pieces = [
            span.starts_gps()
            for span in self.reference_spans[block.span_slice[0] : block.span_slice[1]]
        ]
        if not pieces:
            return np.empty(0, dtype=float)
        return np.concatenate(pieces)


def _owning_segment(
    candidates: Sequence[Segment], gps: float, window_s: float
) -> Optional[Segment]:
    """
    First segment in time order that can hold a whole window starting at ``gps``.

    Admits a window sitting up to half a sample outside the segment. A follower
    detector's segment starts at a multiple of half a sample, so a reference window
    landing exactly on a chunk boundary can fall a fraction before the follower's first
    sample; rounding puts it on the first sample, which is the intended behaviour, and
    refusing it would drop a window the reference kept and misalign the batch.
    """
    for segment in candidates:
        tolerance = 0.5 / segment.sample_rate
        if (
            segment.gps_start - tolerance <= gps
            and gps + window_s <= segment.gps_end + tolerance
        ):
            return segment
    return None


def run(spec, **kwargs) -> dict:
    """
    Stage driver: build the zero-lag window lattice and record what it covers.

    Nothing is persisted. The lattice is a deterministic function of the geometry and the
    segment sidecars -- 12.5 M windows in 5.5 s on the real O3a release -- so every
    downstream stage rebuilds it rather than reading a copy that could disagree with the
    segments it was derived from. What is recorded is the accounting: how many windows,
    how much analysed time, and how many blocks the campaign will be cut into.

    ``n_windows * stride_s`` is the analysed livetime by identity, and it is strictly less
    than the coincident livetime because a window needs a whole window of contiguous data:
    the deficit is the boundary loss, and reporting the coincident time instead would
    credit the search with moments it could not have triggered on.
    """
    from sage.search.segments import coincident_intervals, load_segments

    geometry = spec.geometry_object()
    segments = {
        detector: load_segments(
            Path(spec.data.release_dir)
            / f"data_{detector}_{spec.data.observing_run}_segments.json"
        )
        for detector in spec.data.detectors
    }
    coincident = coincident_intervals(segments)
    grid = AnalysisGrid.build(geometry, segments, coincident)
    blocks = grid.blocks(float(spec.engine.block_seconds))
    coincident_s = float(sum(hi - lo for lo, hi in coincident))
    analysed_s = float(grid.livetime_s)
    return {
        "n_windows": int(len(grid)),
        "n_blocks": len(blocks),
        "analysed_livetime_s": analysed_s,
        "coincident_livetime_s": coincident_s,
        "boundary_loss_s": coincident_s - analysed_s,
        "stride_s": float(geometry.stride_s),
        "segments_by_detector": {d: len(v) for d, v in segments.items()},
        # The lattice is rebuilt, never read back, so the fingerprint is what a rebuild
        # would have to reproduce for downstream products to still describe this data --
        # and that is the window starts themselves, not a count of them. Counts collide:
        # a lattice shifted by one sample, or one whose windows moved between two
        # detectors' segments, holds the same number of windows over the same livetime
        # while every window it scores is a different stretch of strain.
        "fingerprint": combine(
            len(grid),
            f"{analysed_s:.6f}",
            len(blocks),
            geometry.stride_samples,
            digest_values(
                {
                    "starts_local": {
                        detector: np.concatenate(
                            [span.starts_local() for span in spans]
                        )
                        if spans
                        else np.zeros(0, dtype=np.int64)
                        for detector, spans in grid.spans_by_detector.items()
                    },
                    "segment_index": {
                        detector: np.array(
                            [span.segment.segment_index for span in spans],
                            dtype=np.int64,
                        )
                        for detector, spans in grid.spans_by_detector.items()
                    },
                    "span_windows": {
                        detector: np.array(
                            [span.n_windows for span in spans], dtype=np.int64
                        )
                        for detector, spans in grid.spans_by_detector.items()
                    },
                    "block_slices": [list(block.span_slice) for block in blocks],
                    "stride_samples": int(geometry.stride_samples),
                    "window_samples": int(geometry.window_samples),
                }
            ),
        ),
    }

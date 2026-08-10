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

Reads are clamped to a single segment. Sample index ``n+1`` at a segment end belongs
to a different chunk whose GPS start is ~496 s away, so a read spanning the boundary
splices two unrelated epochs. Consecutive windows overlap by ``window - stride``
samples, so blocks are read once and expanded with ``unfold`` rather than gathered
per window.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid, Block
from sage.search.segments import Segment


@dataclass
class WindowBatch:
    """One batch of raw, dyn-range-corrected strain windows."""

    strain: "np.ndarray"
    gps: np.ndarray
    segment_index: np.ndarray
    local_start: np.ndarray
    slide_id: int

    def __len__(self) -> int:
        """Number of windows in the batch."""
        raise NotImplementedError


class StreamingStrainReader:
    """
    Iterate a run's window lattice in segment order, one block at a time.

    Parameters
    ----------
    release_dir : path
        Directory holding ``data_{det}_{run}.bin`` and its sidecars.
    grid : AnalysisGrid
        The lattice to walk, including any slide offsets.
    batch_size : int
        Upper bound only; the effective batch is clamped to the windows remaining
        in the current owning segment.
    """

    def __init__(
        self,
        release_dir: str | Path,
        grid: AnalysisGrid,
        geometry: SearchGeometry,
        batch_size: int = 8192,
        prefetch: int = 2,
        pin_memory: bool = True,
    ) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[WindowBatch]:
        """Yield batches in lattice order."""
        raise NotImplementedError

    def iter_block(self, block: Block) -> Iterator[WindowBatch]:
        """Yield batches for a single block."""
        raise NotImplementedError

    def seek(self, block_id: int) -> None:
        """Resume at a block boundary."""
        raise NotImplementedError

    def close(self) -> None:
        """Release memmaps and the prefetch thread."""
        raise NotImplementedError


def read_segment_span(
    mmap: np.ndarray, segment: Segment, first_local: int, n_samples: int
) -> np.ndarray:
    """Read ``n_samples`` from one segment, dividing out ``dyn_range_fac``."""
    raise NotImplementedError


def unfold_windows(block: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    """Expand a contiguous block into overlapping windows without copying per window."""
    raise NotImplementedError

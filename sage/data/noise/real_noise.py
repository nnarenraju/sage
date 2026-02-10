#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : real_noise.py
Description   : Short description of the file

Created on 2026-01-19 14:23:24

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import os
import h5py
import json
import torch
import numpy as np

from pathlib import Path
from typing import Dict, List, Union, Optional
from pycbc import DYN_RANGE_FAC

import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


class HDF5SingleNoiseSampler:
    """
    Duration-weighted random noise sampler from GW noise datasets.

    Supports:
    - single monolithic HDF5 file
    - directory of monolithic HDF5 files

    Sampling:
    - segment chosen ∝ usable duration
    - random contiguous slice returned
    """

    def __init__(self, source: Union[str, Path]):
        """
        Args:
            source:
                - path to monolithic HDF5 file
                - OR directory containing multiple monolithic files
        """
        self.files: List[h5py.File] = []
        self.segments = []  # list of h5py.Dataset
        self.seg_lengths = []  # length in samples

        source = Path(source)

        if source.is_dir():
            paths = sorted(source.glob("*.h5"))
            if not paths:
                raise ValueError(f"No HDF5 files found in {source}")
        else:
            paths = [source]

        # Open files and collect segments
        for p in paths:
            hf = h5py.File(p, "r")
            self.files.append(hf)

            seg_grp = hf["segments"]
            for key in sorted(seg_grp.keys()):
                dset = seg_grp[key]
                self.segments.append(dset)
                self.seg_lengths.append(dset.shape[0])

        self.seg_lengths = np.asarray(self.seg_lengths, dtype=np.int64)

        # Cache of segment probabilities given requested_nsamples
        # Allows for different lengths to be used
        self._prob_cache = {}

        if len(self.segments) == 0:
            raise RuntimeError("No segments found.")

    def _segment_probabilities(self, requested_nsamples: int) -> np.ndarray:
        """
        Compute probabilities ∝ usable duration for each segment.
        """
        if requested_nsamples in self._prob_cache:
            return self._prob_cache[requested_nsamples]

        usable = self.seg_lengths - requested_nsamples
        usable[usable < 0] = 0

        total = usable.sum()
        if total == 0:
            raise ValueError("Requested sample length exceeds all available segments.")

        probs = usable / total
        self._prob_cache[requested_nsamples] = probs
        return probs

    @staticmethod
    def _pick_start(seg_len: int, nsamples: int) -> int:
        max_start = seg_len - nsamples
        return np.random.randint(0, max_start + 1)

    def sample(self, requested_nsamples: int) -> np.ndarray:
        """
        Draw a random noise slice.

        Args:
            requested_nsamples (int):
                Total samples required (already includes corruption padding)

        Returns:
            np.ndarray
        """
        probs = self._segment_probabilities(requested_nsamples)

        while True:
            idx = np.random.choice(len(self.segments), p=probs)
            dset = self.segments[idx]
            seg_len = self.seg_lengths[idx]

            start = self._pick_start(seg_len, requested_nsamples)
            noise = np.asarray(
                dset[start : start + requested_nsamples],
                dtype=np.float32,
            )

            noise /= DYN_RANGE_FAC

            if not np.any(np.isnan(noise)):
                return noise

    def close(self):
        for f in self.files:
            f.close()

    def __call__(self, nsamples: int, **kwargs):
        return self.sample(nsamples)


class MemmapSingleNoiseSampler:
    """
    Duration-weighted random noise sampler from GW noise datasets.

    Supports:
    - single monolithic .bin file with sidecar *_segments.json

    Sampling:
    - segment chosen ∝ usable duration
    - random contiguous slice returned
    """

    def __init__(
        self, source: Union[str, Path], return_tensor=False, tensor_dtype=torch.float32
    ):
        """
        Args:
            source:
                - path to monolithic .bin file
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)

        self.bin_file = source

        # We can return tensors if downstream ops rely on torch
        self.return_tensor = return_tensor
        self.tensor_dtype = tensor_dtype

        meta_path = source.parent / f"{source.stem}_segments.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if len(meta) == 0:
            raise RuntimeError("No segments found in metadata.")

        # dtype handling
        dt = np.dtype(meta[0]["dtype"]).newbyteorder(meta[0]["endianness"])
        self.dtype = dt

        # Open memmap
        self.mm = np.memmap(self.bin_file, dtype=dt, mode="r")

        # Build segment table
        self.segments = np.array(
            [
                (
                    seg["sample_start_idx"],
                    seg["nsamples"],
                )
                for seg in meta
            ],
            dtype=[
                ("start", "i8"),
                ("nsamples", "i8"),
            ],
        )

        self.seg_lengths = self.segments["nsamples"].astype(np.int64)

        # Cache of probabilities per requested_nsamples
        self._prob_cache = {}

        if len(self.segments) == 0:
            raise RuntimeError("No segments found.")

    def _segment_probabilities(self, requested_nsamples: int) -> np.ndarray:
        """
        Compute probabilities ∝ usable duration for each segment.
        """
        if requested_nsamples in self._prob_cache:
            return self._prob_cache[requested_nsamples]

        usable = self.seg_lengths - requested_nsamples
        usable[usable < 0] = 0

        total = usable.sum()
        if total == 0:
            raise ValueError("Requested sample length exceeds all available segments.")

        probs = usable / total
        self._prob_cache[requested_nsamples] = probs
        return probs

    @staticmethod
    def _pick_start(seg_len: int, nsamples: int, rng: np.random.Generator) -> int:
        max_start = seg_len - nsamples
        return rng.integers(0, max_start + 1)

    def sample(
        self,
        requested_nsamples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Draw a random noise slice.

        Args:
            requested_nsamples (int):
                Total samples required (already includes corruption padding)

        Returns:
            np.ndarray of shape (requested_nsamples,)
        """
        rng = rng or np.random.default_rng()
        probs = self._segment_probabilities(requested_nsamples)

        while True:
            idx = rng.choice(len(self.segments), p=probs)
            seg = self.segments[idx]
            seg_len = seg["nsamples"]

            start_offset = self._pick_start(seg_len, requested_nsamples, rng)
            start = seg["start"] + start_offset

            noise = np.asarray(
                self.mm[start : start + requested_nsamples],
                dtype=np.float32,
                copy=True,
            )

            # Undo dynamic range scaling
            noise /= DYN_RANGE_FAC

            if not np.any(np.isnan(noise)):
                if self.return_tensor:
                    return torch.tensor(noise, dtype=self.tensor_dtype)
                else:
                    return noise

    def close(self):
        """Release memmap"""
        del self.mm

    def __call__(self, nsamples: int, **kwargs):
        return self.sample(nsamples)


class MemmapNoiseSampler:
    """
    GPU batch sampler for monolithic .bin files with async prefetch.

    Features:
    - Multiple detectors/files
    - Weighted random sampling by segment duration
    - Precompute random starts per batch
    - Prefetch N batches to GPU asynchronously
    """

    def __init__(
        self,
        bin_files: List[Path],
        seq_len: int,
        device: str = "cuda",
        batch_size: int = 64,
        prefetch: int = 2,
        postprocess_fn=None,
    ):
        self.seq_len = seq_len
        self.device = device
        self.prefetch = prefetch
        self.bin_files = [Path(f) for f in bin_files]
        self.n_detectors = len(bin_files)
        self._batch_size = batch_size
        self.postprocess_fn = postprocess_fn

        self.mmaps = []
        self.seg_index = []
        self.segment_probs = []
        self.dtypes = []

        # Load metadata and memmaps
        for p in self.bin_files:
            meta_path = p.parent / f"{p.stem}_segments.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata {meta_path} not found")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            dtype = np.dtype(meta[0]["dtype"]).newbyteorder(meta[0]["endianness"])
            self.dtypes.append(dtype)

            mm = np.memmap(p, dtype=dtype, mode="r")
            self.mmaps.append(mm)

            seg_idx_arr = np.array(
                [
                    (
                        seg["segment_index"],
                        seg["sample_start_idx"],
                        seg["sample_start_idx"] + seg["nsamples"],
                        seg["nsamples"],
                    )
                    for seg in meta
                ],
                dtype=[
                    ("idx", "i4"),
                    ("start", "i8"),
                    ("end", "i8"),
                    ("nsamples", "i8"),
                ],
            )
            self.seg_index.append(seg_idx_arr)

            usable = seg_idx_arr["nsamples"] - self.seq_len
            usable[usable < 0] = 0
            total = usable.sum()
            if total == 0:
                raise ValueError("seq_len exceeds all segments")
            probs = usable / total
            self.segment_probs.append(probs)

        self.rng = np.random.default_rng()

        # Prefetch queue
        self.queue = Queue(maxsize=self.prefetch)
        self._stop_event = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )
        self._prefetch_thread.start()

    def _sample_starts_batch(self, batch_size: int):
        start_indices = []
        segment_indices = []

        for d in range(self.n_detectors):
            seg_idx = self.seg_index[d]
            probs = self.segment_probs[d]
            chosen_segments = self.rng.choice(len(seg_idx), size=batch_size, p=probs)

            starts = np.empty(batch_size, dtype=np.int64)
            seg_ids = np.empty(batch_size, dtype=np.int32)
            for i, seg_i in enumerate(chosen_segments):
                seg = seg_idx[seg_i]
                max_offset = seg["nsamples"] - self.seq_len
                offset = self.rng.integers(0, max_offset + 1) if max_offset > 0 else 0
                starts[i] = seg["start"] + offset
                seg_ids[i] = seg["idx"]

            start_indices.append(starts)
            segment_indices.append(seg_ids)

        return start_indices, segment_indices

    def _read_batch(self, batch_size: int):
        B = batch_size
        D = self.n_detectors
        seq_len = self.seq_len

        start_indices, segment_indices = self._sample_starts_batch(B)
        batch_tensor = torch.empty(
            (B, D, seq_len), dtype=torch.float32, device=self.device
        )

        def read_detector(d):
            mm = self.mmaps[d]
            starts = start_indices[d]
            arr = np.empty((B, seq_len), dtype=np.float32)

            for i, s in enumerate(starts):
                arr[i] = mm[s : s + seq_len]

            # Get the original scale back
            arr /= DYN_RANGE_FAC

            return arr

        with ThreadPoolExecutor(max_workers=D) as executor:
            results = list(executor.map(read_detector, range(D)))

        for d, arr in enumerate(results):
            cpu_tensor = torch.from_numpy(arr).pin_memory()
            batch_tensor[:, d, :].copy_(cpu_tensor, non_blocking=True)

        # convert segment indices to a GPU tensor
        segment_ids = torch.empty((B, D), dtype=torch.int32, device=self.device)

        for d in range(D):
            segment_ids[:, d].copy_(
                torch.from_numpy(segment_indices[d]).to(self.device),
                non_blocking=True,
            )

        if self.postprocess_fn is not None:
            batch_tensor = self.postprocess_fn(batch_tensor, segment_ids)
        else:
            # default: TD to FD only
            batch_tensor = torch.fft.rfft(batch_tensor, dim=-1)

        return batch_tensor

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            if not self.queue.full():
                batch_tensor = self._read_batch(self._batch_size)
                self.queue.put(batch_tensor)
            else:
                # sleep briefly to yield CPU
                self._stop_event.wait(0.01)

    def sample_batch(self, batch_size: int):
        """
        Return a GPU batch. Starts async prefetching if first call.
        """
        self._batch_size = batch_size
        # If queue has a ready batch, return it
        batch_tensor = self.queue.get()
        return batch_tensor

    def shutdown(self):
        """Stop prefetch thread"""
        self._stop_event.set()
        self._prefetch_thread.join()
        for mm in self.mmaps:
            del mm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real detector noise samplers: random windows from GWOSC strain stored
as HDF5 or flat ``.bin`` files, with a prefetching batch sampler.

Filename      : real_noise.py
Description   : Short description of the file

Created on 2026-01-19 14:23:24

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

from __future__ import annotations

import os
import h5py
import json
import torch
import numpy as np
import torch.nn as nn

from pathlib import Path
from typing import Dict, List, Union, Optional

import atexit
import weakref
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from ._pycbc_lazy import dyn_range_fac
from .data_quality import clean
from .selection import select_records, summary

from sage.config.machine import segments_file
from sage.core.geometry import Window
from sage.core.logger import get_logger

logger = get_logger(__name__)


class HDF5SingleNoiseSampler:
    """
    Random noise windows from monolithic HDF5 strain files.

    A segment is drawn with probability proportional to its number of
    valid window starts, then a uniformly random window within it.

    Parameters
    ----------
    source : str or Path
        An HDF5 file, or a directory of ``*.h5`` files, each holding
        its segments under the ``segments`` group.

    Raises
    ------
    ValueError
        If ``source`` is a directory without HDF5 files.
    RuntimeError
        If no segments are found.
    """

    def __init__(self, source: Union[str, Path]):
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

        # Segment probabilities per requested window length
        self._prob_cache = {}

        if len(self.segments) == 0:
            raise RuntimeError("No segments found.")

    def _segment_probabilities(self, requested_nsamples: int) -> np.ndarray:
        """
        Probability of each segment, proportional to its valid starts.

        Parameters
        ----------
        requested_nsamples : int
            Window length in samples.

        Returns
        -------
        numpy.ndarray
            Probability per segment.

        Raises
        ------
        ValueError
            If the window is longer than every segment.
        """
        if requested_nsamples in self._prob_cache:
            return self._prob_cache[requested_nsamples]

        # Valid starts: nsamples - window + 1; shorter segments get 0
        usable = self.seg_lengths - requested_nsamples + 1
        usable[usable < 0] = 0

        total = usable.sum()
        if total == 0:
            raise ValueError("Requested sample length exceeds all available segments.")

        probs = usable / total
        self._prob_cache[requested_nsamples] = probs
        return probs

    @staticmethod
    def _pick_start(seg_len: int, nsamples: int) -> int:
        """Uniform start offset of a window in a segment."""
        max_start = seg_len - nsamples
        return np.random.randint(0, max_start + 1)

    def sample(self, requested_nsamples: int) -> np.ndarray:
        """
        Draw a random noise window from the global NumPy RNG.

        Windows holding NaN are redrawn.

        Parameters
        ----------
        requested_nsamples : int
            Window length in samples, padding included.

        Returns
        -------
        numpy.ndarray, shape ``(requested_nsamples,)``
            Strain in float64, divided by ``DYN_RANGE_FAC``.
        """
        probs = self._segment_probabilities(requested_nsamples)

        while True:
            idx = np.random.choice(len(self.segments), p=probs)
            dset = self.segments[idx]
            seg_len = self.seg_lengths[idx]

            start = self._pick_start(seg_len, requested_nsamples)
            noise = np.asarray(
                dset[start : start + requested_nsamples],
                dtype=np.float64,
            )

            noise /= dyn_range_fac()

            if not np.any(np.isnan(noise)):
                return noise

    def close(self):
        """Close all open HDF5 file handles."""
        for f in self.files:
            f.close()

    def __call__(self, nsamples: int, **kwargs):
        return self.sample(nsamples)


class MemmapSingleNoiseSampler:
    """
    Random noise windows from one monolithic ``.bin`` strain file.

    A segment from the ``*_segments.json`` sidecar is drawn with
    probability proportional to its number of valid window starts, then
    a uniformly random window within it.

    Parameters
    ----------
    source : str or Path
        The ``.bin`` strain file.
    bad : list
        Bad intervals, from
        :func:`~sage.data.noise.data_quality.read_bad_time`; segments
        overlapping them are never drawn.
    return_tensor : bool
        Return torch tensors instead of NumPy arrays.
    tensor_dtype : torch.dtype
        Dtype of the returned tensors.

    Raises
    ------
    FileNotFoundError
        If the strain file or its sidecar is missing.
    RuntimeError
        If the sidecar lists no segments, or all overlap bad time.
    """

    def __init__(
        self, source: Union[str, Path], bad: list, return_tensor=False, tensor_dtype=torch.float32
    ):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)

        self.bin_file = source

        self.return_tensor = return_tensor
        self.tensor_dtype = tensor_dtype

        meta_path = segments_file(source)
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if len(meta) == 0:
            raise RuntimeError("No segments found in metadata.")
        meta = [record for record, keep in zip(meta, clean(meta, bad)) if keep]
        if len(meta) == 0:
            raise RuntimeError(f"Every segment of {source} overlaps bad time.")

        dt = np.dtype(meta[0]["dtype"]).newbyteorder(meta[0]["endianness"])
        self.dtype = dt

        self.mm = np.memmap(self.bin_file, dtype=dt, mode="r")

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

        # Segment probabilities per requested window length
        self._prob_cache = {}

        if len(self.segments) == 0:
            raise RuntimeError("No segments found.")

    def _segment_probabilities(self, requested_nsamples: int) -> np.ndarray:
        """
        Probability of each segment, proportional to its valid starts.

        Parameters
        ----------
        requested_nsamples : int
            Window length in samples.

        Returns
        -------
        numpy.ndarray
            Probability per segment.

        Raises
        ------
        ValueError
            If the window is longer than every segment.
        """
        if requested_nsamples in self._prob_cache:
            return self._prob_cache[requested_nsamples]

        # Valid starts: nsamples - window + 1; shorter segments get 0
        usable = self.seg_lengths - requested_nsamples + 1
        usable[usable < 0] = 0

        total = usable.sum()
        if total == 0:
            raise ValueError("Requested sample length exceeds all available segments.")

        probs = usable / total
        self._prob_cache[requested_nsamples] = probs
        return probs

    @staticmethod
    def _pick_start(seg_len: int, nsamples: int, rng: np.random.Generator) -> int:
        """Uniform start offset of a window in a segment."""
        max_start = seg_len - nsamples
        return rng.integers(0, max_start + 1)

    def sample(
        self,
        requested_nsamples: int,
        rng = None,
    ) -> np.ndarray:
        """
        Draw a random noise window.

        Windows holding NaN are redrawn.

        Parameters
        ----------
        requested_nsamples : int
            Window length in samples, padding included.
        rng : numpy.random.Generator or None
            Random generator; ``None`` uses a new unseeded generator.

        Returns
        -------
        numpy.ndarray or torch.Tensor, shape ``(requested_nsamples,)``
            Strain in float32 divided by ``DYN_RANGE_FAC``; a tensor of
            ``tensor_dtype`` if ``return_tensor``.
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

            # Undo the dynamic-range scaling of the stored strain
            noise /= dyn_range_fac()

            if not np.any(np.isnan(noise)):
                if self.return_tensor:
                    return torch.tensor(noise, dtype=self.tensor_dtype)
                else:
                    return noise

    def close(self):
        """Delete the memmap."""
        del self.mm

    def __call__(self, nsamples: int, **kwargs):
        return self.sample(nsamples)


class MemmapNoiseSampler(torch.nn.Module):
    """
    Batch sampler of real noise windows with a prefetching thread.

    Windows are drawn per detector from pooled ``.bin`` strain files,
    segments weighted by their valid starts; a daemon thread fills a
    queue of FD batches, consumed by :meth:`sample_batch`. Optionally a
    batch replays mined hard windows (:meth:`set_hard_bank`).

    Parameters
    ----------
    runs : list of dict
        Observing runs to pool, each ``{"run", "data_dir", "bins"}``:
        the run name, its ASD directory, and one strain file per
        detector in channel order; optional ``portion`` or
        ``gps_range`` restrict the segments drawn from
        (:mod:`sage.data.noise.selection`).
    detectors : list of str
        Detectors, in channel order.
    window : Window
        Window geometry; sets the window length.
    batch_size : int
        Windows per batch.
    bad_segments : dict
        GWOSC bad time keyed by ``(observing run, detector)``, from
        :func:`sage.data.noise.data_quality.noise_bad_segments`;
        segments overlapping it are never drawn.
    seed : int
        Seed for the internal NumPy RNG.
    prefetch_batches : int
        Maximum number of batches held in the prefetch queue.
    read_workers : int
        Threads reading windows from disk.
    hard_start_jitter_s : float
        Replayed hard windows are shifted by up to this many seconds.
    hard_focal_gamma : float
        Focal weighting exponent of hard-window replay; 0 is uniform.
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Dtype of the noise targets.
    postprocess_fn : callable or None
        Applied to ``(batch_td, segment_ids, run_ids)`` to convert from
        TD to FD, typically :class:`RecolourPostprocess`. If ``None``,
        a plain ``torch.fft.rfft`` with ``norm="forward"`` is used.

    Raises
    ------
    FileNotFoundError
        If a strain file's sidecar is missing.
    ValueError
        If a sidecar's detector or observing run does not match, or the
        window is longer than every usable segment of a detector.

    Attributes
    ----------
    GRAPH_READY : bool
        ``False``: the prefetch queue pop cannot be traced by
        ``torch.compile``.
    """

    GRAPH_READY = False

    def __init__(
        self,
        runs: list,
        detectors: list,
        window: Window,
        batch_size: int,
        bad_segments: dict,
        seed: int,
        prefetch_batches: int,
        read_workers: int,
        hard_start_jitter_s: float,
        hard_focal_gamma: float,
        device,
        dtype: torch.dtype,
        postprocess_fn=None,
    ):
        super().__init__()

        self.seq_len = window.padded_length_in_samples
        self.sample_rate = float(window.sample_rate_in_hz)
        # Uniform jitter of replayed hard starts, within their segment
        self._hard_jitter_s = float(hard_start_jitter_s)
        self._seg_bounds_lut = None            # built on first jitter
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.detectors = list(detectors)
        self.n_detectors = len(detectors)
        self.batch_size = batch_size
        self.postprocess_fn = postprocess_fn

        # A window's run id selects the run's memmap and ASDs
        self.run_names = [r["run"] for r in runs]
        self.run_data_dirs = [r["data_dir"] for r in runs]
        self.n_runs = len(runs)

        self.mmaps = [[] for _ in range(self.n_detectors)]      # [d][run] -> memmap
        self.seg_index = [None] * self.n_detectors              # [d] -> pooled struct
        self.segment_probs = [None] * self.n_detectors          # [d] -> pooled probs
        self.dtypes = [[] for _ in range(self.n_detectors)]     # [d][run] -> dtype
        self.bin_files = []                                     # flat, for logging

        self.noise_target = torch.zeros((self.batch_size, 1)).to(
            dtype=dtype, device=device
        )

        # All detectors' records are read first: a portion cut is shared
        # across a run's detectors
        records = []                                            # [run][d] -> records
        for r in runs:
            by_detector = {}
            for d, detector in enumerate(self.detectors):
                p = Path(r["bins"][d])
                meta_path = segments_file(p)
                if not meta_path.exists():
                    raise FileNotFoundError(
                        f"noise run {r['run']!r} provides no data for detector "
                        f"{detector!r}: expected {meta_path}"
                    )
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                # Files bind to channels by position, so the sidecar's
                # detector and observing run must match
                meta_det = str(meta[0].get("detector", "")).upper()
                if meta_det != str(detector).upper():
                    raise ValueError(
                        f"noise run {r['run']!r}: {p.name} holds detector {meta_det!r}, "
                        f"channel {d} is {detector}"
                    )
                meta_run = str(meta[0].get("observing_run", "")).upper()
                if meta_run != str(r["run"]).upper():
                    raise ValueError(
                        f"noise run {r['run']!r}: {p.name} is from observing run "
                        f"{meta_run!r}"
                    )
                by_detector[detector] = meta
            selected = select_records(by_detector, r)
            for detector in self.detectors:
                logger.info("noise %s %s: %s", r["run"], detector, summary(selected[detector]))
            records.append([selected[detector] for detector in self.detectors])

        # The run id keeps segment indices of pooled runs distinct
        _seg_dtype = [("run", "i8"), ("idx", "i8"), ("start", "i8"),
                      ("end", "i8"), ("nsamples", "i8")]
        for d in range(self.n_detectors):
            seg_tables, usable_parts = [], []
            for run_id, r in enumerate(runs):
                p = Path(r["bins"][d])
                self.bin_files.append(p)
                meta = records[run_id][d]
                dtype = np.dtype(meta[0]["dtype"]).newbyteorder(meta[0]["endianness"])
                self.dtypes[d].append(dtype)
                self.mmaps[d].append(np.memmap(p, dtype=dtype, mode="r"))
                seg_arr = np.array(
                    [
                        (
                            run_id,
                            seg["segment_index"],
                            seg["sample_start_idx"],
                            seg["sample_start_idx"] + seg["nsamples"],
                            seg["nsamples"],
                        )
                        for seg in meta
                    ],
                    dtype=_seg_dtype,
                )
                seg_tables.append(seg_arr)
                # Valid starts: nsamples - seq_len + 1; at least 0
                usable = seg_arr["nsamples"] - self.seq_len + 1
                usable[usable < 0] = 0
                # Skip segments with CBC CAT2 or hardware-injection time
                key = (meta[0]["observing_run"], meta[0]["detector"])
                usable[~clean(meta, bad_segments[key])] = 0
                usable_parts.append(usable)
            pooled = (seg_tables[0] if len(seg_tables) == 1
                      else np.concatenate(seg_tables))
            usable = (usable_parts[0] if len(usable_parts) == 1
                      else np.concatenate(usable_parts))
            total = usable.sum()
            if total == 0:
                raise ValueError("seq_len exceeds all segments")
            self.seg_index[d] = pooled
            self.segment_probs[d] = usable / total

        self.rng = np.random.default_rng(seed)

        # Hard windows a batch replays with probability
        # _hard_bias_prob; set between epochs by HardMiningCallback
        # through set_hard_bank
        self._hard_dataset   = None
        self._hard_bias_prob = 0.0
        self._hard_lock      = threading.Lock()
        # Replay weight of a window with stat s: sigmoid(s)**gamma
        self._hard_focal_gamma = float(hard_focal_gamma)
        self._hard_cdf         = None      # rebuilt on dataset swap

        # Random reads from network storage are latency-bound, so
        # windows are read in parallel
        self._read_pool = ThreadPoolExecutor(max_workers=int(read_workers))

        self.queue = Queue(maxsize=self.prefetch_batches)
        self._stop_event = threading.Event()
        self._closed = False
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )
        self._prefetch_thread.start()

        # Stop the prefetch thread at interpreter exit; a running
        # thread can abort the process during finalisation. The weakref
        # keeps the registration from holding the sampler alive.
        _ref = weakref.ref(self)

        def _stop_at_exit(_ref=_ref):
            """Shut the sampler down if it still exists."""
            obj = _ref()
            if obj is not None:
                obj.shutdown()

        atexit.register(_stop_at_exit)

    def set_hard_dataset(self, dataset, hard_bias_prob: float | None = None):
        """
        Replace the hard windows the prefetch thread replays.

        Safe to call while the prefetch thread runs; the next batch it
        starts reading uses the new windows.

        Parameters
        ----------
        dataset : StartTimeDataset or None
            Hard windows to replay; ``None`` disables replay.
        hard_bias_prob : float or None
            Probability that a batch replays hard windows; ``None``
            keeps the current value.
        """
        with self._hard_lock:
            self._hard_dataset = dataset
            self._hard_cdf = self._build_hard_cdf(dataset)
            if hard_bias_prob is not None:
                self._hard_bias_prob = float(hard_bias_prob)

    def _build_hard_cdf(self, dataset):
        """
        Cumulative focal replay distribution over ``dataset``.

        A window with ranking stat ``s`` has weight
        ``sigmoid(s)**gamma``, the focal-loss modulating factor.

        Parameters
        ----------
        dataset : StartTimeDataset or None
            Hard windows with their ``scores``.

        Returns
        -------
        numpy.ndarray or None
            Cumulative probabilities, or ``None`` (uniform replay) if
            ``gamma <= 0``, the dataset is missing or empty, or no
            score is usable.
        """
        gamma = self._hard_focal_gamma
        if dataset is None or gamma <= 0.0 or len(dataset) == 0:
            return None
        s = np.asarray(dataset.scores, dtype=np.float64)
        if s.size != len(dataset) or not np.any(np.isfinite(s)):
            return None
        w = np.exp(-gamma * np.logaddexp(0.0, -s))     # sigmoid(s)**gamma
        w[~np.isfinite(w)] = 0.0
        total = w.sum()
        if total <= 0.0:
            return None
        cdf = np.cumsum(w / total)
        cdf[-1] = 1.0                                   # exact upper end
        return cdf

    def set_hard_bank(self, bank, active_indices, hard_bias_prob: float | None = None):
        """
        Replay the given rows of a hard-window bank.

        The rows' starts and latest re-evaluation stats are read from
        the bank file once, here, and passed to
        :meth:`set_hard_dataset`.

        Parameters
        ----------
        bank : HardMiningBank
            Bank of mined windows (:mod:`sage.data.noise.hard_bank`).
        active_indices : array-like of int
            Bank rows to replay; empty disables replay.
        hard_bias_prob : float or None
            Probability that a batch replays hard windows; ``None``
            keeps the current value.
        """
        from sage.data.noise.lowfar_noise import StartTimeDataset

        ai = np.asarray(active_indices, dtype=np.int64)
        if ai.size == 0:                                  # nothing currently hard
            self.set_hard_dataset(None, hard_bias_prob=hard_bias_prob)
            return
        # (start, segment, run) per detector; the run selects the memmap
        starts, segs, runs = bank.read_starts(ai)         # (K, D) each
        # Latest re-evaluation stat; sets the focal replay weights
        scores = bank.latest_eval(ai)
        ds = StartTimeDataset(
            detectors=bank.detectors,
            start_indices=starts,
            segment_indices=segs,
            run_indices=runs,
            gps_times=np.zeros(ai.size, dtype=np.float64),     # unused by sampling
            scores=scores,
            bin_files=bank.bin_files,
            sample_rate=bank.sample_rate,
            seq_len=bank.seq_len,
        )
        self.set_hard_dataset(ds, hard_bias_prob=hard_bias_prob)

    def _sample_hard_starts(self, dataset, batch_size: int):
        """
        Draw starts of ``batch_size`` hard windows from ``dataset``.

        Windows are drawn uniformly, or from the focal distribution of
        :meth:`_build_hard_cdf`. Each start is jittered uniformly by up
        to ``hard_start_jitter_s`` and clamped to its own segment.

        Parameters
        ----------
        dataset : StartTimeDataset
            Hard windows; if empty, random windows are drawn instead.
        batch_size : int
            Number of windows.

        Returns
        -------
        tuple of list of numpy.ndarray
            Starts, segment ids and run ids, as from
            :meth:`_sample_starts_batch`.
        """
        n = len(dataset)
        if n == 0:
            return self._sample_starts_batch(batch_size)
        cdf = self._hard_cdf
        if cdf is not None and cdf.size == n:
            idx = np.searchsorted(cdf, self.rng.random(batch_size), side="right")
            np.clip(idx, 0, n - 1, out=idx)
        else:
            idx = self.rng.integers(0, n, size=batch_size)

        jmax = int(round(self._hard_jitter_s * self.sample_rate))
        if jmax > 0 and self._seg_bounds_lut is None:
            self._build_seg_bounds_lut()

        start_indices, segment_indices, run_indices = [], [], []
        for d in range(self.n_detectors):
            base = dataset.start_indices[idx, d].astype(np.int64)
            segs = dataset.segment_indices[idx, d].astype(np.int64)
            runs = dataset.run_indices[idx, d].astype(np.int64)
            if jmax > 0:
                starts = base + self.rng.integers(-jmax, jmax + 1, size=batch_size)
                lut = self._seg_bounds_lut[d]
                for i in range(batch_size):
                    # Clamp to the segment; unknown segment: no jitter
                    lo, hi = lut.get((int(runs[i]), int(segs[i])),
                                     (int(base[i]), int(base[i])))
                    if starts[i] < lo:
                        starts[i] = lo
                    elif starts[i] > hi:
                        starts[i] = hi
            else:
                starts = base
            start_indices.append(starts)
            segment_indices.append(segs)
            run_indices.append(runs)
        return start_indices, segment_indices, run_indices

    def _build_seg_bounds_lut(self):
        """
        Build per detector the map ``(run_id, segment_id) -> (lo, hi)``.

        ``lo`` and ``hi`` are the first and last memmap indices at
        which a full window starts inside the segment.
        """
        lut = []
        for d in range(self.n_detectors):
            m = {}
            for row in self.seg_index[d]:
                start = int(row["start"])
                m[(int(row["run"]), int(row["idx"]))] = (
                    start, start + int(row["nsamples"]) - self.seq_len
                )
            lut.append(m)
        self._seg_bounds_lut = lut

    def _sample_starts_batch(self, batch_size: int, rng=None):
        """
        Draw random start indices for one batch.

        Per detector, ``batch_size`` segments are drawn with
        replacement, weighted by valid starts, each with a uniform start
        offset.
        Without ``rng``, uses ``self.rng``, which only the prefetch
        thread may use.

        Parameters
        ----------
        batch_size : int
            Number of windows to sample.
        rng : numpy.random.Generator, optional
            Used instead of ``self.rng``; safe from any thread.

        Returns
        -------
        start_indices : list of np.ndarray
            Per-detector array of absolute memmap start indices, shape
            ``(batch_size,)``.
        segment_indices : list of np.ndarray
            Per-detector array of segment IDs (for ASD look-up), shape
            ``(batch_size,)``.
        run_indices : list of np.ndarray
            Per-detector array of run IDs (the pooled run of each
            window; indexes ``self.mmaps[d]`` and the recolour segment
            bank), shape ``(batch_size,)``.
        """
        rng = self.rng if rng is None else rng
        start_indices = []
        segment_indices = []
        run_indices = []

        for d in range(self.n_detectors):
            seg_idx = self.seg_index[d]
            probs = self.segment_probs[d]
            chosen_segments = rng.choice(len(seg_idx), size=batch_size, p=probs)

            starts = np.empty(batch_size, dtype=np.int64)
            seg_ids = np.empty(batch_size, dtype=np.int64)
            run_ids = np.empty(batch_size, dtype=np.int64)
            for i, seg_i in enumerate(chosen_segments):
                seg = seg_idx[seg_i]
                max_offset = seg["nsamples"] - self.seq_len
                offset = rng.integers(0, max_offset + 1) if max_offset > 0 else 0
                starts[i] = seg["start"] + offset
                seg_ids[i] = seg["idx"]
                run_ids[i] = seg["run"]

            start_indices.append(starts)
            segment_indices.append(seg_ids)
            run_indices.append(run_ids)

        return start_indices, segment_indices, run_indices

    def sample_segments(self, batch_size: int, rng):
        """
        Segment and run ids of random windows, reading no strain.

        Drawn as a batch draws its windows, but from ``rng``.

        Parameters
        ----------
        batch_size : int
            Number of windows.
        rng : numpy.random.Generator
            Random generator.

        Returns
        -------
        segment_ids, run_ids : torch.Tensor, shape ``(B, D)``
            Segment and run ids, int64.
        """
        _, segment_indices, run_indices = self._sample_starts_batch(batch_size, rng)
        return (
            torch.from_numpy(np.stack(segment_indices, axis=1)),
            torch.from_numpy(np.stack(run_indices, axis=1)),
        )

    def _read_batch(self, batch_size: int):
        """
        Read one batch from the memmaps and convert it to FD.

        With probability ``hard_bias_prob`` the batch replays hard
        windows. Windows are read in parallel; uses ``self.rng``, so
        only the prefetch thread may call it.

        Parameters
        ----------
        batch_size : int
            Number of windows to read.

        Returns
        -------
        torch.Tensor
            ``postprocess_fn`` output, or the ``rfft`` of the windows,
            shape ``(B, D, F)`` complex64.
        """
        B = batch_size
        D = self.n_detectors
        seq_len = self.seq_len

        with self._hard_lock:
            hard_ds   = self._hard_dataset
            hard_prob = self._hard_bias_prob

        use_hard = (
            hard_ds is not None
            and hard_prob > 0.0
            and self.rng.random() < hard_prob
        )

        if use_hard:
            start_indices, segment_indices, run_indices = \
                self._sample_hard_starts(hard_ds, B)
        else:
            start_indices, segment_indices, run_indices = \
                self._sample_starts_batch(B)
        batch_tensor = torch.empty(
            (B, D, seq_len), dtype=torch.float32, device=self.device
        )

        # Read every (detector, window) slice in parallel
        arrs = [np.empty((B, seq_len), dtype=np.float32) for _ in range(D)]
        mmaps = self.mmaps

        def _read_window(job):
            """Read window ``i`` of detector ``d`` into ``arrs``."""
            d, i = job
            s = start_indices[d][i]
            r = run_indices[d][i]                 # run whose memmap holds it
            arrs[d][i] = mmaps[d][r][s : s + seq_len]

        list(self._read_pool.map(
            _read_window, [(d, i) for d in range(D) for i in range(B)]
        ))

        # Pinned memory serves the async copy to CUDA only
        pin = torch.device(self.device).type == "cuda"
        for d in range(D):
            arrs[d] /= dyn_range_fac()  # undo the dynamic-range scaling
            cpu_tensor = torch.from_numpy(arrs[d])
            if pin:
                cpu_tensor = cpu_tensor.pin_memory()
            batch_tensor[:, d, :].copy_(cpu_tensor, non_blocking=pin)

        # Segment and run ids (B, D); recolour keys its ASDs by them
        segment_ids = torch.empty((B, D), dtype=torch.int64)
        run_ids     = torch.empty((B, D), dtype=torch.int64)
        for d in range(D):
            segment_ids[:, d].copy_(torch.from_numpy(segment_indices[d]))
            run_ids[:, d].copy_(torch.from_numpy(run_indices[d]))

        if self.postprocess_fn is not None:
            batch_tensor = self.postprocess_fn(batch_tensor, segment_ids, run_ids)
        else:
            batch_tensor = torch.fft.rfft(batch_tensor, dim=-1, norm="forward")

        return batch_tensor

    def _prefetch_loop(self):
        """Fill the prefetch queue until the stop event is set."""
        while not self._stop_event.is_set():
            try:
                if not self.queue.full():
                    batch_tensor = self._read_batch(self.batch_size)
                    self.queue.put(batch_tensor)
                else:
                    self._stop_event.wait(0.01)
            except RuntimeError as exc:
                # The pool cannot schedule work at interpreter shutdown
                if self._stop_event.is_set() or "interpreter shutdown" in str(exc):
                    break
                raise

    def sample_batch(self):
        """
        Next batch from the prefetch queue, waiting until one is ready.

        Returns
        -------
        torch.Tensor
            FD noise batch (see :meth:`_read_batch`).
        """
        batch_tensor = self.queue.get()
        return batch_tensor

    @torch.no_grad()
    def forward(self):
        """
        Next noise batch and its targets.

        Returns
        -------
        tuple of torch.Tensor
            The batch (:meth:`sample_batch`) and zero targets of shape
            ``(batch_size, 1)``.
        """
        return self.sample_batch(), self.noise_target

    def shutdown(self):
        """
        Stop the prefetch thread; release memmaps and postprocess.

        Calls after the first do nothing.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._stop_event.set()
        # Drain so a thread blocked in put() sees the stop flag
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass
        t = getattr(self, "_prefetch_thread", None)
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5.0)
        pool = getattr(self, "_read_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
        for mm in getattr(self, "mmaps", []):
            del mm
        self.mmaps = []
        self.postprocess_fn = None

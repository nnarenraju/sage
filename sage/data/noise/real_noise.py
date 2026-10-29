#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
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

# Packages
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

# pycbc is an optional dependency; pull its dynamic-range constant lazily so the
# noise package imports without pycbc (see ._pycbc_lazy).
from ._pycbc_lazy import dyn_range_fac
from .data_quality import clean
from .selection import select_records, summary

# LOCAL
from sage.config.machine import segments_file
from sage.core.geometry import Window
from sage.core.logger import get_logger

logger = get_logger(__name__)


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

        # Weight ∝ number of valid start offsets = nsamples - seq_len + 1 (a
        # segment of exactly seq_len has one valid start at offset 0). Segments
        # shorter than seq_len get 0 weight. The old form dropped the +1, slightly
        # under-weighting short segments relative to long ones.
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

        meta_path = segments_file(source)
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

        # Weight ∝ number of valid start offsets = nsamples - seq_len + 1 (a
        # segment of exactly seq_len has one valid start at offset 0). Segments
        # shorter than seq_len get 0 weight. The old form dropped the +1, slightly
        # under-weighting short segments relative to long ones.
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
        max_start = seg_len - nsamples
        return rng.integers(0, max_start + 1)

    def sample(
        self,
        requested_nsamples: int,
        rng = None,
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
            noise /= dyn_range_fac()

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


class MemmapNoiseSampler(torch.nn.Module):
    """
    GPU-resident batch sampler for monolithic ``.bin`` noise files with
    asynchronous prefetching.

    Each detector's noise data is stored as a flat binary file (``float32``
    or ``float64``) accompanied by a sidecar ``*_segments.json`` file that
    records the GPS start/end times, sample indices, and per-segment metadata
    for every contiguous data segment.

    Sampling is duration-weighted: longer segments are proportionally more
    likely to be drawn so that no useful data is wasted.  Within each chosen
    segment, a uniformly random start offset is selected.

    An async daemon thread continuously fills a bounded
    :class:`queue.Queue` with pre-processed FD batches so that the GPU is
    never starved waiting for disk I/O.  Callers consume batches through
    :meth:`sample_batch`, which performs a thread-safe ``queue.get()``.

    .. warning::
        :meth:`_read_batch` and :meth:`_sample_starts_batch` access a
        non-thread-safe NumPy RNG and must **not** be called from the main
        thread while the prefetch daemon is running.  Always use
        :meth:`sample_batch` (queue pop) to consume batches safely.

    Parameters
    ----------
    runs : list of dict
        Observing runs to pool, each ``{"run", "data_dir", "bins"}``: the
        run name, its ASD directory, and one strain file per detector in
        channel order; optional ``portion`` or ``gps_range`` restrict the
        segments drawn from (:mod:`sage.data.noise.selection`).
    detectors : list of str
        Detectors, in channel order.
    window : Window
        Window geometry; sets the window length.
    batch_size : int
        Windows per batch.
    bad_segments : dict
        GWOSC bad time keyed by ``(observing run, detector)``, from
        :func:`sage.data.noise.data_quality.noise_bad_segments`; segments
        overlapping it are never drawn.
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
        Applied to ``(batch_td, segment_ids, run_ids)`` to convert from TD
        to FD. If ``None``, a plain ``torch.fft.rfft`` is used. The typical
        choice is :class:`RecolourPostprocess`.

    Attributes
    ----------
    GRAPH_READY : bool
        ``False`` — the prefetch queue pop cannot be traced by
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
        # Replayed hard windows are jittered by +/- this many seconds around the
        # mined start (uniform, clamped to the SAME segment) so biasing toward a
        # mined window keeps yielding varied (overlapping) noise rather than the
        # identical realisation every draw. 0 disables. Segment/run ids are left
        # unchanged, so the whitening ASD and recolour keys still match.
        self._hard_jitter_s = float(hard_start_jitter_s)
        self._seg_bounds_lut = None            # lazily built: (run,seg)->(lo,hi)/det
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.detectors = list(detectors)
        self.n_detectors = len(detectors)
        self.batch_size = batch_size
        self.postprocess_fn = postprocess_fn

        # A run id is carried through every window so it can be read from the
        # right run's mmap and recoloured with the right run's per-segment ASD.
        self.run_names = [r["run"] for r in runs]
        self.run_data_dirs = [r["data_dir"] for r in runs]
        self.n_runs = len(runs)

        self.mmaps = [[] for _ in range(self.n_detectors)]      # [d][run] -> memmap
        self.seg_index = [None] * self.n_detectors              # [d] -> pooled struct
        self.segment_probs = [None] * self.n_detectors          # [d] -> pooled probs
        self.dtypes = [[] for _ in range(self.n_detectors)]     # [d][run] -> dtype
        self.bin_files = []                                     # flat, for logging

        # Targets for noise batch
        self.noise_target = torch.zeros((self.batch_size, 1)).to(
            dtype=dtype, device=device
        )

        # Sidecar records of every run, all detectors read first: a portion
        # cut is shared across a run's detectors.
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
                # Bins bind to channels positionally (mmaps[d] <- bins[d]), so
                # the sidecar's own detector and observing run must match.
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

        # Segment table carries a run id so pooled runs never collide on the
        # per-(detector,run) dense ``segment_index``.
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
                # Weight ∝ number of valid start offsets = nsamples-seq_len+1
                # (exactly-seq_len segments have one valid start at offset 0);
                # shorter-than-seq_len segments get 0 weight.
                usable = seg_arr["nsamples"] - self.seq_len + 1
                usable[usable < 0] = 0
                # Skip segments holding CBC Category 2 or hardware-injection time.
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

        # deterministic RNG for reproducible training
        self.rng = np.random.default_rng(seed)

        # Hard-noise biasing ---------------------------------------------------
        # A StartTimeDataset of currently-hard windows.  The prefetch thread
        # draws from it with probability hard_bias_prob instead of sampling
        # randomly.  Populated live by HardMiningCallback via set_hard_bank()
        # (which calls set_hard_dataset) between epochs; starts empty.
        self._hard_dataset   = None
        self._hard_bias_prob = 0.0
        self._hard_lock      = threading.Lock()
        # Focal-style replay weighting: a window with ranking stat s is drawn with
        # weight sigmoid(s)**gamma (see _sample_hard_starts). 0.0 -> uniform.
        self._hard_focal_gamma = float(hard_focal_gamma)
        self._hard_cdf         = None      # cached, rebuilt on dataset swap
        # -----------------------------------------------------------------------

        # Worker pool that reads the batch's per-window noise slices in
        # parallel. NFS random reads are latency-bound, so reading a detector's
        # 128 windows serially costs ~1 s and starves the GPU; a wide pool
        # overlaps the read latency (~100x faster on the data mount).
        self._read_pool = ThreadPoolExecutor(max_workers=int(read_workers))

        # Prefetch queue
        self.queue = Queue(maxsize=self.prefetch_batches)
        self._stop_event = threading.Event()
        self._closed = False
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )
        self._prefetch_thread.start()

        # Stop the daemon prefetch thread cleanly at interpreter exit. Without
        # this it keeps running into finalization and can try to schedule work
        # (or touch CUDA / pinned memory) on a tearing-down interpreter, which
        # aborts the process. weakref so the registration can't keep the sampler
        # alive on its own.
        _ref = weakref.ref(self)

        def _stop_at_exit(_ref=_ref):
            obj = _ref()
            if obj is not None:
                obj.shutdown()

        atexit.register(_stop_at_exit)

    def set_hard_dataset(self, dataset, hard_bias_prob: float | None = None):
        """
        Replace the in-RAM hard-noise dataset (the prefetch thread's bias pool).

        Thread-safe: can be called from the main thread while the prefetch
        daemon is running.  The change takes effect on the next batch the
        prefetch thread starts reading.  Persistence of the hard windows is the
        HDF5 bank's job (see :meth:`set_hard_bank`); this only swaps the live
        snapshot the sampler draws from.

        Parameters
        ----------
        dataset : StartTimeDataset or None
            The hard windows to bias toward.  Pass ``None`` to disable biasing.
        hard_bias_prob : float or None
            New bias probability.  ``None`` keeps the existing value.
        """
        with self._hard_lock:
            self._hard_dataset = dataset
            self._hard_cdf = self._build_hard_cdf(dataset)
            if hard_bias_prob is not None:
                self._hard_bias_prob = float(hard_bias_prob)

    def _build_hard_cdf(self, dataset):
        """Cache the focal replay distribution for ``dataset`` (None -> uniform).

        Weight of a hard window with ranking stat ``s`` is ``sigmoid(s)**gamma``
        -- the focal-loss modulating factor for a negative the model is calling
        signal. It is monotone in ``s`` and BOUNDED: within the active set
        (``s >= keep_threshold``, typically 0) the hardest-to-easiest weight
        ratio is at most ``2**gamma``, so replay concentrates on the tail that
        sets the FAR without any window being replayed orders of magnitude more
        than its peers. ``bias_replays_per_epoch`` still caps the absolute count.

        Built once per dataset swap (between epochs), not per batch.
        """
        gamma = self._hard_focal_gamma
        if dataset is None or gamma <= 0.0 or len(dataset) == 0:
            return None
        s = np.asarray(dataset.scores, dtype=np.float64)
        if s.size != len(dataset) or not np.any(np.isfinite(s)):
            return None
        w = np.exp(-gamma * np.logaddexp(0.0, -s))     # sigmoid(s)**gamma, stable
        w[~np.isfinite(w)] = 0.0
        total = w.sum()
        if total <= 0.0:
            return None
        cdf = np.cumsum(w / total)
        cdf[-1] = 1.0                                   # guard fp drift
        return cdf

    def set_hard_bank(self, bank, active_indices, hard_bias_prob: float | None = None):
        """Augment random start-times with the *currently-hard* windows from a
        :class:`~sage.data.noise.hard_bank.HardMiningBank`.

        ``active_indices`` are the bank rows whose latest re-evaluation ranking
        stat is above the keep threshold (computed by the mining callback after
        re-eval).  Their start/segment indices are read **once** from the bank
        file here -- between epochs, after the miner has finished writing and
        while the prefetch thread only touches the in-RAM snapshot, so there is
        never concurrent HDF5 access.  The compact active subset (start/segment
        indices only) becomes the in-RAM hard dataset; the full bank (all
        history, embeddings, re-eval matrix) stays on disk.

        Nothing else in the hot path changes -- this simply swaps which windows
        ``_sample_hard_starts`` draws from, via the existing thread-safe
        :meth:`set_hard_dataset`.
        """
        from sage.data.noise.lowfar_noise import StartTimeDataset

        ai = np.asarray(active_indices, dtype=np.int64)
        if ai.size == 0:                                  # nothing currently hard
            self.set_hard_dataset(None, hard_bias_prob=hard_bias_prob)
            return
        # (start, segment, run) per detector -- the run tells the reader which
        # pooled file's mmap to read this hard window from.
        starts, segs, runs = bank.read_starts(ai)         # (K, D) each
        # Latest re-eval stat per active window -- drives the focal replay
        # weighting in _sample_hard_starts (ignored when hard_focal_gamma == 0).
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
        # epoch/save_dir omitted -> swap only, no .npz write (the bank persists).
        self.set_hard_dataset(ds, hard_bias_prob=hard_bias_prob)

    def _sample_hard_starts(self, dataset, batch_size: int):
        """
        Draw ``batch_size`` start indices from ``dataset``, each jittered by
        +/- ``self._hard_jitter_s`` seconds (uniform) and clamped to stay inside
        its OWN segment.

        Windows are drawn UNIFORMLY when ``hard_focal_gamma`` is 0, else from the
        cached focal distribution (``_build_hard_cdf``) so harder windows are
        replayed more often -- bounded, never unboundedly.

        The jitter (per window, per detector) means repeatedly biasing toward the
        same mined window yields varied, overlapping noise realisations instead of
        the identical window every time. Segment/run ids are unchanged, so the
        per-segment whitening ASD and recolour keys still line up. Returns lists
        in the same format as ``_sample_starts_batch``.
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
                    # clamp to [seg_start, seg_last_valid_start]; if the segment
                    # is unknown (shouldn't happen) fall back to no jitter.
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
        """Per-detector map ``(run_id, segment_id) -> (min_start, max_start)`` for
        clamping jittered hard-window starts to their own segment. ``max_start``
        is the last offset at which a full ``seq_len`` window still fits.
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

        For each detector, selects ``batch_size`` segments (with replacement,
        weighted by usable duration) and uniformly samples a start offset
        within each chosen segment.

        .. warning::
            Uses ``self.rng`` (non-thread-safe).  Must only be called from
            the prefetch daemon thread, never from the main thread.

        Parameters
        ----------
        batch_size : int
            Number of windows to sample.
        rng : numpy.random.Generator, optional
            Draw from this instead of ``self.rng``. Safe from any thread, since
            the prefetch stream is left untouched.

        Returns
        -------
        start_indices : list of np.ndarray
            Per-detector array of absolute memmap start indices, shape
            ``(batch_size,)``.
        segment_indices : list of np.ndarray
            Per-detector array of segment IDs (for ASD look-up), shape
            ``(batch_size,)``.
        run_indices : list of np.ndarray
            Per-detector array of run IDs (which pooled run each window came
            from; indexes ``self.mmaps[d]`` and the recolour segment bank),
            shape ``(batch_size,)``.
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
        Segment and run ids of ``batch_size`` windows, ``(B, D)`` int64 each.

        Drawn the way a batch draws its windows (segments weighted by usable
        duration, independently per detector) but from ``rng``, without reading
        any strain.
        """
        _, segment_indices, run_indices = self._sample_starts_batch(batch_size, rng)
        return (
            torch.from_numpy(np.stack(segment_indices, axis=1)),
            torch.from_numpy(np.stack(run_indices, axis=1)),
        )

    def _read_batch(self, batch_size: int):
        """
        Read one batch from the memmaps and return a preprocessed FD tensor.

        Reads are parallelised across detectors using a
        :class:`~concurrent.futures.ThreadPoolExecutor`.  The resulting
        numpy arrays are pinned and asynchronously copied to the GPU.

        .. warning::
            Calls :meth:`_sample_starts_batch` (non-thread-safe RNG).
            Must only be called from the prefetch daemon thread.

        Parameters
        ----------
        batch_size : int
            Number of windows to read.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)`` complex64 (if postprocess_fn is
        None, plain rfft output) or whatever shape ``postprocess_fn`` returns.
        """
        B = batch_size
        D = self.n_detectors
        seq_len = self.seq_len

        # Decide whether to draw from the hard-noise dataset
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

        # Read every (detector, window) slice in parallel on the shared pool.
        # NFS reads are latency-bound, so a wide pool overlaps the waits: a
        # detector's 128 windows cost ~1 s read serially, ~10 ms in parallel.
        arrs = [np.empty((B, seq_len), dtype=np.float32) for _ in range(D)]
        mmaps = self.mmaps

        def _read_window(job):
            d, i = job
            s = start_indices[d][i]
            r = run_indices[d][i]                 # which run's mmap this came from
            arrs[d][i] = mmaps[d][r][s : s + seq_len]

        list(self._read_pool.map(
            _read_window, [(d, i) for d in range(D) for i in range(B)]
        ))

        # pin_memory + non_blocking only help the async host->CUDA copy; on a
        # CPU-only node pin_memory raises (no driver), so gate on the target.
        pin = torch.device(self.device).type == "cuda"
        for d in range(D):
            arrs[d] /= dyn_range_fac()  # restore original scale
            cpu_tensor = torch.from_numpy(arrs[d])
            if pin:
                cpu_tensor = cpu_tensor.pin_memory()
            batch_tensor[:, d, :].copy_(cpu_tensor, non_blocking=pin)

        # Segment + run ids as CPU tensors (B, D) -- recolour keys its per-run
        # whitening ASD by (run_id, segment_index).
        segment_ids = torch.empty((B, D), dtype=torch.int64)
        run_ids     = torch.empty((B, D), dtype=torch.int64)
        for d in range(D):
            segment_ids[:, d].copy_(torch.from_numpy(segment_indices[d]))
            run_ids[:, d].copy_(torch.from_numpy(run_indices[d]))

        if self.postprocess_fn is not None:
            batch_tensor = self.postprocess_fn(batch_tensor, segment_ids, run_ids)
        else:
            # default: TD to FD only
            batch_tensor = torch.fft.rfft(batch_tensor, dim=-1, norm="forward")

        return batch_tensor

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            try:
                if not self.queue.full():
                    batch_tensor = self._read_batch(self.batch_size)
                    self.queue.put(batch_tensor)
                else:
                    # sleep briefly to yield CPU
                    self._stop_event.wait(0.01)
            except RuntimeError as exc:
                # At interpreter shutdown a fresh ThreadPoolExecutor can no
                # longer schedule work; stop quietly instead of aborting the
                # daemon thread with a traceback.
                if self._stop_event.is_set() or "interpreter shutdown" in str(exc):
                    break
                raise

    def sample_batch(self):
        """
        Return a GPU batch. Starts async prefetching if first call.
        """
        # If queue has a ready batch, return it
        batch_tensor = self.queue.get()
        return batch_tensor

    @torch.no_grad()
    def forward(self):
        return self.sample_batch(), self.noise_target

    def shutdown(self):
        """Stop the prefetch thread and release the memmaps and the
        postprocess (the recolour banks, ~16 GB per detector).

        Idempotent and safe to call explicitly, from atexit, or twice.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._stop_event.set()
        # Drain the queue so a thread parked in queue.put() can finish its
        # current iteration and observe the stop flag.
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

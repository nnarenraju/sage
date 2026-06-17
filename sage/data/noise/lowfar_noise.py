#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Start-time noise dataset and mining reader for GW detection training.

The standard training strategy randomly samples noise windows, so the model
rarely sees the extreme-tail background that dominates searches at very low
false alarm rates (FARs).  "Hard" noise mining targets those windows — ones that
fool a trained model into high ranking statistics — and persists only their
per-detector start times (not raw strain) for later replay during training.

This module provides the two shared primitives for that workflow:

  StartTimeDataset  — a persisted set of per-detector (start, segment) indices
                      for hard windows, with .save()/.load() (npz + json sidecar).
  _MiningReader     — reads noise windows at given per-detector start times from
                      the memmap noise sampler, for (re)scoring by the model.

The mining algorithms that produce these datasets live in
:mod:`sage.data.noise.qd_mining` (CMA-ME and CMA-MEGA).
"""

import json

import numpy as np
import torch

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import tqdm

# pycbc is optional; pull DYN_RANGE_FAC lazily (see ._pycbc_lazy).
from ._pycbc_lazy import dyn_range_fac

from sage.core.config import get_cfg


# ---------------------------------------------------------------------------
# StartTimeDataset
# ---------------------------------------------------------------------------

class StartTimeDataset:
    """
    Persisted record of per-detector noise window start times and their scores.

    Each of the N samples is identified by D per-detector absolute memmap
    indices (``start_indices``) and the corresponding D segment IDs
    (``segment_indices``).  The segment IDs match the ``segment_index`` field
    from the sidecar JSON metadata and are needed to look up PSDs during
    postprocessing.

    Saved as a compressed ``.npz`` archive; string metadata (detector names,
    file paths) is written to a companion ``.json`` file next to the ``.npz``.

    Parameters
    ----------
    detectors : list[str]
        Ordered detector names, e.g. ``["H1", "L1"]``.
    start_indices : np.ndarray, shape (N, D), dtype int64
        Absolute memmap start index for each sample × detector.
    segment_indices : np.ndarray, shape (N, D), dtype int64
        Segment ID for each sample × detector.
    gps_times : np.ndarray, shape (N,), dtype float64
        GPS start time of the H1 window (detector index 0), for reference.
    scores : np.ndarray, shape (N,), dtype float32
        Model ranking statistic for each sample.
    bin_files : list[str]
        Absolute paths to the original ``.bin`` noise files, one per detector.
    sample_rate : float
        Sample rate in Hz.
    seq_len : int
        Window length in samples (including padding).
    """

    def __init__(
        self,
        detectors,
        start_indices,
        segment_indices,
        gps_times,
        scores,
        bin_files,
        sample_rate,
        seq_len,
    ):
        self.detectors = list(detectors)
        self.start_indices = np.asarray(start_indices, dtype=np.int64)
        self.segment_indices = np.asarray(segment_indices, dtype=np.int64)
        self.gps_times = np.asarray(gps_times, dtype=np.float64)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.bin_files = [str(f) for f in bin_files]
        self.sample_rate = float(sample_rate)
        self.seq_len = int(seq_len)

    # ------------------------------------------------------------------
    def save(self, path):
        """Save to ``path`` (``.npz`` + companion ``.json`` for string metadata)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        np.savez_compressed(
            str(path),
            start_indices=self.start_indices,
            segment_indices=self.segment_indices,
            gps_times=self.gps_times,
            scores=self.scores,
            sample_rate=np.array([self.sample_rate]),
            seq_len=np.array([self.seq_len], dtype=np.int64),
        )
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({"detectors": self.detectors, "bin_files": self.bin_files}, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load from ``path`` (``.npz`` + companion ``.json``)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        d = np.load(str(path), allow_pickle=False)
        return cls(
            detectors=meta["detectors"],
            start_indices=d["start_indices"],
            segment_indices=d["segment_indices"],
            gps_times=d["gps_times"],
            scores=d["scores"],
            bin_files=meta["bin_files"],
            sample_rate=float(d["sample_rate"][0]),
            seq_len=int(d["seq_len"][0]),
        )

    # ------------------------------------------------------------------
    def filter(self, min_score):
        """Return a new dataset keeping only samples with score >= min_score."""
        mask = self.scores >= min_score
        return StartTimeDataset(
            detectors=self.detectors,
            start_indices=self.start_indices[mask],
            segment_indices=self.segment_indices[mask],
            gps_times=self.gps_times[mask],
            scores=self.scores[mask],
            bin_files=self.bin_files,
            sample_rate=self.sample_rate,
            seq_len=self.seq_len,
        )

    def merge(self, other):
        """Concatenate two compatible datasets (same detectors and bin_files)."""
        assert self.detectors == other.detectors, "detector lists must match"
        return StartTimeDataset(
            detectors=self.detectors,
            start_indices=np.concatenate([self.start_indices, other.start_indices], axis=0),
            segment_indices=np.concatenate([self.segment_indices, other.segment_indices], axis=0),
            gps_times=np.concatenate([self.gps_times, other.gps_times]),
            scores=np.concatenate([self.scores, other.scores]),
            bin_files=self.bin_files,
            sample_rate=self.sample_rate,
            seq_len=self.seq_len,
        )

    def __len__(self):
        return len(self.scores)

    def __repr__(self):
        if len(self) == 0:
            return "StartTimeDataset(0 samples)"
        return (
            f"StartTimeDataset({len(self):,} samples, "
            f"detectors={self.detectors}, "
            f"score=[{self.scores.min():.3f}, {self.scores.max():.3f}])"
        )


# ---------------------------------------------------------------------------
# _MiningReader  (internal)
# ---------------------------------------------------------------------------

class _MiningReader:
    """
    Reads noise windows for explicit per-detector memmap start indices.

    Borrows the already-open memmaps and segment metadata from an existing
    ``MemmapNoiseSampler`` to avoid reopening large binary files.  Uses its
    own NumPy RNG, completely independent of the sampler's prefetch thread.

    Parameters
    ----------
    noise_sampler : MemmapNoiseSampler
    seed : int or None
    """

    def __init__(self, noise_sampler, seed=None):
        self.mmaps = noise_sampler.mmaps
        self.seg_index = noise_sampler.seg_index        # list of structured arrays per detector
        self.segment_probs = noise_sampler.segment_probs
        self.seq_len = noise_sampler.seq_len
        self.n_detectors = noise_sampler.n_detectors
        self.device = noise_sampler.device
        self.postprocess_fn = noise_sampler.postprocess_fn
        self.rng = np.random.default_rng(seed)

        # Load GPS metadata from sidecar JSON files
        self.gps_meta = []   # per det: {segment_index -> {gps_start, sample_start_idx, sample_rate}}
        for p in noise_sampler.bin_files:
            meta_path = p.parent / f"{p.stem}_segments.json"
            with open(meta_path) as f:
                raw = json.load(f)
            self.gps_meta.append({
                m["segment_index"]: {
                    "gps_start": float(m["gps_start"]),
                    "sample_start_idx": int(m["sample_start_idx"]),
                    "sample_rate": float(m["sample_rate"]),
                }
                for m in raw
            })

        first_meta = next(iter(self.gps_meta[0].values()))
        self.sample_rate = first_meta["sample_rate"]

        # Vectorised lookup arrays (one per detector)
        self._gps_lookup = []    # for gps_from_starts
        self._seg_bounds = []    # for mutate_starts
        for d in range(self.n_detectors):
            seg_arr = self.seg_index[d]
            seg_ids = seg_arr["idx"].astype(np.int64)
            max_id = int(seg_ids.max())
            id_to_pos = np.full(max_id + 1, -1, dtype=np.int64)
            for i, sid in enumerate(seg_ids.tolist()):
                id_to_pos[int(sid)] = i

            gps_starts = np.array(
                [self.gps_meta[d][int(sid)]["gps_start"] for sid in seg_ids], dtype=np.float64
            )
            ssi = np.array(
                [self.gps_meta[d][int(sid)]["sample_start_idx"] for sid in seg_ids], dtype=np.float64
            )
            sr = np.array(
                [self.gps_meta[d][int(sid)]["sample_rate"] for sid in seg_ids], dtype=np.float64
            )
            self._gps_lookup.append(
                {"id_to_pos": id_to_pos, "gps_starts": gps_starts, "ssi": ssi, "sr": sr}
            )
            self._seg_bounds.append(
                {
                    "id_to_pos": id_to_pos,
                    "starts": seg_arr["start"].astype(np.int64),
                    "ends": seg_arr["end"].astype(np.int64),
                }
            )

    # ------------------------------------------------------------------
    def random_starts(self, batch_size, weights=None):
        """
        Draw ``batch_size`` random noise windows.

        Parameters
        ----------
        batch_size : int
        weights : list[np.ndarray] or None
            Per-detector segment sampling weights ``(n_segs_d,)`` each.
            Defaults to duration-weighted ``self.segment_probs``.

        Returns
        -------
        starts : (B, D) int64 — absolute memmap start indices
        segs   : (B, D) int64 — segment IDs
        """
        if weights is None:
            weights = self.segment_probs

        starts = np.empty((batch_size, self.n_detectors), dtype=np.int64)
        segs = np.empty((batch_size, self.n_detectors), dtype=np.int64)

        for d in range(self.n_detectors):
            seg_arr = self.seg_index[d]
            chosen = self.rng.choice(len(seg_arr), size=batch_size, p=weights[d])
            chosen_segs = seg_arr[chosen]
            max_offsets = np.maximum(
                0, chosen_segs["nsamples"].astype(np.int64) - self.seq_len
            )
            u = self.rng.random(batch_size)
            offsets = np.minimum((u * (max_offsets + 1)).astype(np.int64), max_offsets)
            starts[:, d] = chosen_segs["start"].astype(np.int64) + offsets
            segs[:, d] = chosen_segs["idx"].astype(np.int64)

        return starts, segs

    # ------------------------------------------------------------------
    def read_batch(self, starts, segs):
        """
        Read noise windows and convert to frequency domain.

        Parameters
        ----------
        starts : (B, D) int64
        segs   : (B, D) int64

        Returns
        -------
        torch.Tensor, shape (B, D, F), complex64, on ``self.device``
        """
        B = len(starts)
        D = self.n_detectors
        batch_td = torch.empty(
            (B, D, self.seq_len), dtype=torch.float32, device=self.device
        )

        def read_det(d):
            mm = self.mmaps[d]
            arr = np.empty((B, self.seq_len), dtype=np.float32)
            for i in range(B):
                s = int(starts[i, d])
                arr[i] = mm[s : s + self.seq_len].astype(np.float32)
            arr /= dyn_range_fac()
            return arr

        with ThreadPoolExecutor(max_workers=D) as pool:
            results = list(pool.map(read_det, range(D)))

        for d, arr in enumerate(results):
            cpu_t = torch.from_numpy(arr).pin_memory()
            batch_td[:, d, :].copy_(cpu_t, non_blocking=True)

        segment_ids = torch.from_numpy(segs.astype(np.int64))   # (B, D) CPU

        if self.postprocess_fn is not None:
            return self.postprocess_fn(batch_td, segment_ids)
        return torch.fft.rfft(batch_td, dim=-1, norm="forward")

    # ------------------------------------------------------------------
    def mutate_starts(self, starts, segs, sigma_samples):
        """
        Gaussian-perturb start indices, clamped to segment bounds.

        Parameters
        ----------
        starts        : (B, D) int64
        segs          : (B, D) int64
        sigma_samples : int   standard deviation in samples

        Returns
        -------
        new_starts : (B, D) int64
        segs       : (B, D) int64 (unchanged copy)
        """
        deltas = (self.rng.standard_normal(len(starts)) * sigma_samples).astype(np.int64)
        new_starts = starts.copy()

        for d in range(self.n_detectors):
            bounds = self._seg_bounds[d]
            itp = bounds["id_to_pos"]
            clamped_ids = np.clip(segs[:, d], 0, len(itp) - 1)
            positions = itp[clamped_ids]
            lo = bounds["starts"][positions]
            hi = np.maximum(bounds["ends"][positions] - self.seq_len, lo)
            new_starts[:, d] = np.clip(starts[:, d] + deltas, lo, hi)

        return new_starts, segs.copy()

    # ------------------------------------------------------------------
    def gps_from_starts(self, starts, segs):
        """
        GPS start time for each sample using detector 0 (H1).

        Parameters
        ----------
        starts : (B, D) int64
        segs   : (B, D) int64

        Returns
        -------
        (B,) float64
        """
        lk = self._gps_lookup[0]
        itp = lk["id_to_pos"]
        clamped = np.clip(segs[:, 0], 0, len(itp) - 1)
        pos = itp[clamped]
        return lk["gps_starts"][pos] + (starts[:, 0].astype(np.float64) - lk["ssi"][pos]) / lk["sr"][pos]

    # ------------------------------------------------------------------
    def gps_range(self):
        """Return ``(t_min, t_max)`` GPS over all segments × detectors."""
        t_min, t_max = float("inf"), float("-inf")
        for d in range(self.n_detectors):
            for m in self.gps_meta[d].values():
                t_min = min(t_min, m["gps_start"])
                t_max = max(t_max, m["gps_start"])
        return t_min, t_max

    def _empty_dataset(self, noise_sampler):
        cfg = get_cfg()
        D = self.n_detectors
        return StartTimeDataset(
            detectors=cfg.detectors,
            start_indices=np.empty((0, D), dtype=np.int64),
            segment_indices=np.empty((0, D), dtype=np.int64),
            gps_times=np.empty(0, dtype=np.float64),
            scores=np.empty(0, dtype=np.float32),
            bin_files=[str(p) for p in noise_sampler.bin_files],
            sample_rate=self.sample_rate,
            seq_len=self.seq_len,
        )

    @staticmethod
    def _score_percentile_str(scores):
        if len(scores) == 0:
            return "n/a"
        pcts = np.percentile(scores, [50, 75, 90, 95, 99])
        return "50/75/90/95/99 = " + "/".join(f"{p:.3f}" for p in pcts)


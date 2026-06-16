#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Low-FAR noise mining for GW detection training.

The standard training strategy randomly samples noise windows, which means the
model never encounters the extreme-tail background events that dominate searches
at very low false alarm rates (FARs).  This module mines for "hard" noise
samples — windows that fool a trained model into high ranking statistics — and
persists only their per-detector start times (not raw strain) for later use.

Three mining strategies:

  BruteForceMiner        — random sampling + score threshold (Method 1)
  MAPElitesMiner         — MAP-Elites quality-diversity; 1000-cell GPS time
                           archive × K samples/cell for temporal diversity
                           (Method 2a)
  CEMRareEventMiner      — Cross-Entropy Method; adapts per-detector segment
                           sampling weights toward high-score tails while a
                           diversity floor prevents mode collapse (Method 2b)

Usage example
-------------
  # --- offline mining (run once) ---
  miner = MAPElitesMiner(threshold=4.0)
  dataset = miner.mine(model, noise_sampler, processor, device="cuda:0")
  dataset.save("hard_noise_map_elites.npz")

  # --- during training ---
  dataset = StartTimeDataset.load("hard_noise_map_elites.npz")
  hard_sampler = StartTimeNoiseSampler(dataset, postprocess_fn, cfg.batch_size, cfg.device)
  noise_fd, noise_target = hard_sampler()
"""

import json
import math
import heapq
import threading

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from contextlib import nullcontext
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


# ---------------------------------------------------------------------------
# BruteForceMiner — Method 1
# ---------------------------------------------------------------------------

class BruteForceMiner:
    """
    Randomly sample noise windows, score with a model, keep those above a
    threshold.  Saves only per-detector start times — not raw strain.

    Parameters
    ----------
    threshold : float
        Minimum ranking statistic (raw logit) to retain a sample.
    max_samples : int
        Hard cap; when exceeded, a streaming top-K prune keeps only the
        highest-scoring ``max_samples`` seen so far.
    batch_size : int
        Windows evaluated per GPU forward pass.
    prune_every : int
        Prune the accumulated buffer every this many batches.
    autocast : bool
        Enable mixed-precision (float16) during inference.
    """

    def __init__(
        self,
        threshold: float,
        max_samples: int = 10_000_000,
        batch_size: int = 256,
        prune_every: int = 50,
        autocast: bool = True,
    ):
        self.threshold = threshold
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.prune_every = prune_every
        self.autocast = autocast

    @torch.no_grad()
    def mine(
        self,
        model,
        noise_sampler,
        processor,
        device: str,
        n_windows: int = 50_000_000,
    ) -> StartTimeDataset:
        """
        Parameters
        ----------
        model : nn.Module
            ``out[0]`` must be the ranking statistic, shape ``(B, 1)``.
        noise_sampler : MemmapNoiseSampler
        processor : callable
            Whitening + multirate preprocessing.
        device : str
        n_windows : int
            Total noise windows to evaluate.
        """
        reader = _MiningReader(noise_sampler)
        was_training = model.training
        model.eval()

        cast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.autocast else nullcontext()
        )

        n_batches = math.ceil(n_windows / self.batch_size)
        acc_starts, acc_segs, acc_scores = [], [], []
        n_kept = 0
        n_total = 0

        for i in tqdm(range(n_batches), desc="[BruteForce] mining"):
            starts, segs = reader.random_starts(self.batch_size)
            noise_fd = reader.read_batch(starts, segs)
            x = processor(noise_fd)
            with cast:
                out = model(x)
            scores = out[0].squeeze(1).float().cpu().numpy()
            n_total += len(scores)

            mask = scores >= self.threshold
            if mask.any():
                acc_starts.append(starts[mask])
                acc_segs.append(segs[mask])
                acc_scores.append(scores[mask])
                n_kept += int(mask.sum())

            if (i + 1) % self.prune_every == 0 and n_kept > self.max_samples:
                all_s = np.concatenate(acc_scores)
                k = self.max_samples
                top_k = np.argpartition(all_s, -k)[-k:]
                all_st = np.concatenate(acc_starts, axis=0)
                all_sg = np.concatenate(acc_segs, axis=0)
                acc_starts = [all_st[top_k]]
                acc_segs = [all_sg[top_k]]
                acc_scores = [all_s[top_k]]
                n_kept = k

        if was_training:
            model.train()

        if not acc_starts:
            print(
                f"[BruteForce] No samples above threshold {self.threshold:.3f} "
                f"(evaluated {n_total:,} windows)."
            )
            return reader._empty_dataset(noise_sampler)

        all_starts = np.concatenate(acc_starts, axis=0)
        all_segs = np.concatenate(acc_segs, axis=0)
        all_scores = np.concatenate(acc_scores).astype(np.float32)

        if len(all_scores) > self.max_samples:
            k = self.max_samples
            top_k = np.argpartition(all_scores, -k)[-k:]
            all_starts, all_segs, all_scores = all_starts[top_k], all_segs[top_k], all_scores[top_k]

        gps = reader.gps_from_starts(all_starts, all_segs)
        sort_idx = np.argsort(all_scores)[::-1]

        print(
            f"[BruteForce] Evaluated {n_total:,} windows — "
            f"kept {len(all_scores):,} above threshold {self.threshold:.3f}"
        )
        print(f"  Score percentiles: {reader._score_percentile_str(all_scores)}")

        cfg = get_cfg()
        return StartTimeDataset(
            detectors=cfg.detectors,
            start_indices=all_starts[sort_idx],
            segment_indices=all_segs[sort_idx],
            gps_times=gps[sort_idx],
            scores=all_scores[sort_idx],
            bin_files=[str(p) for p in noise_sampler.bin_files],
            sample_rate=reader.sample_rate,
            seq_len=noise_sampler.seq_len,
        )


# ---------------------------------------------------------------------------
# _MAPElitesArchive  (internal)
# ---------------------------------------------------------------------------

class _MAPElitesArchive:
    """
    1-D GPS-time-binned archive; each cell holds the top ``samples_per_cell``
    scoring samples found in that temporal region.

    Per-cell min-heaps (keyed by score) give O(log K) insert and O(1)
    worst-score access.  Heap entries are plain tuples so comparisons work
    natively:
        ``(score, start_det0, start_det1, ..., seg_det0, seg_det1, ...)``
    """

    def __init__(self, n_cells, samples_per_cell, gps_t_min, gps_t_max, n_detectors):
        self.n_cells = n_cells
        self.samples_per_cell = samples_per_cell
        self.gps_t_min = gps_t_min
        self.cell_width = (gps_t_max - gps_t_min) / n_cells
        self.n_detectors = n_detectors
        self.cells = [[] for _ in range(n_cells)]
        self._flat_valid = False
        self._flat_starts = None
        self._flat_segs = None
        self._flat_scores = None

    def _gps_to_cell(self, gps):
        return max(0, min(int((gps - self.gps_t_min) / self.cell_width), self.n_cells - 1))

    def _pack(self, score, starts_row, segs_row):
        return (float(score),) + tuple(int(s) for s in starts_row) + tuple(int(g) for g in segs_row)

    def _unpack(self, item):
        D = self.n_detectors
        score = item[0]
        starts = np.array(item[1 : 1 + D], dtype=np.int64)
        segs = np.array(item[1 + D :], dtype=np.int64)
        return score, starts, segs

    def update(self, scores, starts, segs, gps_times):
        """Insert a batch; returns the number of archive improvements."""
        n_improved = 0
        for i in range(len(scores)):
            cell_id = self._gps_to_cell(float(gps_times[i]))
            cell = self.cells[cell_id]
            item = self._pack(float(scores[i]), starts[i], segs[i])
            if len(cell) < self.samples_per_cell:
                heapq.heappush(cell, item)
                n_improved += 1
                self._flat_valid = False
            elif float(scores[i]) > cell[0][0]:
                heapq.heapreplace(cell, item)
                n_improved += 1
                self._flat_valid = False
        return n_improved

    def _rebuild_flat(self):
        all_st, all_sg, all_sc = [], [], []
        for cell in self.cells:
            for item in cell:
                sc, st, sg = self._unpack(item)
                all_st.append(st)
                all_sg.append(sg)
                all_sc.append(sc)
        if not all_st:
            return
        self._flat_starts = np.stack(all_st, axis=0)
        self._flat_segs = np.stack(all_sg, axis=0)
        self._flat_scores = np.array(all_sc, dtype=np.float32)
        self._flat_valid = True

    def propose_mutations(self, n, rng, sigma_samples, reader):
        """Sample n elites uniformly and Gaussian-mutate their start times."""
        if not self._flat_valid:
            self._rebuild_flat()
        if self._flat_starts is None or len(self._flat_starts) == 0:
            return reader.random_starts(n)
        idx = rng.integers(0, len(self._flat_starts), size=n)
        return reader.mutate_starts(self._flat_starts[idx], self._flat_segs[idx], sigma_samples)

    @property
    def total_samples(self):
        return sum(len(c) for c in self.cells)

    @property
    def n_filled_cells(self):
        return sum(1 for c in self.cells if c)


# ---------------------------------------------------------------------------
# MAPElitesMiner — Method 2a
# ---------------------------------------------------------------------------

class MAPElitesMiner:
    """
    MAP-Elites quality-diversity mining.

    The GPS timeline is divided into ``n_cells`` equal-width bins.  Each bin
    keeps the ``samples_per_cell`` highest-scoring windows found in that
    temporal region.  A 50/50 explore/exploit loop drives the search: random
    new windows explore unstudied regions while Gaussian mutations of archive
    elites exploit known high-score neighbourhoods.

    Default capacity: ``n_cells=1000 × samples_per_cell=10_000`` → 10 M samples.

    Parameters
    ----------
    n_cells : int
        Number of GPS time bins in the archive.
    samples_per_cell : int
        Maximum samples retained per bin.
    init_batches : int
        Random batches used to seed the archive before the main loop.
    n_iterations : int
        Main explore/exploit loop iterations.
    explore_fraction : float
        Fraction of each batch sampled uniformly at random; remainder are
        mutations of existing archive elites.
    mutation_sigma_s : float
        Standard deviation of Gaussian mutations in seconds.
    batch_size : int
    threshold : float
        Minimum score for samples in the returned dataset.
    autocast : bool
    """

    def __init__(
        self,
        n_cells: int = 1000,
        samples_per_cell: int = 10_000,
        init_batches: int = 500,
        n_iterations: int = 5_000,
        explore_fraction: float = 0.5,
        mutation_sigma_s: float = 300.0,
        batch_size: int = 256,
        threshold: float = 3.0,
        autocast: bool = True,
    ):
        self.n_cells = n_cells
        self.samples_per_cell = samples_per_cell
        self.init_batches = init_batches
        self.n_iterations = n_iterations
        self.explore_fraction = explore_fraction
        self.mutation_sigma_s = mutation_sigma_s
        self.batch_size = batch_size
        self.threshold = threshold
        self.autocast = autocast

    @torch.no_grad()
    def mine(self, model, noise_sampler, processor, device: str) -> StartTimeDataset:
        """
        Run the MAP-Elites pass.

        Returns a ``StartTimeDataset`` of all archive entries above
        ``self.threshold``.
        """
        reader = _MiningReader(noise_sampler)
        was_training = model.training
        model.eval()

        cast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.autocast else nullcontext()
        )

        t_min, t_max = reader.gps_range()
        sigma_samples = int(self.mutation_sigma_s * reader.sample_rate)

        archive = _MAPElitesArchive(
            n_cells=self.n_cells,
            samples_per_cell=self.samples_per_cell,
            gps_t_min=t_min,
            gps_t_max=t_max + 3600.0,   # buffer to avoid boundary effects
            n_detectors=reader.n_detectors,
        )

        def eval_and_update(starts, segs):
            noise_fd = reader.read_batch(starts, segs)
            x = processor(noise_fd)
            with cast:
                out = model(x)
            sc = out[0].squeeze(1).float().cpu().numpy()
            gps = reader.gps_from_starts(starts, segs)
            archive.update(sc, starts, segs, gps)
            return sc

        # --- Init phase: random exploration to seed the archive ---
        print(f"[MAP-Elites] Init: {self.init_batches} random batches …")
        for _ in tqdm(range(self.init_batches), desc="MAP-Elites init", leave=False):
            starts, segs = reader.random_starts(self.batch_size)
            eval_and_update(starts, segs)

        print(
            f"[MAP-Elites] Archive seeded: "
            f"{archive.n_filled_cells}/{self.n_cells} cells, "
            f"{archive.total_samples:,} samples"
        )

        # --- Main loop: 50 % explore + 50 % exploit ---
        n_explore = max(1, int(self.explore_fraction * self.batch_size))
        n_exploit = self.batch_size - n_explore
        log_every = max(1, self.n_iterations // 10)

        print(
            f"[MAP-Elites] Main loop: {self.n_iterations} iters "
            f"({n_explore} explore + {n_exploit} exploit per batch) …"
        )
        for i in tqdm(range(self.n_iterations), desc="MAP-Elites", leave=False):
            starts_e, segs_e = reader.random_starts(n_explore)
            starts_x, segs_x = archive.propose_mutations(n_exploit, reader.rng, sigma_samples, reader)
            starts = np.concatenate([starts_e, starts_x], axis=0)
            segs = np.concatenate([segs_e, segs_x], axis=0)
            eval_and_update(starts, segs)

            if (i + 1) % log_every == 0:
                if not archive._flat_valid:
                    archive._rebuild_flat()
                n_above = (
                    int((archive._flat_scores >= self.threshold).sum())
                    if archive._flat_scores is not None else 0
                )
                print(
                    f"  [iter {i+1:,}/{self.n_iterations}] "
                    f"cells: {archive.n_filled_cells}/{self.n_cells}, "
                    f"total: {archive.total_samples:,}, "
                    f"above threshold: {n_above:,}"
                )

        if was_training:
            model.train()

        archive._rebuild_flat()
        if archive._flat_starts is None or len(archive._flat_starts) == 0:
            print("[MAP-Elites] Archive is empty — no samples returned.")
            return reader._empty_dataset(noise_sampler)

        mask = archive._flat_scores >= self.threshold
        starts_out = archive._flat_starts[mask]
        segs_out = archive._flat_segs[mask]
        scores_out = archive._flat_scores[mask]
        gps_out = reader.gps_from_starts(starts_out, segs_out)
        sort_idx = np.argsort(scores_out)[::-1]

        print(
            f"[MAP-Elites] Done — {int(mask.sum()):,} samples "
            f"above threshold {self.threshold:.3f}"
        )
        print(f"  Score percentiles: {reader._score_percentile_str(scores_out)}")

        cfg = get_cfg()
        return StartTimeDataset(
            detectors=cfg.detectors,
            start_indices=starts_out[sort_idx],
            segment_indices=segs_out[sort_idx],
            gps_times=gps_out[sort_idx],
            scores=scores_out[sort_idx],
            bin_files=[str(p) for p in noise_sampler.bin_files],
            sample_rate=reader.sample_rate,
            seq_len=noise_sampler.seq_len,
        )


# ---------------------------------------------------------------------------
# CEMRareEventMiner — Method 2b
# ---------------------------------------------------------------------------

class CEMRareEventMiner:
    """
    Cross-Entropy Method for rare-event noise mining.

    Maintains per-detector segment sampling weights and iteratively updates
    them toward the high-scoring tail of the distribution.  Each generation:

    1. Sample ``batch_size`` windows proportional to current segment weights.
    2. Evaluate model; collect samples above ``threshold``.
    3. Identify the top ``elite_fraction`` by score.
    4. Update each detector's segment weights using an EMA toward the
       normalised elite segment counts.
    5. Add ``diversity_floor`` to prevent any segment from being starved.

    Parameters
    ----------
    n_generations : int
    batch_size : int
    elite_fraction : float
        Top fraction of each generation used for the weight update (e.g. 0.05).
    learning_rate : float
        EMA learning rate: ``w ← (1-lr)*w + lr*elite_counts``.
    diversity_floor : float
        Additive constant preventing any segment weight from reaching zero.
    threshold : float
        Minimum score for a sample to be included in the output dataset.
    autocast : bool
    """

    def __init__(
        self,
        n_generations: int = 200,
        batch_size: int = 1024,
        elite_fraction: float = 0.05,
        learning_rate: float = 0.3,
        diversity_floor: float = 1e-4,
        threshold: float = 3.0,
        autocast: bool = True,
    ):
        self.n_generations = n_generations
        self.batch_size = batch_size
        self.elite_fraction = elite_fraction
        self.learning_rate = learning_rate
        self.diversity_floor = diversity_floor
        self.threshold = threshold
        self.autocast = autocast

    @torch.no_grad()
    def mine(self, model, noise_sampler, processor, device: str) -> StartTimeDataset:
        """
        Run the CEM mining pass.

        Returns a ``StartTimeDataset`` of all samples above ``self.threshold``
        encountered across all generations.
        """
        reader = _MiningReader(noise_sampler)
        was_training = model.training
        model.eval()

        cast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.autocast else nullcontext()
        )

        # Start from duration-weighted uniform (same as MemmapNoiseSampler)
        weights = [reader.segment_probs[d].copy() for d in range(reader.n_detectors)]

        all_starts, all_segs, all_scores = [], [], []
        log_every = max(1, self.n_generations // 10)

        print(f"[CEM] {self.n_generations} generations × batch {self.batch_size} …")
        for gen in tqdm(range(self.n_generations), desc="CEM mining"):
            starts, segs = reader.random_starts(self.batch_size, weights=weights)

            noise_fd = reader.read_batch(starts, segs)
            x = processor(noise_fd)
            with cast:
                out = model(x)
            scores = out[0].squeeze(1).float().cpu().numpy()

            # Collect samples above threshold
            mask = scores >= self.threshold
            if mask.any():
                all_starts.append(starts[mask])
                all_segs.append(segs[mask])
                all_scores.append(scores[mask])

            # CEM weight update toward elite segment counts
            n_elite = max(1, int(self.batch_size * self.elite_fraction))
            elite_idx = np.argpartition(scores, -n_elite)[-n_elite:]
            elite_mask = np.zeros(len(scores), dtype=bool)
            elite_mask[elite_idx] = True

            for d in range(reader.n_detectors):
                itp = reader._seg_bounds[d]["id_to_pos"]
                n_segs_d = len(weights[d])
                elite_seg_ids = segs[elite_mask, d].astype(np.int64)
                valid = (elite_seg_ids >= 0) & (elite_seg_ids < len(itp))
                positions = itp[elite_seg_ids[valid]]
                pos_valid = positions >= 0
                counts = np.zeros(n_segs_d, dtype=np.float64)
                np.add.at(counts, positions[pos_valid], 1.0)

                new_w = (
                    (1.0 - self.learning_rate) * weights[d]
                    + self.learning_rate * (counts / max(n_elite, 1))
                )
                new_w += self.diversity_floor
                new_w /= new_w.sum()
                weights[d] = new_w

            if (gen + 1) % log_every == 0:
                n_above = sum(len(s) for s in all_scores)
                h = -float(np.sum(weights[0] * np.log(np.maximum(weights[0], 1e-12))))
                h_uniform = float(np.log(len(weights[0])))
                print(
                    f"  [gen {gen+1}/{self.n_generations}] "
                    f"above threshold: {n_above:,}, "
                    f"score p90: {np.percentile(scores, 90):.3f}, "
                    f"weight entropy H1: {h:.2f}/{h_uniform:.2f}"
                )

        if was_training:
            model.train()

        if not all_starts:
            print(
                f"[CEM] No samples above threshold {self.threshold:.3f} "
                f"after {self.n_generations} generations."
            )
            return reader._empty_dataset(noise_sampler)

        combined_starts = np.concatenate(all_starts, axis=0)
        combined_segs = np.concatenate(all_segs, axis=0)
        combined_scores = np.concatenate(all_scores).astype(np.float32)
        gps = reader.gps_from_starts(combined_starts, combined_segs)
        sort_idx = np.argsort(combined_scores)[::-1]

        print(
            f"[CEM] Done — {len(combined_scores):,} samples "
            f"above threshold {self.threshold:.3f}"
        )
        print(f"  Score percentiles: {reader._score_percentile_str(combined_scores)}")

        cfg = get_cfg()
        return StartTimeDataset(
            detectors=cfg.detectors,
            start_indices=combined_starts[sort_idx],
            segment_indices=combined_segs[sort_idx],
            gps_times=gps[sort_idx],
            scores=combined_scores[sort_idx],
            bin_files=[str(p) for p in noise_sampler.bin_files],
            sample_rate=reader.sample_rate,
            seq_len=noise_sampler.seq_len,
        )


# ---------------------------------------------------------------------------
# StartTimeNoiseSampler — training-time sampler
# ---------------------------------------------------------------------------

class StartTimeNoiseSampler(nn.Module):
    """
    Drop-in replacement for ``MemmapNoiseSampler`` that serves noise windows
    whose start times were pre-mined.

    Re-opens the original ``.bin`` files and uses the same async prefetch-queue
    architecture as ``MemmapNoiseSampler`` to keep the GPU fed during training.

    Parameters
    ----------
    dataset : StartTimeDataset
        A loaded dataset of mined start times.
    postprocess_fn : callable or None
        Postprocessing applied to ``(batch_td, segment_ids)`` to convert from
        TD to FD (e.g. ``RecolourPostprocess``).  If ``None``, plain rfft.
    batch_size : int
    device : str
    seed : int or None
    prefetch : int
        Prefetch queue depth.
    """

    GRAPH_READY = False

    def __init__(
        self,
        dataset: StartTimeDataset,
        postprocess_fn,
        batch_size: int,
        device: str,
        seed=None,
        prefetch: int = 3,
    ):
        super().__init__()
        self.dataset = dataset
        self.postprocess_fn = postprocess_fn
        self.batch_size = batch_size
        self.device = device
        self.seq_len = dataset.seq_len
        self.n_detectors = len(dataset.detectors)
        self.n_samples = len(dataset)
        self.rng = np.random.default_rng(seed)

        # Open memmaps from original bin files
        self.mmaps = []
        for bin_file in dataset.bin_files:
            p = Path(bin_file)
            meta_path = p.parent / f"{p.stem}_segments.json"
            with open(meta_path) as f:
                meta_raw = json.load(f)
            dtype = np.dtype(meta_raw[0]["dtype"]).newbyteorder(meta_raw[0]["endianness"])
            self.mmaps.append(np.memmap(p, dtype=dtype, mode="r"))

        self.noise_target = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

        self._queue = Queue(maxsize=prefetch)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _read_batch(self):
        B = self.batch_size
        idx = self.rng.integers(0, self.n_samples, size=B)
        starts = self.dataset.start_indices[idx]     # (B, D)
        segs = self.dataset.segment_indices[idx]     # (B, D)

        batch_td = torch.empty(
            (B, self.n_detectors, self.seq_len), dtype=torch.float32, device=self.device
        )

        def read_det(d):
            mm = self.mmaps[d]
            arr = np.empty((B, self.seq_len), dtype=np.float32)
            for i in range(B):
                s = int(starts[i, d])
                arr[i] = mm[s : s + self.seq_len].astype(np.float32)
            arr /= dyn_range_fac()
            return arr

        with ThreadPoolExecutor(max_workers=self.n_detectors) as pool:
            results = list(pool.map(read_det, range(self.n_detectors)))

        for d, arr in enumerate(results):
            cpu_t = torch.from_numpy(arr).pin_memory()
            batch_td[:, d, :].copy_(cpu_t, non_blocking=True)

        segment_ids = torch.from_numpy(segs.astype(np.int64))

        if self.postprocess_fn is not None:
            return self.postprocess_fn(batch_td, segment_ids)
        return torch.fft.rfft(batch_td, dim=-1, norm="forward")

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            if not self._queue.full():
                self._queue.put(self._read_batch())
            else:
                self._stop_event.wait(0.01)

    def sample_batch(self):
        """Return a prefetched FD batch, shape ``(B, D, F)`` complex64."""
        return self._queue.get()

    @torch.no_grad()
    def forward(self):
        """Return ``(noise_fd, noise_target)`` — same interface as MemmapNoiseSampler."""
        return self.sample_batch(), self.noise_target

    def shutdown(self):
        """Stop prefetch thread and release memmaps."""
        self._stop_event.set()
        self._thread.join()
        for mm in self.mmaps:
            del mm

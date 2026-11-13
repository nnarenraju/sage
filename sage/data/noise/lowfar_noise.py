"""
Hard noise windows as per-detector start times: the persisted
:class:`StartTimeDataset`, and :class:`_MiningReader`, which reads the
windows at given starts for :mod:`sage.data.noise.cma_mae_mining`.
"""

import json

import numpy as np
import torch

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import tqdm

from sage.config.machine import segments_file

from ._pycbc_lazy import dyn_range_fac


class StartTimeDataset:
    """
    Noise windows given by per-detector start indices, with scores.

    Each of the N windows has D absolute memmap start indices, segment
    ids (``segment_index`` of the sidecar, for the ASD look-up) and run
    ids. Saved as ``.npz`` with a ``.json`` sidecar for the strings.

    Parameters
    ----------
    detectors : list of str
        Detector names, in channel order.
    start_indices : numpy.ndarray, shape ``(N, D)``
        Absolute memmap start index of each window and detector.
    segment_indices : numpy.ndarray, shape ``(N, D)``
        Segment id of each window and detector.
    gps_times : numpy.ndarray, shape ``(N,)``
        GPS start time of the detector-0 window, for reference.
    scores : numpy.ndarray, shape ``(N,)``
        Ranking statistic of each window.
    bin_files : list of str
        Paths of the ``.bin`` noise files.
    sample_rate : float
        Sample rate (Hz).
    seq_len : int
        Window length in samples, padding included.
    run_indices : numpy.ndarray, shape ``(N, D)``, or None
        Pooled run of each window and detector; ``None`` is run 0.
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
        run_indices=None,
    ):
        self.detectors = list(detectors)
        self.start_indices = np.asarray(start_indices, dtype=np.int64)
        self.segment_indices = np.asarray(segment_indices, dtype=np.int64)
        self.run_indices = (np.zeros_like(self.start_indices) if run_indices is None
                            else np.asarray(run_indices, dtype=np.int64))
        self.gps_times = np.asarray(gps_times, dtype=np.float64)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.bin_files = [str(f) for f in bin_files]
        self.sample_rate = float(sample_rate)
        self.seq_len = int(seq_len)

    def save(self, path):
        """
        Save as ``.npz`` with a ``.json`` sidecar for the string fields.

        Parameters
        ----------
        path : str or Path
            Output path; its suffix is replaced by ``.npz``.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        np.savez_compressed(
            str(path),
            start_indices=self.start_indices,
            segment_indices=self.segment_indices,
            run_indices=self.run_indices,
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
        """
        Load a dataset written by :meth:`save`.

        Parameters
        ----------
        path : str or Path
            The ``.npz`` path; its suffix is replaced by ``.npz``.

        Returns
        -------
        StartTimeDataset
            The loaded dataset; run 0 if the file has no run ids.
        """
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
            run_indices=d["run_indices"] if "run_indices" in d else None,
            gps_times=d["gps_times"],
            scores=d["scores"],
            bin_files=meta["bin_files"],
            sample_rate=float(d["sample_rate"][0]),
            seq_len=int(d["seq_len"][0]),
        )

    def filter(self, min_score):
        """
        Windows with score >= ``min_score``.

        Parameters
        ----------
        min_score : float
            Lowest score kept.

        Returns
        -------
        StartTimeDataset
            New dataset.
        """
        mask = self.scores >= min_score
        return StartTimeDataset(
            detectors=self.detectors,
            start_indices=self.start_indices[mask],
            segment_indices=self.segment_indices[mask],
            run_indices=self.run_indices[mask],
            gps_times=self.gps_times[mask],
            scores=self.scores[mask],
            bin_files=self.bin_files,
            sample_rate=self.sample_rate,
            seq_len=self.seq_len,
        )

    def merge(self, other):
        """
        Concatenate with another dataset of the same detectors.

        The result keeps this dataset's ``bin_files``, sample rate and
        window length.

        Parameters
        ----------
        other : StartTimeDataset
            Dataset appended after this one.

        Returns
        -------
        StartTimeDataset
            New dataset.

        Raises
        ------
        AssertionError
            If the detector lists differ.
        """
        assert self.detectors == other.detectors, "detector lists must match"
        return StartTimeDataset(
            detectors=self.detectors,
            start_indices=np.concatenate([self.start_indices, other.start_indices], axis=0),
            segment_indices=np.concatenate([self.segment_indices, other.segment_indices], axis=0),
            run_indices=np.concatenate([self.run_indices, other.run_indices], axis=0),
            gps_times=np.concatenate([self.gps_times, other.gps_times]),
            scores=np.concatenate([self.scores, other.scores]),
            bin_files=self.bin_files,
            sample_rate=self.sample_rate,
            seq_len=self.seq_len,
        )

    def dedup(self):
        """
        Dataset with each window once, at its highest score.

        A window is identified by its per-detector run and start
        indices.
        The kept windows stay in their original order.

        Returns
        -------
        StartTimeDataset
            New dataset, or this one if it holds at most one window.
        """
        if len(self) <= 1:
            return self
        # Score-descending, so np.unique's first occurrence is the best
        order = np.argsort(-self.scores, kind="stable")
        key = np.concatenate([self.run_indices, self.start_indices], axis=1)
        _, first = np.unique(key[order], axis=0, return_index=True)
        keep = np.sort(order[first])            # original order
        return StartTimeDataset(
            detectors=self.detectors,
            start_indices=self.start_indices[keep],
            segment_indices=self.segment_indices[keep],
            run_indices=self.run_indices[keep],
            gps_times=self.gps_times[keep],
            scores=self.scores[keep],
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


class _MiningReader:
    """
    Reads noise windows at given per-detector memmap start indices.

    Shares the memmaps and segment tables of a ``MemmapNoiseSampler``
    and has its own NumPy RNG. The GPS and segment-bound look-ups cover
    run 0 only.

    Parameters
    ----------
    noise_sampler : MemmapNoiseSampler
        Sampler whose memmaps and segment tables are used.
    seed : int or None
        Seed of the reader's RNG.
    postprocess : bool
        Apply the sampler's ``postprocess_fn`` (recolour) to the windows
        read. Mining and re-evaluation use ``False``, so a window's
        score does not depend on a random ASD draw.
    """

    def __init__(self, noise_sampler, seed=None, postprocess=False):
        self.mmaps = noise_sampler.mmaps                # [d][run]
        self.seg_index = noise_sampler.seg_index        # pooled, per detector
        self.segment_probs = noise_sampler.segment_probs
        self.seq_len = noise_sampler.seq_len
        self.n_detectors = noise_sampler.n_detectors
        self.n_runs = len(self.mmaps[0]) if self.mmaps else 1
        self.device = noise_sampler.device
        self.postprocess_fn = noise_sampler.postprocess_fn if postprocess else None
        self.rng = np.random.default_rng(seed)

        # Look-ups for mutate_starts, gps_from_starts and gps_range,
        # run 0 only: segment ids repeat across pooled runs. bin_files
        # is detector-major (detector d, run r at d * n_runs + r).
        self.gps_meta = []   # per detector: {segment_index -> {...}}
        for d in range(self.n_detectors):
            p = noise_sampler.bin_files[d * self.n_runs]     # detector d, run 0
            meta_path = segments_file(p)
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

        # Look-up arrays per detector, run 0 only
        self._gps_lookup = []    # for gps_from_starts
        self._seg_bounds = []    # for mutate_starts
        for d in range(self.n_detectors):
            seg_arr = self.seg_index[d]
            seg_arr = seg_arr[seg_arr["run"] == 0]      # run-0 segments only
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

    def random_starts(self, batch_size, weights=None):
        """
        Draw random noise windows.

        Parameters
        ----------
        batch_size : int
            Number of windows.
        weights : list of numpy.ndarray or None
            Per-detector segment probabilities; ``None`` uses the
            sampler's ``segment_probs``.

        Returns
        -------
        starts : numpy.ndarray, shape ``(B, D)``
            Absolute memmap start indices, int64.
        segs : numpy.ndarray, shape ``(B, D)``
            Segment ids, int64.
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

    def read_batch(self, starts, segs, runs=None):
        """
        Read noise windows and convert them to the frequency domain.

        Parameters
        ----------
        starts : numpy.ndarray, shape ``(B, D)``
            Absolute memmap start indices.
        segs : numpy.ndarray, shape ``(B, D)``
            Segment ids.
        runs : numpy.ndarray, shape ``(B, D)``, or None
            Pooled run of each window (indexes ``mmaps[d]``); ``None``
            is run 0.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)``
            ``postprocess_fn`` output, or the ``rfft`` (complex64) of
            the windows, on ``self.device``.
        """
        B = len(starts)
        D = self.n_detectors
        runs = np.zeros_like(np.asarray(starts)) if runs is None \
            else np.asarray(runs)
        batch_td = torch.empty(
            (B, D, self.seq_len), dtype=torch.float32, device=self.device
        )

        def read_det(d):
            """Read the windows of detector ``d``, rescaled."""
            arr = np.empty((B, self.seq_len), dtype=np.float32)
            for i in range(B):
                r = int(runs[i, d])          # the window's run memmap
                s = int(starts[i, d])
                arr[i] = self.mmaps[d][r][s : s + self.seq_len].astype(np.float32)
            arr /= dyn_range_fac()
            return arr

        with ThreadPoolExecutor(max_workers=D) as pool:
            results = list(pool.map(read_det, range(D)))

        for d, arr in enumerate(results):
            cpu_t = torch.from_numpy(arr)
            # Pinning serves async copies to CUDA only
            if torch.cuda.is_available():
                cpu_t = cpu_t.pin_memory()
            batch_td[:, d, :].copy_(cpu_t, non_blocking=True)

        segment_ids = torch.from_numpy(segs.astype(np.int64))   # (B, D) CPU
        run_ids = torch.from_numpy(runs.astype(np.int64))       # (B, D) CPU

        if self.postprocess_fn is not None:
            # Recolour keys its segment ASDs by (run_id, segment_index)
            return self.postprocess_fn(batch_td, segment_ids, run_ids)
        return torch.fft.rfft(batch_td, dim=-1, norm="forward")

    def mutate_starts(self, starts, segs, sigma_samples):
        """
        Shift start indices by a Gaussian step, clamped to the segment.

        One step per window is shared by all detectors; segment bounds
        come from run 0.

        Parameters
        ----------
        starts : numpy.ndarray, shape ``(B, D)``
            Absolute memmap start indices.
        segs : numpy.ndarray, shape ``(B, D)``
            Segment ids.
        sigma_samples : int
            Standard deviation of the step, in samples.

        Returns
        -------
        new_starts : numpy.ndarray, shape ``(B, D)``
            Shifted start indices.
        segs : numpy.ndarray, shape ``(B, D)``
            Copy of the segment ids.
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

    def gps_from_starts(self, starts, segs):
        """
        GPS start time of each window in detector 0 (run 0).

        Parameters
        ----------
        starts : numpy.ndarray, shape ``(B, D)``
            Absolute memmap start indices.
        segs : numpy.ndarray, shape ``(B, D)``
            Segment ids.

        Returns
        -------
        numpy.ndarray, shape ``(B,)``
            GPS times, float64.
        """
        lk = self._gps_lookup[0]
        itp = lk["id_to_pos"]
        clamped = np.clip(segs[:, 0], 0, len(itp) - 1)
        pos = itp[clamped]
        return lk["gps_starts"][pos] + (starts[:, 0].astype(np.float64) - lk["ssi"][pos]) / lk["sr"][pos]

    def gps_range(self):
        """
        Earliest and latest segment GPS start, all detectors, run 0.

        Returns
        -------
        tuple of float
            ``(t_min, t_max)``.
        """
        t_min, t_max = float("inf"), float("-inf")
        for d in range(self.n_detectors):
            for m in self.gps_meta[d].values():
                t_min = min(t_min, m["gps_start"])
                t_max = max(t_max, m["gps_start"])
        return t_min, t_max

    @staticmethod
    def _score_percentile_str(scores):
        """
        The 50/75/90/95/99th percentiles of ``scores`` as text.

        Parameters
        ----------
        scores : array-like
            Scores.

        Returns
        -------
        str
            Percentiles, or ``"n/a"`` for no scores.
        """
        if len(scores) == 0:
            return "n/a"
        pcts = np.percentile(scores, [50, 75, 90, 95, 99])
        return "50/75/90/95/99 = " + "/".join(f"{p:.3f}" for p in pcts)


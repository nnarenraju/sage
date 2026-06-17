#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple hard-negative noise mining with CMA-MAE (pyribs).

A single, package-backed Quality-Diversity miner that runs **once per epoch
after validation**.  It searches per-detector start times for noise windows that
fool the current model (high ranking statistic) and keeps as many *diverse* hard
windows as possible -- diversity measured in the model's own attention embedding
space, so it never collapses onto one glitch type.

ALL of the CMA / archive / QD machinery is pyribs (Fontaine & Nikolaidis), the
reference implementation of CMA-MAE + CVT archives -- nothing here is a
hand-rolled optimiser:

  * ``ribs.archives.CVTArchive``        -- CVT archive over the embedding
                                           (``learning_rate`` = alpha, with a
                                           ``threshold_min`` floor => CMA-MAE).
  * ``ribs.emitters.EvolutionStrategyEmitter`` (``ranker='2imp'`` default)
                                        -- CMA-ES with improvement ranking.
  * ``ribs.schedulers.Scheduler``       -- the ask / tell loop.

What lives here is only the gravitational-wave glue:

  * genotype (a bounded vector in ``[0, 1]^D``, D = number of detectors) ->
    per-detector ``(start_index, segment_id)`` via the segment valid-position
    table (clip, no logit map);
  * a black-box ``evaluate_fn(starts, segs) -> (scores, embeddings)`` seam that
    the training hook fills with "read noise -> preprocess -> model -> tap the
    attention embedding"; unit tests pass a trivial stand-in;
  * accumulation of every window scoring ``>= keep_threshold`` into a
    :class:`~sage.data.noise.lowfar_noise.StartTimeDataset` (start times only,
    not strain), which the noise sampler later replays with probability ``p``.

The embedding is high-dimensional, so it is reduced with PCA (a handful of dims)
before the CVT -- both the PCA and the CVT centroids are re-fit at the start of
each ``mine`` call, which naturally tracks the model's drifting embedding across
epochs (AURORA-style refresh).  Cross-epoch continuity comes from seeding the
search at known-hard start times via ``seed_dataset``; the accumulated dataset is
what persists and grows.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from sage.core.pipeline import GWBatch, Grid, ProcessingState
from .lowfar_noise import StartTimeDataset


def make_miner_preprocessor(processor, signal_sampler=None):
    """Build ``preprocess_fn(noise_fd) -> net_input`` mirroring the training
    noise pipeline (whitening -> IFFT -> multirate, or coarse-FD selection),
    so the miner scores noise shaped exactly like what the model sees in
    training.  ``signal_sampler`` (when given) supplies the multibanding state.
    """
    if signal_sampler is not None:
        initial_state = getattr(
            signal_sampler, "output_state", ProcessingState(Grid.FD_UNIFORM)
        )
        selector = getattr(signal_sampler, "selector", None)
        freqs = selector.coarse_freqs if selector is not None else None
        coarse_idx = selector.coarse_indices if selector is not None else None
    else:
        initial_state, selector, freqs, coarse_idx = (
            ProcessingState(Grid.FD_UNIFORM), None, None, None
        )

    def preprocess_fn(noise_fd):
        if selector is not None:
            noise_fd = selector(noise_fd)
        batch = GWBatch(noise_fd, state=initial_state, freqs=freqs,
                        coarse_indices=coarse_idx)
        return processor(batch).to_network_input()

    return preprocess_fn


class _StartTimeCodec:
    """Bounded genotype ``[0, 1]^D`` <-> per-detector ``(start, segment)``.

    Each genotype component is a fraction in ``[0, 1]`` (the emitter is bounded,
    so we just clip -- no sigmoid/logit map).  The fraction indexes linearly into
    the detector's valid window-start positions, respecting segment boundaries.

    Parameters
    ----------
    seg_index : list of structured np.ndarray
        One per detector; fields ``['idx', 'start', 'end', 'nsamples']``.
    seq_len : int
        Window length in samples (a start is valid only if the whole window fits
        inside its segment).
    """

    def __init__(self, seg_index, seq_len):
        self.D = len(seg_index)
        self.seq_len = int(seq_len)
        self.cum_valid, self.abs_starts, self.seg_ids, self.valid_per_seg, self.N = (
            [], [], [], [], []
        )
        for seg_arr in seg_index:
            valid = np.maximum(0, seg_arr["nsamples"].astype(np.int64) - self.seq_len)
            cum = np.concatenate([[0], np.cumsum(valid)])
            self.cum_valid.append(cum)
            self.abs_starts.append(seg_arr["start"].astype(np.int64))
            self.seg_ids.append(seg_arr["idx"].astype(np.int64))
            self.valid_per_seg.append(valid)
            self.N.append(int(cum[-1]))

    def decode(self, genotypes):
        """(B, D) fractions -> (starts (B, D) int64, segs (B, D) int64)."""
        g = np.clip(np.asarray(genotypes, dtype=np.float64), 0.0, 1.0)
        B = g.shape[0]
        starts = np.zeros((B, self.D), dtype=np.int64)
        segs = np.zeros((B, self.D), dtype=np.int64)
        for d in range(self.D):
            if self.N[d] <= 0:                      # detector with no valid window
                continue
            lin = np.clip((g[:, d] * self.N[d]).astype(np.int64), 0, self.N[d] - 1)
            arr = np.clip(
                np.searchsorted(self.cum_valid[d], lin, side="right") - 1,
                0, len(self.abs_starts[d]) - 1,
            )
            off = np.clip(lin - self.cum_valid[d][arr], 0,
                          np.maximum(0, self.valid_per_seg[d][arr] - 1))
            starts[:, d] = self.abs_starts[d][arr] + off
            segs[:, d] = self.seg_ids[d][arr]
        return starts, segs

    def encode(self, starts, segs):
        """(B, D) (starts, segs) -> (B, D) genotype fractions in [0, 1]."""
        starts = np.asarray(starts, dtype=np.int64)
        segs = np.asarray(segs, dtype=np.int64)
        B = starts.shape[0]
        g = np.zeros((B, self.D), dtype=np.float64)
        for d in range(self.D):
            if self.N[d] <= 0:
                continue
            id_to_pos = {int(s): i for i, s in enumerate(self.seg_ids[d])}
            arr = np.array([id_to_pos.get(int(s), 0) for s in segs[:, d]], dtype=np.int64)
            off = starts[:, d] - self.abs_starts[d][arr]
            lin = self.cum_valid[d][arr] + off
            # map to the *centre* of the linear bin so decode (which floors
            # g*N) recovers exactly this position -- makes decode∘encode stable.
            g[:, d] = np.clip((lin.astype(np.float64) + 0.5) / max(self.N[d], 1),
                              0.0, 1.0)
        return g


class CMAMAEMiner:
    """CMA-MAE hard-negative noise miner (pyribs); one ``mine`` call per epoch.

    Parameters
    ----------
    detectors : list[str]
        Detector names; ``D = len(detectors)`` sets the search dimension.
    seg_index : list of structured np.ndarray
        Per-detector segment tables (fields ``idx/start/end/nsamples``).
    seq_len : int
        Window length in samples.
    bin_files, sample_rate :
        Passed through to the emitted :class:`StartTimeDataset`.
    keep_threshold : float
        Windows scoring ``>=`` this are saved to the dataset (what we mine).
    descriptor_dim : int
        PCA-reduced embedding dimension used as the QD measure space.
    n_cells : int
        Number of CVT cells (diversity niches).
    learning_rate : float
        CMA-MAE archive learning rate alpha (1.0 = CMA-ME, < 1 = CMA-MAE).
    threshold_min : float
        Archive floor for empty cells (drives cell-filling; keep below typical
        scores so exploration can enter new niches).
    n_emitters, emitter_batch_size, sigma0 :
        CMA-ES emitter ensemble settings.
    n_warmup : int
        Random windows evaluated up front to fit the PCA + CVT centroids
        (>= ``n_cells``).
    seed : int or None
    """

    def __init__(
        self,
        detectors,
        seg_index,
        seq_len,
        bin_files,
        sample_rate,
        keep_threshold=5.0,
        descriptor_dim=8,
        n_cells=1024,
        learning_rate=0.1,
        threshold_min=0.0,
        n_emitters=1,
        emitter_batch_size=36,
        sigma0=0.2,
        n_warmup=2048,
        seed=None,
    ):
        self.detectors = list(detectors)
        self.D = len(self.detectors)
        self.codec = _StartTimeCodec(seg_index, seq_len)
        self.seq_len = int(seq_len)
        self.bin_files = list(bin_files)
        self.sample_rate = float(sample_rate)

        self.keep_threshold = float(keep_threshold)
        self.descriptor_dim = int(descriptor_dim)
        self.n_cells = int(n_cells)
        self.learning_rate = float(learning_rate)
        self.threshold_min = float(threshold_min)
        self.n_emitters = int(n_emitters)
        self.emitter_batch_size = int(emitter_batch_size)
        self.sigma0 = float(sigma0)
        self.n_warmup = max(int(n_warmup), self.n_cells)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _embed_measures(self, embeddings, fit=False):
        """PCA-reduce raw embeddings to ``descriptor_dim`` measures."""
        emb = np.asarray(embeddings, dtype=np.float64)
        if fit:
            dim = min(self.descriptor_dim, emb.shape[1], emb.shape[0])
            self._pca = PCA(n_components=dim, random_state=0).fit(emb)
        return self._pca.transform(emb)

    def _seed_genotypes(self, n, seed_dataset):
        """Mix random genotypes with ones encoded from known-hard start times."""
        g = self._rng.random((n, self.D))
        if seed_dataset is not None and len(seed_dataset) > 0:
            k = min(n // 2, len(seed_dataset))
            idx = self._rng.choice(len(seed_dataset), size=k, replace=False)
            g[:k] = self.codec.encode(
                seed_dataset.start_indices[idx], seed_dataset.segment_indices[idx]
            )
        return g

    # ------------------------------------------------------------------
    def mine(self, evaluate_fn, n_iters, seed_dataset=None):
        """Run one mining pass and return the hard windows as a StartTimeDataset.

        Parameters
        ----------
        evaluate_fn : callable
            ``(starts (B, D) int64, segs (B, D) int64) -> (scores (B,) float,
            embeddings (B, E) float)`` -- reads + preprocesses the noise windows,
            runs the model, and returns the ranking statistic and the attention
            embedding.  Pure black box: the miner never touches the model.
        n_iters : int
            Number of ask/tell generations after warmup.
        seed_dataset : StartTimeDataset or None
            Known-hard windows (e.g. last epoch's) to seed the search with.
        """
        kept_starts, kept_segs, kept_scores = [], [], []

        def _collect(starts, segs, scores):
            m = scores >= self.keep_threshold
            if m.any():
                kept_starts.append(starts[m]); kept_segs.append(segs[m])
                kept_scores.append(scores[m])

        # ── Warmup: evaluate random/seeded windows to fit PCA + CVT centroids ──
        g0 = self._seed_genotypes(self.n_warmup, seed_dataset)
        s0, seg0 = self.codec.decode(g0)
        sc0, emb0 = evaluate_fn(s0, seg0)
        sc0 = np.asarray(sc0, dtype=np.float64).reshape(-1)
        _collect(s0, seg0, sc0)

        meas0 = self._embed_measures(emb0, fit=True)
        dim = meas0.shape[1]
        n_cells = min(self.n_cells, meas0.shape[0])
        centroids = KMeans(n_clusters=n_cells, n_init=3, random_state=0).fit(
            meas0
        ).cluster_centers_
        span = meas0.max(0) - meas0.min(0)
        ranges = list(zip(meas0.min(0) - 0.1 * span - 1e-6,
                          meas0.max(0) + 0.1 * span + 1e-6))

        # ── pyribs CMA-MAE: CVT archive + CMA-ES emitters + scheduler ──────────
        archive = CVTArchive(
            solution_dim=self.D,
            centroids=centroids,
            ranges=ranges,
            learning_rate=self.learning_rate,
            threshold_min=self.threshold_min,
            seed=self.seed,
        )
        # seed each emitter at a strong warmup genotype for fast lock-on
        best = g0[int(np.argmax(sc0))]
        emitters = [
            EvolutionStrategyEmitter(
                archive,
                x0=best,
                sigma0=self.sigma0,
                bounds=[(0.0, 1.0)] * self.D,
                batch_size=self.emitter_batch_size,
                seed=None if self.seed is None else self.seed + i,
            )
            for i in range(self.n_emitters)
        ]
        scheduler = Scheduler(archive, emitters)

        # ── ask / evaluate / tell ──────────────────────────────────────────────
        for _ in range(int(n_iters)):
            sols = scheduler.ask()                       # (n, D)
            starts, segs = self.codec.decode(sols)
            scores, emb = evaluate_fn(starts, segs)
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            measures = self._embed_measures(emb, fit=False)
            scheduler.tell(scores, measures)
            _collect(starts, segs, scores)

        return self._build_dataset(kept_starts, kept_segs, kept_scores)

    # ------------------------------------------------------------------
    def _build_dataset(self, starts_l, segs_l, scores_l):
        if not starts_l:
            empty = np.zeros((0, self.D), dtype=np.int64)
            return StartTimeDataset(
                self.detectors, empty, empty, np.zeros(0), np.zeros(0, np.float32),
                self.bin_files, self.sample_rate, self.seq_len,
            )
        starts = np.concatenate(starts_l, 0)
        segs = np.concatenate(segs_l, 0)
        scores = np.concatenate(scores_l, 0).astype(np.float32)
        # de-duplicate identical (start, seg) windows, keeping the highest score
        key = np.concatenate([starts, segs], axis=1)
        order = np.argsort(-scores)
        _, uniq = np.unique(key[order], axis=0, return_index=True)
        sel = order[uniq]
        # gps_times is reference-only metadata (sampler keys off start/segment);
        # store detector-0 start as seconds since the file origin.
        gps = starts[sel, 0].astype(np.float64) / self.sample_rate
        return StartTimeDataset(
            self.detectors, starts[sel], segs[sel], gps, scores[sel],
            self.bin_files, self.sample_rate, self.seq_len,
        )

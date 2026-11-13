#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hard-negative noise mining with CMA-MAE (pyribs): per-detector start
times encoded as bounded genotypes, scored by a model, and stored in a
:class:`~sage.data.noise.hard_bank.HardMiningBank`.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from sage.core.pipeline import GWBatch


def make_miner_preprocessor(processor):
    """
    Build a function mapping an FD noise batch to the network input.

    Parameters
    ----------
    processor : callable
        Training data processor; batches enter in its input state.

    Returns
    -------
    callable
        ``preprocess_fn(noise_fd) -> network input``.
    """
    def preprocess_fn(noise_fd):
        """Run ``noise_fd`` through the processor."""
        return processor(GWBatch(noise_fd, state=processor.input_state)).to_network_input()

    return preprocess_fn


class _StartTimeCodec:
    """
    Map between genotypes in ``[0, 1]^D`` and per-detector windows.

    Each component indexes linearly into the first
    ``nsamples - seq_len`` start offsets of each segment of its
    detector; segments with ``segment_probs`` 0 have none.

    Parameters
    ----------
    seg_index : list of numpy.ndarray
        Per-detector segment tables (fields ``idx``, ``start``,
        ``nsamples`` and optionally ``run``).
    seq_len : int
        Window length in samples.
    segment_probs : list of numpy.ndarray
        Per-detector sampling probability of each segment.
    """

    def __init__(self, seg_index, seq_len, segment_probs):
        self.D = len(seg_index)
        self.seq_len = int(seq_len)
        (self.cum_valid, self.abs_starts, self.seg_ids, self.run_ids,
         self.valid_per_seg, self.N) = ([], [], [], [], [], [])
        for seg_arr, probs in zip(seg_index, segment_probs):
            valid = np.maximum(0, seg_arr["nsamples"].astype(np.int64) - self.seq_len)
            valid[probs == 0] = 0
            cum = np.concatenate([[0], np.cumsum(valid)])
            self.cum_valid.append(cum)
            self.abs_starts.append(seg_arr["start"].astype(np.int64))
            self.seg_ids.append(seg_arr["idx"].astype(np.int64))
            # Segment ids repeat across pooled runs, so each carries its
            # run id; a table without a run field is run 0
            self.run_ids.append(
                seg_arr["run"].astype(np.int64) if "run" in seg_arr.dtype.names
                else np.zeros(len(seg_arr), dtype=np.int64)
            )
            self.valid_per_seg.append(valid)
            self.N.append(int(cum[-1]))

    def decode(self, genotypes):
        """
        Windows of the given genotypes.

        Parameters
        ----------
        genotypes : array-like, shape ``(B, D)``
            Fractions, clipped to ``[0, 1]``.

        Returns
        -------
        starts, segs, runs : numpy.ndarray, shape ``(B, D)``
            Absolute start indices, segment ids and run ids, int64; zero
            for a detector without valid starts.
        """
        g = np.clip(np.asarray(genotypes, dtype=np.float64), 0.0, 1.0)
        B = g.shape[0]
        starts = np.zeros((B, self.D), dtype=np.int64)
        segs = np.zeros((B, self.D), dtype=np.int64)
        runs = np.zeros((B, self.D), dtype=np.int64)
        for d in range(self.D):
            if self.N[d] <= 0:                      # no valid start
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
            runs[:, d] = self.run_ids[d][arr]
        return starts, segs, runs

    def encode(self, starts, segs, runs):
        """
        Genotypes of the given windows.

        A ``(run, segment)`` not in the table maps to the first segment.

        Parameters
        ----------
        starts, segs, runs : array-like, shape ``(B, D)``
            Absolute start indices, segment ids and run ids.

        Returns
        -------
        numpy.ndarray, shape ``(B, D)``
            Fractions in ``[0, 1]``.
        """
        starts = np.asarray(starts, dtype=np.int64)
        segs = np.asarray(segs, dtype=np.int64)
        runs = np.asarray(runs, dtype=np.int64)
        B = starts.shape[0]
        g = np.zeros((B, self.D), dtype=np.float64)
        for d in range(self.D):
            if self.N[d] <= 0:
                continue
            # Segment ids repeat across runs: key by (run, segment)
            key_to_pos = {(int(r), int(s)): i for i, (r, s)
                          in enumerate(zip(self.run_ids[d], self.seg_ids[d]))}
            arr = np.array(
                [key_to_pos.get((int(runs[b, d]), int(segs[b, d])), 0)
                 for b in range(B)],
                dtype=np.int64,
            )
            off = starts[:, d] - self.abs_starts[d][arr]
            lin = self.cum_valid[d][arr] + off
            # Centre of the linear bin, so decode (which floors g * N)
            # recovers this position
            g[:, d] = np.clip((lin.astype(np.float64) + 0.5) / max(self.N[d], 1),
                              0.0, 1.0)
        return g


class CMAMAEMiner:
    """
    CMA-MAE hard-negative miner storing its results in a hard bank.

    A cold round fits a new IncrementalPCA of the model embeddings; warm
    rounds update the stored PCA, seed the search from banked windows
    and add a novelty bonus for embeddings far from the bank.

    Parameters
    ----------
    detectors : list[str]
        Detector names; their number is the search dimension.
    seg_index : list of numpy.ndarray
        Per-detector segment tables of the noise sampler.
    segment_probs : list of numpy.ndarray
        Per-detector sampling probability of each segment; segments
        with probability 0 are never mined.
    seq_len : int
        Window length in samples.
    bank : HardMiningBank
        Store of start times, embeddings, PCA and re-evaluations.
    keep_threshold : float or None
        Windows with raw ``evaluate_fn`` score >= this are kept;
        ``None`` keeps every window.
    descriptor_dim : int
        PCA-reduced embedding width; must match the bank's.
    n_cells : int
        Number of CVT archive cells, at most the number of measures.
    learning_rate : float
        Archive learning rate (pyribs); below 1 is CMA-MAE, 1 CMA-ME.
    threshold_min : float
        Archive minimum threshold, lowered below the warmup scores.
    n_emitters : int
        Number of CMA-ES emitters.
    emitter_batch_size : int
        Solutions per emitter and iteration.
    sigma0 : float
        Initial CMA-ES step size.
    n_warmup : int
        Windows scored before the search; at least ``n_cells``.
    novelty_weight : float
        Weight of the novelty bonus in the search objective; the keep
        gate uses the raw score only.
    seed : int or None
        Base seed of each round's random streams.
    """

    def __init__(
        self,
        detectors,
        seg_index,
        segment_probs,
        seq_len,
        bank,
        keep_threshold=None,
        descriptor_dim=8,
        n_cells=1024,
        learning_rate=0.1,
        threshold_min=0.0,
        n_emitters=1,
        emitter_batch_size=36,
        sigma0=0.2,
        n_warmup=2048,
        novelty_weight=1.0,
        seed=None,
    ):
        self.detectors = list(detectors)
        self.D = len(self.detectors)
        self.codec = _StartTimeCodec(seg_index, seq_len, segment_probs)
        self.seq_len = int(seq_len)
        self.bank = bank
        self.keep_threshold = (
            float("-inf") if keep_threshold is None else float(keep_threshold)
        )
        self.descriptor_dim = int(descriptor_dim)
        self.n_cells = int(n_cells)
        self.learning_rate = float(learning_rate)
        self.threshold_min = float(threshold_min)
        self.n_emitters = int(n_emitters)
        self.emitter_batch_size = int(emitter_batch_size)
        self.sigma0 = float(sigma0)
        self.n_warmup = max(int(n_warmup), self.n_cells)
        self.novelty_weight = float(novelty_weight)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _seed_genotypes(self, n):
        """
        Random genotypes; for a warm bank up to half are banked.

        Parameters
        ----------
        n : int
            Number of genotypes.

        Returns
        -------
        numpy.ndarray, shape ``(n, D)``
            Genotypes.
        """
        g = self._rng.random((n, self.D))
        if not self.bank.is_cold:
            k = min(n // 2, self.bank.n_starts)
            if k > 0:
                starts, segs, runs = self.bank.sample_starts(k, self._rng)
                g[:k] = self.codec.encode(starts, segs, runs)
        return g

    @staticmethod
    def _novelty(measures, covered):
        """
        Distance of each measure row to the nearest ``covered`` row.

        Parameters
        ----------
        measures : numpy.ndarray, shape ``(B, P)``
            Measures to score.
        covered : numpy.ndarray, shape ``(M, P)``, or None
            Covered set; ``None`` or empty gives zeros.

        Returns
        -------
        numpy.ndarray, shape ``(B,)``
            Euclidean distances.
        """
        if covered is None or len(covered) == 0:
            return np.zeros(len(measures), dtype=np.float64)
        d2 = ((measures[:, None, :] - covered[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2.min(1))

    def mine(self, evaluate_fn, n_iters, epoch):
        """
        Run one mining round and append the kept windows to the bank.

        Parameters
        ----------
        evaluate_fn : callable
            ``evaluate_fn(starts, segs, runs) -> (scores, embeddings)``.
        n_iters : int
            Ask-tell iterations after the warmup.
        epoch : int
            Training epoch; sets the round seed and tags the bank rows.

        Returns
        -------
        dict
            Round statistics for logging.

        Raises
        ------
        AssertionError
            If a cold round's embedding width is below
            ``descriptor_dim``.
        """
        # Seed per round: the same epoch reproduces, epochs differ
        round_seed = None if self.seed is None else self.seed + epoch * 100003
        self._rng = np.random.default_rng(round_seed)

        # The stored PCA is reused even if no start was kept yet
        ipca = self.bank.load_pca()
        cold = ipca is None
        covered = (self.bank.read_embeddings()[0]
                   if self.bank.n_embeddings > 0 else None)         # (M, P) or None

        kept_s, kept_g, kept_r, kept_sc, kept_m = [], [], [], [], []

        def _collect(starts, segs, runs, scores, measures):
            """Store the windows scoring >= ``keep_threshold``."""
            m = scores >= self.keep_threshold
            if m.any():
                kept_s.append(starts[m]); kept_g.append(segs[m]); kept_r.append(runs[m])
                kept_sc.append(scores[m]); kept_m.append(measures[m])

        # Warmup: score random and seeded windows, update the PCA
        g0 = self._seed_genotypes(self.n_warmup)
        s0, seg0, run0 = self.codec.decode(g0)
        sc0, emb0 = evaluate_fn(s0, seg0, run0)
        sc0 = np.asarray(sc0, np.float64).reshape(-1)
        emb0 = np.asarray(emb0, np.float64)

        if ipca is None:
            # The bank stores descriptor_dim components, so the model
            # embedding must be at least that wide
            assert emb0.shape[1] >= self.descriptor_dim, (
                f"model embedding width {emb0.shape[1]} < descriptor_dim "
                f"{self.descriptor_dim}; lower descriptor_dim"
            )
            ipca = IncrementalPCA(n_components=self.descriptor_dim)
        ipca.partial_fit(emb0)                       # update, not refit
        meas0 = ipca.transform(emb0)
        _collect(s0, seg0, run0, sc0, meas0)

        # CVT centroids from the warmup measures and the bank embeddings
        cset = meas0 if (covered is None or len(covered) == 0) else \
            np.concatenate([covered[:, :meas0.shape[1]], meas0], 0)
        n_cells = min(self.n_cells, len(cset))
        centroids = KMeans(n_clusters=n_cells, n_init=3, random_state=0).fit(
            cset
        ).cluster_centers_
        span = cset.max(0) - cset.min(0)
        ranges = list(zip(cset.min(0) - 0.1 * span - 1e-6,
                          cset.max(0) + 0.1 * span + 1e-6))

        # Cell threshold below the warmup scores, so the archive fills
        # from the first generation; an empty archive fails the CMA-ES
        # restart (sample_elites). The keep gate is separate.
        thr_min = self.threshold_min
        finite = sc0[np.isfinite(sc0)]
        if finite.size:
            thr_min = min(thr_min, float(finite.min()) - 1.0)
        archive = CVTArchive(
            solution_dim=self.D, centroids=centroids, ranges=ranges,
            learning_rate=self.learning_rate, threshold_min=thr_min,
            seed=round_seed,
        )
        best = g0[int(np.argmax(sc0))]
        emitters = [
            EvolutionStrategyEmitter(
                archive, x0=best, sigma0=self.sigma0,
                bounds=[(0.0, 1.0)] * self.D, batch_size=self.emitter_batch_size,
                seed=None if round_seed is None else round_seed + i,
            )
            for i in range(self.n_emitters)
        ]
        scheduler = Scheduler(archive, emitters)

        # Novelty is measured against the bank embeddings of this round
        cov = None if (covered is None or len(covered) == 0) else \
            covered[:, :meas0.shape[1]]

        # Ask, evaluate, tell
        for _ in range(int(n_iters)):
            sols = scheduler.ask()
            starts, segs, runs = self.codec.decode(sols)
            scores, emb = evaluate_fn(starts, segs, runs)
            scores = np.asarray(scores, np.float64).reshape(-1)
            meas = ipca.transform(np.asarray(emb, np.float64))
            # Objective: score plus novelty; keep gate: score only
            obj = scores + self.novelty_weight * self._novelty(meas, cov)
            scheduler.tell(obj, meas)
            _collect(starts, segs, runs, scores, meas)

        # Store the PCA, the kept start times and their embeddings
        self.bank.save_pca(ipca)
        n_starts = n_emb = 0
        if kept_s:
            starts = np.concatenate(kept_s, 0)
            segs = np.concatenate(kept_g, 0)
            runs = np.concatenate(kept_r, 0)
            scores = np.concatenate(kept_sc, 0)
            meas = np.concatenate(kept_m, 0)
            # One row per (run, seg, start) window, at its highest score
            key = np.concatenate([runs, segs, starts], axis=1)
            order = np.argsort(-scores)
            _, uniq = np.unique(key[order], axis=0, return_index=True)
            sel = order[uniq]
            self.bank.append_starts(starts[sel], segs[sel], runs[sel], scores[sel], epoch)
            n_starts = len(sel)
            # The bank applies its own novelty and capacity gate
            n_emb = self.bank.add_embeddings(meas[sel], scores[sel], epoch)
        return {"epoch": epoch, "cold": cold,
                "kept_starts": int(n_starts), "kept_embeddings": int(n_emb),
                "bank_starts": self.bank.n_starts,
                "bank_embeddings": self.bank.n_embeddings}

    def reevaluate(self, evaluate_fn, model_epoch, batch_size=4096):
        """
        Re-score every banked window and append the scores to the bank.

        The scores form a new ``eval_stats`` column tagged
        ``model_epoch``. An empty bank is left unchanged.

        Parameters
        ----------
        evaluate_fn : callable
            ``evaluate_fn(starts, segs, runs) -> (scores, embeddings)``.
        model_epoch : int
            Epoch of the model doing the scoring.
        batch_size : int
            Windows per ``evaluate_fn`` call.

        Returns
        -------
        dict
            Re-evaluation statistics for logging.
        """
        N = self.bank.n_starts
        if N == 0:
            return {"reeval_n": 0}
        stats = np.empty(N, dtype=np.float32)
        for sl, starts, segs, runs in self.bank.iter_starts(batch_size):
            sc, _ = evaluate_fn(starts, segs, runs)
            stats[sl] = np.asarray(sc, np.float32).reshape(-1)
        self.bank.append_eval(stats, model_epoch)
        return {"reeval_n": int(N), "model_epoch": int(model_epoch),
                "above_threshold": int((stats >= self.keep_threshold).sum())}

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hard-negative noise mining with CMA-MAE (pyribs), backed by a file-resident
:class:`~sage.data.noise.hard_bank.HardMiningBank`.

One miner, continuous across training:

  * **Cold start (once):** random per-detector start times are scored by the
    current model; a fresh IncrementalPCA + CVT centroids are fit from the
    warmup embeddings; CMA-MAE explores; hard windows (ranking stat >= keep
    threshold) and a diverse subset of their embeddings are written to the bank.
  * **Every warm round:** the persisted IncrementalPCA and the start-time /
    embedding banks are loaded from the file.  CMA-ES is seeded from known-hard
    start times *and* the CVT centroids from the accumulated embedding bank; the
    PCA is ``partial_fit`` (never refit from scratch); a **novelty bonus** drives
    the search toward embedding regions the bank does not yet cover (new glitch
    families) while the *keep* gate still requires a high raw ranking stat.
  * **Re-evaluation:** every round, the entire start-time bank is re-scored with
    the current model and the ranking stats are appended (tagged with the model
    epoch), so a family's hardness trajectory across training is recoverable.

ALL CMA / archive / QD machinery is pyribs (CMA-MAE + CVT archives).  What lives
here is only the gravitational-wave glue: the bounded genotype <-> per-detector
start-time codec, the black-box ``evaluate_fn`` seam, and the bank bookkeeping.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from sage.core.pipeline import GWBatch, Grid, ProcessingState


def make_miner_preprocessor(processor, signal_sampler=None):
    """Build ``preprocess_fn(noise_fd) -> net_input`` mirroring the training
    noise pipeline (whitening -> IFFT -> multirate, or coarse-FD selection), so
    the miner scores noise shaped exactly like what the model sees in training.
    ``signal_sampler`` (when given) supplies the multibanding state.
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
    """

    def __init__(self, seg_index, seq_len):
        self.D = len(seg_index)
        self.seq_len = int(seq_len)
        (self.cum_valid, self.abs_starts, self.seg_ids, self.run_ids,
         self.valid_per_seg, self.N) = ([], [], [], [], [], [])
        for seg_arr in seg_index:
            valid = np.maximum(0, seg_arr["nsamples"].astype(np.int64) - self.seq_len)
            cum = np.concatenate([[0], np.cumsum(valid)])
            self.cum_valid.append(cum)
            self.abs_starts.append(seg_arr["start"].astype(np.int64))
            self.seg_ids.append(seg_arr["idx"].astype(np.int64))
            # Pooled segment tables carry a run id (5a); a window's identity is
            # (run, segment, start) since segment ids collide across runs. Tolerate
            # a legacy run-less table (all run 0).
            self.run_ids.append(
                seg_arr["run"].astype(np.int64) if "run" in seg_arr.dtype.names
                else np.zeros(len(seg_arr), dtype=np.int64)
            )
            self.valid_per_seg.append(valid)
            self.N.append(int(cum[-1]))

    def decode(self, genotypes):
        """(B, D) fractions -> (starts, segs, runs), each (B, D) int64."""
        g = np.clip(np.asarray(genotypes, dtype=np.float64), 0.0, 1.0)
        B = g.shape[0]
        starts = np.zeros((B, self.D), dtype=np.int64)
        segs = np.zeros((B, self.D), dtype=np.int64)
        runs = np.zeros((B, self.D), dtype=np.int64)
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
            runs[:, d] = self.run_ids[d][arr]
        return starts, segs, runs

    def encode(self, starts, segs, runs):
        """(B, D) (starts, segs, runs) -> (B, D) genotype fractions in [0, 1]."""
        starts = np.asarray(starts, dtype=np.int64)
        segs = np.asarray(segs, dtype=np.int64)
        runs = np.asarray(runs, dtype=np.int64)
        B = starts.shape[0]
        g = np.zeros((B, self.D), dtype=np.float64)
        for d in range(self.D):
            if self.N[d] <= 0:
                continue
            # Key by (run, segment): segment ids alone collide across pooled runs.
            key_to_pos = {(int(r), int(s)): i for i, (r, s)
                          in enumerate(zip(self.run_ids[d], self.seg_ids[d]))}
            arr = np.array(
                [key_to_pos.get((int(runs[b, d]), int(segs[b, d])), 0)
                 for b in range(B)],
                dtype=np.int64,
            )
            off = starts[:, d] - self.abs_starts[d][arr]
            lin = self.cum_valid[d][arr] + off
            # map to the *centre* of the linear bin so decode (which floors
            # g*N) recovers exactly this position -- makes decode∘encode stable.
            g[:, d] = np.clip((lin.astype(np.float64) + 0.5) / max(self.N[d], 1),
                              0.0, 1.0)
        return g


class CMAMAEMiner:
    """Continuous CMA-MAE hard-negative miner backed by a :class:`HardMiningBank`.

    Parameters
    ----------
    detectors : list[str]
        Detector names; ``D = len(detectors)`` sets the search dimension.
    seg_index : list of structured np.ndarray
        Per-detector segment tables (fields ``idx/start/end/nsamples``).
    seq_len : int
        Window length in samples.
    bank : HardMiningBank
        The file-resident store (start-times, embeddings, PCA, re-eval history).
    keep_threshold : float or None
        Windows scoring ``>=`` this (in raw ``evaluate_fn`` units, i.e. the
        detection logit) are kept.  ``None`` -> ``-inf`` (keep everything).
    descriptor_dim : int
        PCA-reduced embedding width (must match the bank's ``P``).
    n_cells, learning_rate, threshold_min, n_emitters, emitter_batch_size,
    sigma0, n_warmup :
        CMA-MAE / CVT / CMA-ES settings (see pyribs).  ``learning_rate`` < 1 =>
        CMA-MAE; ``= 1`` => CMA-ME.
    novelty_weight : float
        Strength of the novelty bonus added to the optimisation objective in
        warm rounds (pushes the search toward embedding regions the bank does
        not yet cover).  The *keep* gate still uses the raw ranking stat only.
    seed : int or None
    """

    def __init__(
        self,
        detectors,
        seg_index,
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
        self.codec = _StartTimeCodec(seg_index, seq_len)
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

    # ------------------------------------------------------------------ helpers
    def _seed_genotypes(self, n):
        """Random genotypes, half replaced (warm) by ones encoded from the
        bank's known-hard start times so the search resumes near hard regions."""
        g = self._rng.random((n, self.D))
        if not self.bank.is_cold:
            k = min(n // 2, self.bank.n_starts)
            if k > 0:
                starts, segs, runs = self.bank.sample_starts(k, self._rng)
                g[:k] = self.codec.encode(starts, segs, runs)
        return g

    @staticmethod
    def _novelty(measures, covered):
        """Min distance of each measure row to the ``covered`` set (0 if empty)."""
        if covered is None or len(covered) == 0:
            return np.zeros(len(measures), dtype=np.float64)
        d2 = ((measures[:, None, :] - covered[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2.min(1))

    # ------------------------------------------------------------------ mining
    def mine(self, evaluate_fn, n_iters, epoch):
        """Run one mining round and append results to the bank.

        Returns a small stats dict for logging.
        """
        # Per-round seed: each mine round (one per scheduled epoch) explores
        # deterministically but differently, and re-mining the same epoch after a
        # restart reproduces exactly. The bank state itself resumes via its HDF5
        # file, independent of this seed.
        round_seed = None if self.seed is None else self.seed + epoch * 100003
        self._rng = np.random.default_rng(round_seed)

        # Continuous learning: load the persisted PCA + embedding bank if they
        # exist (decoupled from whether any hard starts were kept yet), so a
        # round that keeps zero starts never resets the PCA from scratch.
        ipca = self.bank.load_pca()
        cold = ipca is None
        covered = (self.bank.read_embeddings()[0]
                   if self.bank.n_embeddings > 0 else None)         # (M, P) or None

        kept_s, kept_g, kept_r, kept_sc, kept_m = [], [], [], [], []

        def _collect(starts, segs, runs, scores, measures):
            m = scores >= self.keep_threshold
            if m.any():
                kept_s.append(starts[m]); kept_g.append(segs[m]); kept_r.append(runs[m])
                kept_sc.append(scores[m]); kept_m.append(measures[m])

        # ── warmup: score random/seeded windows, (cold) fit / (warm) update PCA ─
        g0 = self._seed_genotypes(self.n_warmup)
        s0, seg0, run0 = self.codec.decode(g0)
        sc0, emb0 = evaluate_fn(s0, seg0, run0)
        sc0 = np.asarray(sc0, np.float64).reshape(-1)
        emb0 = np.asarray(emb0, np.float64)

        if ipca is None:
            # bank storage width (P) is fixed at descriptor_dim, so the realized
            # PCA must produce exactly that many components -- guard the model
            # embedding width so add_embeddings can't silently reshape-corrupt.
            assert emb0.shape[1] >= self.descriptor_dim, (
                f"model embedding width {emb0.shape[1]} < descriptor_dim "
                f"{self.descriptor_dim}; lower descriptor_dim"
            )
            ipca = IncrementalPCA(n_components=self.descriptor_dim)
        ipca.partial_fit(emb0)                       # continuous, never from scratch
        meas0 = ipca.transform(emb0)
        _collect(s0, seg0, run0, sc0, meas0)

        # ── CVT centroids: cold from warmup; warm seeded by the bank's diversity ─
        cset = meas0 if (covered is None or len(covered) == 0) else \
            np.concatenate([covered[:, :meas0.shape[1]], meas0], 0)
        n_cells = min(self.n_cells, len(cset))
        centroids = KMeans(n_clusters=n_cells, n_init=3, random_state=0).fit(
            cset
        ).cluster_centers_
        span = cset.max(0) - cset.min(0)
        ranges = list(zip(cset.min(0) - 0.1 * span - 1e-6,
                          cset.max(0) + 0.1 * span + 1e-6))

        # Floor the CMA-MAE cell threshold below the warmup score range so the
        # archive always populates from the first generations -- even when the
        # model scores low (early epochs / untrained). An empty archive crashes
        # the CMA-ES restart (sample_elites). The KEEP gate (keep_threshold on
        # the RAW score) is separate and unaffected by this.
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

        # covered set for the novelty bonus = bank embeddings (fixed this round)
        cov = None if (covered is None or len(covered) == 0) else \
            covered[:, :meas0.shape[1]]

        # ── ask / evaluate / tell ──────────────────────────────────────────────
        for _ in range(int(n_iters)):
            sols = scheduler.ask()
            starts, segs, runs = self.codec.decode(sols)
            scores, emb = evaluate_fn(starts, segs, runs)
            scores = np.asarray(scores, np.float64).reshape(-1)
            meas = ipca.transform(np.asarray(emb, np.float64))
            # objective = hardness + novelty (drives search to hard AND new);
            # keep gate below still uses the RAW score only.
            obj = scores + self.novelty_weight * self._novelty(meas, cov)
            scheduler.tell(obj, meas)
            _collect(starts, segs, runs, scores, meas)

        # ── persist: PCA, hard start-times, diverse+hard embeddings ────────────
        self.bank.save_pca(ipca)
        n_starts = n_emb = 0
        if kept_s:
            starts = np.concatenate(kept_s, 0)
            segs = np.concatenate(kept_g, 0)
            runs = np.concatenate(kept_r, 0)
            scores = np.concatenate(kept_sc, 0)
            meas = np.concatenate(kept_m, 0)
            # de-dup identical (run, seg, start) windows, keep the highest score
            key = np.concatenate([runs, segs, starts], axis=1)
            order = np.argsort(-scores)
            _, uniq = np.unique(key[order], axis=0, return_index=True)
            sel = order[uniq]
            self.bank.append_starts(starts[sel], segs[sel], runs[sel], scores[sel], epoch)
            n_starts = len(sel)
            # embeddings: bank applies its own novelty (distance) + cap gate
            n_emb = self.bank.add_embeddings(meas[sel], scores[sel], epoch)
        return {"epoch": epoch, "cold": cold,
                "kept_starts": int(n_starts), "kept_embeddings": int(n_emb),
                "bank_starts": self.bank.n_starts,
                "bank_embeddings": self.bank.n_embeddings}

    # ------------------------------------------------------------- re-evaluation
    def reevaluate(self, evaluate_fn, model_epoch, batch_size=4096):
        """Re-score the ENTIRE start-time bank with the current model and append
        the ranking stats as a new ``eval_stats`` column (tagged ``model_epoch``).

        Lets us track how each window's / family's hardness moves as training
        progresses.  No-op on a cold (empty) bank.
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

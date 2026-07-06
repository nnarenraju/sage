#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training callbacks — objects that hook into :class:`SageVanillaTraining`'s loop.

A callback can transform the per-batch context and/or run work at epoch
boundaries (e.g. hard-noise mining). Every hook is a no-op by default, so a
trainer constructed with no callbacks behaves exactly like plain vanilla
training.
"""

import numpy as np
import torch

from sage.core.config import get_data_cfg


class Callback:
    """Base training callback. Subclass and override only the hooks you need.

    Hooks
    -----
    on_sample(ctx, trainer)
        Called once per batch *after* the batch is assembled (post signal
        injection) and *before* preprocessing. ``ctx`` is a plain dict the
        trainer threads through the iteration — read/write its tensors
        (``ctx['x']``, ``ctx['targets']``). Mutate ``ctx`` in place. No shipped
        callback overrides this hook today; it is a generic extension point.
    on_epoch_end(nepoch, trainer)
        Called once after an epoch's training iterations complete (e.g. to mine
        hard noise and push it to the sampler).

    ``trainer`` is the :class:`SageVanillaTraining` instance, exposing
    ``model``, ``noise_sampler``, ``signal_sampler``, ``processor``, etc.
    """

    def on_sample(self, ctx, trainer):
        pass

    def on_epoch_end(self, nepoch, trainer):
        pass


class HardMiningCallback(Callback):
    """Continuous CMA-MAE hard-noise mining, as a training callback.

    On scheduled epochs it (1) mines hard noise with
    :class:`~sage.data.noise.cma_mae_mining.CMAMAEMiner`, appending start-times
    and a diverse embedding subset to a file-resident
    :class:`~sage.data.noise.hard_bank.HardMiningBank`; (2) re-evaluates the
    *entire* start-time bank with the current model so each window's hardness
    trajectory across training is recorded; and (3) pushes the **currently-hard**
    start-times (latest re-eval ranking stat >= keep threshold) to the noise
    sampler, which augments its random start-times with them at proportion
    ``hard_bias_prob``.  Nothing is held in RAM between rounds -- the bank is the
    single source of truth on disk; the miner learns continuously (one cold
    start, never rebuilt).

    Schedule
    --------
    ``mine_schedule`` is one argument accepting two forms (epoch indices are
    0-based, matching the training loop's ``nepoch``):
      * ``int N``        -> mine when ``(nepoch + 1) % N == 0`` i.e. at 0-based
                            epochs ``N-1, 2N-1, ...`` ("every N-th epoch"; never
                            at epoch 0).
      * ``list[int]``    -> mine exactly at those 0-based ``nepoch`` values
                            (e.g. the cosine-annealing warm-restart cycle ends).
      NOTE the two forms differ by one in framing: ``N=5`` fires at 4, 9, 14...
      whereas ``[5, 10, 15]`` fires at 5, 10, 15 -- pass the explicit list when
      you need exact epochs.

    The keep bar is ``keep_threshold_raw`` (raw detection logit) or
    ``keep_threshold_sigmoided`` (the same bar as a probability in ``(0, 1)``);
    raw wins if both are given, ``-inf`` if neither.

    pyribs / the miner are imported lazily, so importing this module — and pure
    vanilla training — never requires pyribs.  ``runs``/``detectors`` come from
    cfg; the library stays run-agnostic.
    """

    def __init__(
        self,
        bank_dir,
        mine_schedule=5,
        keep_threshold_raw=None,
        keep_threshold_sigmoided=None,
        hard_bias_prob=0.2,
        mine_iters=200,
        descriptor_dim=8,
        n_cells=1024,
        learning_rate=0.1,
        n_emitters=1,
        emitter_batch_size=36,
        n_warmup=2048,
        novelty_dist=0.1,
        max_embeddings=50_000,
        novelty_weight=1.0,
        mine_seed=None,
    ):
        self.bank_dir = str(bank_dir)
        self.mine_schedule = mine_schedule
        # How signal-like a window must look to count as "hard"; raw logit wins.
        self.keep_threshold_raw = keep_threshold_raw
        self.keep_threshold_sigmoided = keep_threshold_sigmoided
        if keep_threshold_raw is not None:
            self.keep_threshold = float(keep_threshold_raw)
        elif keep_threshold_sigmoided is not None:
            p = float(keep_threshold_sigmoided)
            if not 0.0 < p < 1.0:
                raise ValueError(
                    "keep_threshold_sigmoided must be a detection probability in "
                    f"(0, 1); got {keep_threshold_sigmoided!r}"
                )
            self.keep_threshold = float(np.log(p / (1.0 - p)))
        else:
            self.keep_threshold = float("-inf")     # keep every mined window
        self.hard_bias_prob = float(hard_bias_prob)
        self.mine_iters = int(mine_iters)
        self.descriptor_dim = int(descriptor_dim)
        self.n_cells = int(n_cells)
        self.learning_rate = float(learning_rate)
        self.n_emitters = int(n_emitters)
        self.emitter_batch_size = int(emitter_batch_size)
        self.n_warmup = int(n_warmup)
        self.novelty_dist = float(novelty_dist)
        self.max_embeddings = int(max_embeddings)
        self.novelty_weight = float(novelty_weight)
        self.mine_seed = mine_seed
        self._bank = None
        self._miner = None              # lazily built from the trainer's graph

    def _should_mine(self, nepoch):
        s = self.mine_schedule
        if isinstance(s, (list, tuple, set, np.ndarray)):
            return int(nepoch) in {int(x) for x in s}
        return (int(nepoch) + 1) % int(s) == 0

    def _lazy_init(self, trainer):
        # Local imports: only the hard-mining path needs pyribs / the miner.
        from sage.data.noise.cma_mae_mining import (
            CMAMAEMiner, make_miner_preprocessor,
        )
        from sage.data.noise.lowfar_noise import _MiningReader
        from sage.data.noise.hard_bank import HardMiningBank, default_bank_path

        ns = trainer.noise_sampler
        if not hasattr(ns, "set_hard_bank"):
            raise TypeError(
                "HardMiningCallback needs a noise sampler with set_hard_bank "
                "(e.g. MemmapNoiseSampler)."
            )
        dcfg = get_data_cfg()
        # runs / detectors are config-sourced (library stays run-agnostic).
        runs = (getattr(trainer.cfg, "train_runs", None)
                or getattr(dcfg, "train_runs", None)
                or [getattr(trainer.cfg, "run", "unknown")])
        detectors = list(trainer.cfg.detectors)
        bank_path = default_bank_path(self.bank_dir, runs, detectors)
        self._bank = HardMiningBank(
            bank_path, detectors=detectors, runs=runs, seq_len=ns.seq_len,
            sample_rate=float(dcfg.sample_rate),
            bin_files=[str(f) for f in ns.bin_files],
            descriptor_dim=self.descriptor_dim, novelty_dist=self.novelty_dist,
            max_embeddings=self.max_embeddings,
        )
        self._reader = _MiningReader(ns, seed=self.mine_seed)
        self._preprocess = make_miner_preprocessor(
            trainer.processor, trainer.signal_sampler
        )
        self._miner = CMAMAEMiner(
            detectors=detectors, seg_index=ns.seg_index, seq_len=ns.seq_len,
            bank=self._bank, keep_threshold=self.keep_threshold,
            descriptor_dim=self.descriptor_dim, n_cells=self.n_cells,
            learning_rate=self.learning_rate, n_emitters=self.n_emitters,
            emitter_batch_size=self.emitter_batch_size, n_warmup=self.n_warmup,
            novelty_weight=self.novelty_weight, seed=self.mine_seed,
        )

    # ------------------------------------------------------------------
    def on_epoch_end(self, nepoch, trainer):
        if not self._should_mine(nepoch):
            return
        if self._miner is None:
            self._lazy_init(trainer)
        self._mine(nepoch, trainer)

    def attach_for_resume(self, trainer):
        """Re-bias the sampler from the persisted bank on resume.

        The freshly-built noise sampler starts unbiased; without this, a resumed
        run would train with purely random noise until the next scheduled mine
        epoch (up to ``mine_schedule`` epochs of lost hard biasing).  Call once
        after restoring a checkpoint when ``start_epoch > 0``.  No-op if no bank
        exists yet (nothing mined before the checkpoint).
        """
        if self._miner is None:
            self._lazy_init(trainer)
        if self._bank.is_cold:
            return
        active = self._bank.active_start_indices(self.keep_threshold)
        trainer.noise_sampler.set_hard_bank(
            self._bank, active, hard_bias_prob=self.hard_bias_prob
        )
        print(
            f"[HardMining] resume: re-biased sampler from bank — "
            f"{len(active):,} active / {self._bank.n_starts:,} starts",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _build_evaluate_fn(self, trainer):
        """``(starts, segs) -> (scores, embeddings)`` via read -> model -> embed.

        The QD diversity descriptor is the model's own **frontend** embedding
        (pre-backend per-detector morphology, where glitch families stay
        separable) -- NOT the backend/ranking-stat feature, which is collapsed
        toward the detection decision.
          * ``MSCNN1D_2DResNetCBAM_HardMining`` -> set ``return_frontend_embedding``
            and read ``model.frontend_embedding`` (compile-safe; no forward hook).

        Runs the eager (compile-free, no-grad) model in chunks.  Returns
        ``(evaluate_fn, cleanup_fn)``.
        """
        eager = getattr(trainer.model, "_orig_mod", trainer.model)
        chunk = trainer.cfg.batch_size
        cleanup = lambda: None

        if hasattr(eager, "return_frontend_embedding"):
            # Frontend embedding via the model's flag (no hook -> fullgraph-safe).
            _prev = eager.return_frontend_embedding
            eager.return_frontend_embedding = True

            def cleanup(eager=eager, prev=_prev):
                eager.return_frontend_embedding = prev

            def run(net_input):
                out = eager(net_input)
                score = (out[0] if isinstance(out, tuple) else out).reshape(-1).float()
                return score, eager.frontend_embedding.float()   # (B, D*C), unit/det
        else:
            raise TypeError(
                "Hard mining needs a model exposing the FRONTEND embedding: "
                "MSCNN1D_2DResNetCBAM_HardMining (return_frontend_embedding)."
            )

        def evaluate_fn(starts, segs, runs):
            scores, embs = [], []
            for i in range(0, len(starts), chunk):
                net_input = self._preprocess(
                    self._reader.read_batch(starts[i:i + chunk], segs[i:i + chunk],
                                            runs[i:i + chunk])
                )
                score, emb = run(net_input)
                scores.append(score.cpu().numpy())
                embs.append(emb.cpu().numpy())
            return np.concatenate(scores), np.concatenate(embs, axis=0)

        return evaluate_fn, cleanup

    @torch.inference_mode()
    def _mine(self, nepoch, trainer):
        was_training = trainer.model.training
        trainer.model.eval()
        evaluate_fn, cleanup = self._build_evaluate_fn(trainer)
        try:
            # 1. mine new hard windows -> append start-times + diverse embeddings
            mstats = self._miner.mine(evaluate_fn, self.mine_iters, epoch=nepoch)
            # 2. re-score the ENTIRE bank with the current model (family drift)
            rstats = self._miner.reevaluate(evaluate_fn, model_epoch=nepoch)
        finally:
            cleanup()                                 # reset the frontend-embed flag
            if was_training:
                trainer.model.train()

        # 3. push only the CURRENTLY-hard start-times (latest re-eval >= bar);
        #    nothing is removed from the bank -- this is just a refreshed view.
        active = self._bank.active_start_indices(self.keep_threshold)
        trainer.noise_sampler.set_hard_bank(
            self._bank, active, hard_bias_prob=self.hard_bias_prob
        )
        print(
            f"[HardMining] epoch {nepoch}: mined +{mstats['kept_starts']:,} starts "
            f"(+{mstats['kept_embeddings']} emb) | bank {mstats['bank_starts']:,} "
            f"starts, {mstats['bank_embeddings']:,} emb | active "
            f"{len(active):,}/{rstats.get('reeval_n', 0):,}",
            flush=True,
        )

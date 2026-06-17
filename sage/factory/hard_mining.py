#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hard negative mining training.

``SageHardMiningTraining`` is a **drop-in alternative to**
:class:`~sage.factory.training.SageVanillaTraining`.  It runs the exact same
training loop, and additionally — once per epoch, after the epoch's training —
mines hard noise with CMA-MAE (pyribs) and replays it via the noise sampler's
hard-dataset mechanism.

Swapping is the whole point of the structure::

    trainer = SageVanillaTraining(...)        # default: random noise, no mining
    # or, to enable hard-negative mining, change one line:
    trainer = SageHardMiningTraining(...)      # same loop + per-epoch CMA-MAE mining

    for epoch in range(num_epochs):
        trainer(nepoch=epoch)                  # identical call site

When the vanilla trainer is used, **none** of the mining machinery is touched —
pyribs is not even imported (the import is local to this class), and the noise
sampler keeps sampling random windows.

Mining details
--------------
* The search variable is one start time per detector (``len(cfg.detectors)``).
* CMA-MAE (:class:`~sage.data.noise.cma_mae_mining.CMAMAEMiner`) keeps diverse
  hard windows, diversity measured in the model's own attention-pooled embedding
  (tapped from ``model.per_det_head.tc_attn`` via a forward hook).
* Every window scoring ``>= keep_threshold`` is accumulated across epochs (the
  dataset persists and grows) and pushed to the sampler with
  ``set_hard_dataset(..., hard_bias_prob=p)`` — so subsequent batches draw hard
  windows with probability ``p`` instead of random ones.
"""

import numpy as np
import torch
import torch.nn.functional as F

from sage.core.config import get_data_cfg
from .training import SageVanillaTraining


class SageHardMiningTraining(SageVanillaTraining):
    """Vanilla training + per-epoch CMA-MAE hard-noise mining.

    Accepts every :class:`SageVanillaTraining` argument, plus the mining knobs
    below.  Requires a noise sampler with ``set_hard_dataset`` (e.g.
    :class:`~sage.data.noise.real_noise.MemmapNoiseSampler`) and a model that
    exposes ``per_det_head.tc_attn`` (the consistency model).

    Parameters
    ----------
    hard_bias_prob : float
        Probability a training batch draws from the mined hard dataset.
    keep_threshold : float
        Ranking-statistic bar; windows ``>=`` this are kept.
    warmup_epochs : int
        Train this many epochs on random noise before mining starts (the model
        must be non-trivial for "hard" to mean anything).
    mine_iters : int
        CMA-MAE ask/tell generations per epoch.
    hard_dataset_dir : str or None
        Where to persist the accumulated dataset each epoch (optional).
    max_total_samples : int
        Cap on the accumulated dataset (keeps the highest-scoring windows).
    descriptor_dim, n_cells, learning_rate, n_emitters, emitter_batch_size,
    n_warmup, mine_seed :
        Forwarded to :class:`CMAMAEMiner`.
    """

    def __init__(
        self,
        *vanilla_args,
        hard_bias_prob=0.5,
        keep_threshold=5.0,
        warmup_epochs=1,
        mine_iters=200,
        hard_dataset_dir=None,
        max_total_samples=30_000_000,
        descriptor_dim=8,
        n_cells=1024,
        learning_rate=0.1,
        n_emitters=1,
        emitter_batch_size=36,
        n_warmup=2048,
        mine_seed=None,
        **vanilla_kwargs,
    ):
        super().__init__(*vanilla_args, **vanilla_kwargs)

        # Local imports: only the hard-mining path needs pyribs / the miner.
        from sage.data.noise.cma_mae_mining import (
            CMAMAEMiner, make_miner_preprocessor,
        )
        from sage.data.noise.lowfar_noise import _MiningReader

        ns = self.noise_sampler
        if not hasattr(ns, "set_hard_dataset"):
            raise TypeError(
                "SageHardMiningTraining needs a noise sampler with "
                "set_hard_dataset (e.g. MemmapNoiseSampler)."
            )
        dcfg = get_data_cfg()
        self._reader = _MiningReader(ns, seed=mine_seed)
        self._preprocess = make_miner_preprocessor(self.processor, self.signal_sampler)
        self._miner = CMAMAEMiner(
            detectors=list(self.cfg.detectors),
            seg_index=ns.seg_index,
            seq_len=ns.seq_len,
            bin_files=[str(f) for f in ns.bin_files],
            sample_rate=float(dcfg.sample_rate),
            keep_threshold=keep_threshold,
            descriptor_dim=descriptor_dim,
            n_cells=n_cells,
            learning_rate=learning_rate,
            n_emitters=n_emitters,
            emitter_batch_size=emitter_batch_size,
            n_warmup=n_warmup,
            seed=mine_seed,
        )

        self.hard_bias_prob = float(hard_bias_prob)
        self.warmup_epochs = int(warmup_epochs)
        self.mine_iters = int(mine_iters)
        self.hard_dataset_dir = hard_dataset_dir
        self.max_total_samples = int(max_total_samples)
        self._accumulated = None

    # ------------------------------------------------------------------
    def forward(self, nepoch):
        """One standard training epoch, then (after warmup) a mining pass."""
        super().forward(nepoch)
        if nepoch >= self.warmup_epochs:
            self._mine(nepoch)

    # ------------------------------------------------------------------
    def _build_evaluate_fn(self):
        """``(starts, segs) -> (scores, embeddings)`` via read -> model -> embed.

        The QD diversity descriptor is the model's own learned embedding:
          * **consistency model** -> ``model(x, return_embedding=True)`` returns
            the per-detector attention-pooled frontend feature ``(B, D, C)``; we
            L2-norm per detector and flatten.  This is the explicit, opt-in path
            (no forward hook), so it is guaranteed and never silently falls back.
          * **any other model** -> the feature feeding the ranking head, via a
            pre-hook on ``get_ranking_statistic`` (best available fallback).

        Runs the eager (compile-free, no-grad) model in chunks.  Returns
        ``(run_fn, hook_handle_or_None)``.
        """
        eager = getattr(self.model, "_orig_mod", self.model)
        chunk = self.cfg.batch_size

        if hasattr(eager, "per_det_head"):
            handle = None

            def run(net_input):                              # explicit opt-in
                out, emb = eager(net_input, return_embedding=True)
                score = out[0].reshape(-1).float()
                emb = F.normalize(emb.float(), dim=-1).flatten(1)   # (B, D*C)
                return score, emb
        elif hasattr(eager, "get_ranking_statistic"):
            captured = []
            handle = eager.get_ranking_statistic.register_forward_pre_hook(
                lambda m, inp: captured.append(inp[0].detach())
            )

            def run(net_input):                              # fallback hook
                captured.clear()
                out = eager(net_input)
                score = (out[0] if isinstance(out, tuple) else out).reshape(-1).float()
                emb = F.normalize(captured[-1].float().flatten(1), dim=1)
                return score, emb
        else:
            raise TypeError(
                "Hard mining needs a model with return_embedding support "
                "(consistency model) or get_ranking_statistic to embed from."
            )

        def evaluate_fn(starts, segs):
            scores, embs = [], []
            for i in range(0, len(starts), chunk):
                net_input = self._preprocess(
                    self._reader.read_batch(starts[i:i + chunk], segs[i:i + chunk])
                )
                score, emb = run(net_input)
                scores.append(score.cpu().numpy())
                embs.append(emb.cpu().numpy())
            return np.concatenate(scores), np.concatenate(embs, axis=0)

        return evaluate_fn, handle

    @torch.inference_mode()
    def _mine(self, nepoch):
        self.model.eval()
        evaluate_fn, handle = self._build_evaluate_fn()
        try:
            fresh = self._miner.mine(
                evaluate_fn, self.mine_iters, seed_dataset=self._accumulated
            )
        finally:
            if handle is not None:                    # consistency model uses no hook
                handle.remove()

        self._accumulated = (
            fresh if self._accumulated is None else self._accumulated.merge(fresh)
        )
        if len(self._accumulated) > self.max_total_samples:
            keep = np.argsort(-self._accumulated.scores)[: self.max_total_samples]
            self._accumulated = self._accumulated.filter(
                float(self._accumulated.scores[keep[-1]])
            )
        self.noise_sampler.set_hard_dataset(
            self._accumulated,
            hard_bias_prob=self.hard_bias_prob,
            epoch=nepoch,
            save_dir=self.hard_dataset_dir,
        )
        print(
            f"[HardMining] epoch {nepoch}: +{len(fresh):,} mined "
            f"-> {len(self._accumulated):,} accumulated hard windows",
            flush=True,
        )

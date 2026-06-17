#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training callbacks — objects that hook into :class:`SageVanillaTraining`'s loop.

A callback can transform the per-batch context (e.g. inject non-astrophysical
samples) and/or run work at epoch boundaries (e.g. hard-noise mining). Every
hook is a no-op by default, so a trainer constructed with no callbacks behaves
exactly like plain vanilla training.
"""

import numpy as np
import torch
import torch.nn.functional as F

from sage.core.config import get_data_cfg


class Callback:
    """Base training callback. Subclass and override only the hooks you need.

    Hooks
    -----
    on_sample(ctx, trainer)
        Called once per batch *after* the batch is assembled (post signal
        injection) and *before* preprocessing. ``ctx`` is a plain dict the
        trainer threads through the iteration — read/write its tensors
        (``ctx['x']``, ``ctx['targets']`` and any extras a callback adds, e.g.
        ``ctx['per_det_mask']``). Mutate ``ctx`` in place.
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


class MaskingCallback(Callback):
    """Assemble the 4-class consistency batch with a ``NonAstrophysicalMasker``.

    Replaces the trainer's default injection (sets ``ctx['x']``): the extra
    signal-pool injections are decohered into non-astrophysical (class-0) pairs
    dropped into noise slots, producing coherent (class 1) + non-astro (class 0)
    + pure-noise slots, plus a per-detector supervision mask. Sets:
      * ``ctx['x']``           — assembled strain (noise + injections)
      * ``ctx['targets']``     — full width ``[pe..., class, tc_det..., mc_det...]``
      * ``ctx['per_det_mask']``— ``(B, D)`` per-detector supervision mask

    Requires a signal sampler built with ``append_per_det_targets=True`` and an
    ``extra_batch`` pool (so ``signal_targets`` carries the per-detector columns
    and there are ``> S`` signals to decohere). With no masker / no extra pool it
    reduces to the 2-class coherent-signal vs noise regime.
    """

    def __init__(self, masker):
        self.masker = masker

    def on_sample(self, ctx, trainer):
        cfg = trainer.cfg
        device = cfg.device
        D = len(cfg.detectors)
        num_pe = len(cfg.do_point_estimate)
        mw = num_pe + 1
        fw = mw + 2 * D
        tc0, mc0 = mw, mw + D
        B, S = trainer.B, trainer.S

        signal_data = ctx["signal_data"]
        signal_targets = ctx["signal_targets"]
        noise_data = ctx["noise_data"]
        noise_class = ctx["noise_targets"]                   # (B, 1) class label

        coh_data = signal_data[:S]
        coh_tgt = signal_targets[:S]                          # class 1, full width
        coh_mask = torch.ones(S, D, device=device, dtype=signal_targets.dtype)

        # Non-astrophysical pool -> class-0 injections (eat the noise budget,
        # never the class-1 signal budget).
        extra = 0
        if self.masker is not None and signal_data.shape[0] > S:
            pool_data = signal_data[S:]
            pool_tc = signal_targets[S:, tc0:mc0]            # (extra, D)
            pool_mc = signal_targets[S:, mc0 : mc0 + D]      # (extra, D)
            na_data, na_tc, na_mc, na_mask = self.masker(pool_data, pool_tc, pool_mc)
            extra = na_data.shape[0]
            na_tgt = torch.zeros(extra, fw, device=device, dtype=signal_targets.dtype)
            na_tgt[:, tc0:mc0] = na_tc
            na_tgt[:, mc0 : mc0 + D] = na_mc

        # Assemble B slots: S coherent (cls 1), `extra` non-astro (cls 0), rest
        # pure noise.
        perm = torch.randperm(B, device=device)
        coh_slots = perm[:S]
        na_slots = perm[S : S + extra]

        inj = torch.zeros_like(noise_data)
        targets = torch.zeros(B, fw, device=device, dtype=signal_targets.dtype)
        per_det_mask = torch.zeros(B, D, device=device, dtype=coh_mask.dtype)

        # pure-noise class label for every slot first, then overwrite the
        # injected slots with their own full-width targets.
        targets[:, num_pe : num_pe + 1] = noise_class

        inj[coh_slots] = coh_data
        targets[coh_slots] = coh_tgt
        per_det_mask[coh_slots] = coh_mask

        if extra > 0:
            inj[na_slots] = na_data
            targets[na_slots] = na_tgt
            per_det_mask[na_slots] = na_mask

        ctx["x"] = noise_data + inj
        ctx["targets"] = targets
        ctx["per_det_mask"] = per_det_mask


class HardMiningCallback(Callback):
    """Per-epoch CMA-MAE hard-noise mining, as a training callback.

    After each epoch (past ``warmup_epochs``) it mines hard noise with
    :class:`~sage.data.noise.cma_mae_mining.CMAMAEMiner`, accumulates the windows
    across epochs, and pushes them to the noise sampler via ``set_hard_dataset``
    (so subsequent batches draw hard windows with probability ``hard_bias_prob``).
    Requires a noise sampler with ``set_hard_dataset`` and a model exposing the
    opt-in embedding (consistency model) or ``get_ranking_statistic``.

    pyribs / the miner are imported lazily on first use, so merely importing this
    module — and pure vanilla training — never requires pyribs. The miner is
    built lazily from the trainer's graph on the first mining pass.
    """

    def __init__(
        self,
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
    ):
        self.hard_bias_prob = float(hard_bias_prob)
        self.keep_threshold = float(keep_threshold)
        self.warmup_epochs = int(warmup_epochs)
        self.mine_iters = int(mine_iters)
        self.hard_dataset_dir = hard_dataset_dir
        self.max_total_samples = int(max_total_samples)
        self.descriptor_dim = int(descriptor_dim)
        self.n_cells = int(n_cells)
        self.learning_rate = float(learning_rate)
        self.n_emitters = int(n_emitters)
        self.emitter_batch_size = int(emitter_batch_size)
        self.n_warmup = int(n_warmup)
        self.mine_seed = mine_seed
        self._accumulated = None
        self._miner = None              # lazily built from the trainer's graph

    def _lazy_init(self, trainer):
        # Local imports: only the hard-mining path needs pyribs / the miner.
        from sage.data.noise.cma_mae_mining import (
            CMAMAEMiner, make_miner_preprocessor,
        )
        from sage.data.noise.lowfar_noise import _MiningReader

        ns = trainer.noise_sampler
        if not hasattr(ns, "set_hard_dataset"):
            raise TypeError(
                "HardMiningCallback needs a noise sampler with set_hard_dataset "
                "(e.g. MemmapNoiseSampler)."
            )
        dcfg = get_data_cfg()
        self._reader = _MiningReader(ns, seed=self.mine_seed)
        self._preprocess = make_miner_preprocessor(
            trainer.processor, trainer.signal_sampler
        )
        self._miner = CMAMAEMiner(
            detectors=list(trainer.cfg.detectors),
            seg_index=ns.seg_index,
            seq_len=ns.seq_len,
            bin_files=[str(f) for f in ns.bin_files],
            sample_rate=float(dcfg.sample_rate),
            keep_threshold=self.keep_threshold,
            descriptor_dim=self.descriptor_dim,
            n_cells=self.n_cells,
            learning_rate=self.learning_rate,
            n_emitters=self.n_emitters,
            emitter_batch_size=self.emitter_batch_size,
            n_warmup=self.n_warmup,
            seed=self.mine_seed,
        )

    # ------------------------------------------------------------------
    def on_epoch_end(self, nepoch, trainer):
        if nepoch < self.warmup_epochs:
            return
        if self._miner is None:
            self._lazy_init(trainer)
        self._mine(nepoch, trainer)

    # ------------------------------------------------------------------
    def _build_evaluate_fn(self, trainer):
        """``(starts, segs) -> (scores, embeddings)`` via read -> model -> embed.

        The QD diversity descriptor is the model's own learned embedding:
          * **consistency model** -> ``model(x, return_embedding=True)`` returns
            the per-detector attention-pooled frontend feature ``(B, D, C)``; we
            L2-norm per detector and flatten.  This is the explicit, opt-in path
            (no forward hook), so it is guaranteed and never silently falls back.
          * **any other model** -> the feature feeding the ranking head, via a
            pre-hook on ``get_ranking_statistic`` (best available fallback).

        Runs the eager (compile-free, no-grad) model in chunks.  Returns
        ``(evaluate_fn, hook_handle_or_None)``.
        """
        eager = getattr(trainer.model, "_orig_mod", trainer.model)
        chunk = trainer.cfg.batch_size

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
    def _mine(self, nepoch, trainer):
        trainer.model.eval()
        evaluate_fn, handle = self._build_evaluate_fn(trainer)
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
        trainer.noise_sampler.set_hard_dataset(
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

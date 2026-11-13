#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training callbacks: hooks into the :class:`SageVanillaTraining` loop.

Hard-noise mining and the per-step weight EMA. Every hook does nothing
by default, so a trainer without callbacks trains plainly.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class Callback:
    """
    Base training callback; subclasses override the hooks they need.

    Each hook receives the :class:`SageVanillaTraining` instance, which
    exposes ``model``, ``noise_sampler``, ``signal_sampler``,
    ``processor`` and the other loop state.
    """

    def on_sample(self, ctx, trainer):
        """
        Hook called once per batch, after sampling and before assembly.

        ``ctx`` holds the sampled ``signal_data``, ``signal_targets``,
        ``noise_data`` and ``noise_targets``, with ``x`` and ``targets``
        None; setting both replaces the default batch assembly.

        Parameters
        ----------
        ctx : dict
            Per-batch context, changed in place.
        trainer : SageVanillaTraining
            The trainer.
        """
        pass

    def on_batch_end(self, nbatch, nepoch, trainer):
        """
        Hook called after each batch's optimiser and scheduler step.

        Parameters
        ----------
        nbatch : int
            Batch index within the epoch.
        nepoch : int
            Epoch.
        trainer : SageVanillaTraining
            The trainer.
        """
        pass

    def on_epoch_end(self, nepoch, trainer):
        """
        Hook called once after an epoch's training batches.

        Parameters
        ----------
        nepoch : int
            Epoch just trained.
        trainer : SageVanillaTraining
            The trainer.
        """
        pass


class HardMiningCallback(Callback):
    """
    Continuous CMA-MAE hard-noise mining, as a training callback.

    After each epoch in ``mine_epochs`` it mines hard noise into the
    on-disk bank, re-scores the whole bank with the current model and
    gives the noise sampler the starts scoring >= ``keep_threshold``.

    Parameters
    ----------
    mining : dict
        Checked ``mining`` section (keep threshold, iterations, replay,
        archive and emitter settings).
    bank_path : Path
        The run's bank file (``mining/bank.h5``).
    mine_epochs : list of int
        Epochs after which a mining round runs.
    seed : int
        Seed of the miner and its reader.
    detectors : list of str
        Detectors, in channel order.
    sample_rate : float
        Sample rate of the noise windows.
    """

    def __init__(self, mining, bank_path, mine_epochs, seed, detectors, sample_rate):
        replay, archive, emitter = mining["replay"], mining["archive"], mining["emitter"]
        self.bank_path = Path(bank_path)
        self.mine_epochs = set(mine_epochs)
        # Raw logit at or above which a window counts as hard
        self.keep_threshold = mining["keep_threshold"]
        self.mine_iters = mining["iterations"]
        self.hard_bias_prob = replay["probability"]      # target/max bias
        self.bias_replays = replay["replays_per_epoch"]
        self.descriptor_dim = archive["descriptor_dim"]
        self.n_cells = archive["n_cells"]
        self.novelty_dist = archive["novelty_dist"]
        self.novelty_weight = archive["novelty_weight"]
        self.max_embeddings = archive["max_embeddings"]
        self.n_emitters = emitter["n_emitters"]
        self.emitter_batch_size = emitter["batch_size"]
        self.learning_rate = emitter["learning_rate"]
        self.n_warmup = emitter["warmup"]
        self.mine_seed = seed
        self.detectors = list(detectors)
        self.sample_rate = float(sample_rate)
        self._bank = None
        self._miner = None              # lazily built from the trainer's graph

    def _should_mine(self, nepoch):
        """
        Whether a mining round runs after epoch ``nepoch``.

        Parameters
        ----------
        nepoch : int
            Epoch just trained.

        Returns
        -------
        bool
            True for an epoch in ``mine_epochs``.
        """
        return int(nepoch) in self.mine_epochs

    def _bias_for(self, active, trainer):
        """
        Replay probability of hard windows, annealed with their count.

        Each active window is replayed at most about ``bias_replays``
        times per epoch: ``p = min(hard_bias_prob, bias_replays *
        n_active / (iterations * batch_size))``.

        Parameters
        ----------
        active : array-like
            Indices of the active (currently hard) starts.
        trainer : SageVanillaTraining
            The trainer, for its batches per epoch and batch size.

        Returns
        -------
        float
            Replay probability; 0 without active starts.
        """
        n = int(len(active))
        draws = int(trainer.num_iterations) * int(trainer.batch_size)
        if n <= 0 or draws <= 0:
            return 0.0
        return float(min(self.hard_bias_prob, self.bias_replays * n / draws))

    def _lazy_init(self, trainer):
        """
        Build the bank, mining reader, miner preprocessing and miner.

        Parameters
        ----------
        trainer : SageVanillaTraining
            The trainer, whose noise sampler and processor are used.

        Raises
        ------
        TypeError
            If the noise sampler cannot take a hard bank.
        """
        # Local imports: only hard mining needs pyribs and the miner
        from sage.data.noise.cma_mae_mining import (
            CMAMAEMiner, make_miner_preprocessor,
        )
        from sage.data.noise.lowfar_noise import _MiningReader
        from sage.data.noise.hard_bank import HardMiningBank

        ns = trainer.noise_sampler
        if not hasattr(ns, "set_hard_bank"):
            raise TypeError(
                "HardMiningCallback needs a noise sampler with set_hard_bank "
                "(e.g. MemmapNoiseSampler)."
            )
        # Runs in ns.run_names order: it set the run ids of the segment
        # table and of the bank's start_runs, so replay reads the right
        # run's memmap.
        runs = list(ns.run_names)
        detectors = self.detectors
        self._bank = HardMiningBank(
            self.bank_path, detectors=detectors, runs=runs, seq_len=ns.seq_len,
            sample_rate=self.sample_rate,
            bin_files=[str(f) for f in ns.bin_files],
            descriptor_dim=self.descriptor_dim, novelty_dist=self.novelty_dist,
            max_embeddings=self.max_embeddings,
        )
        # postprocess=False: mining and re-evaluation read unrecoloured
        # noise, so hardness is a property of the stored window; replay
        # through the training sampler is still recoloured.
        self._reader = _MiningReader(ns, seed=self.mine_seed, postprocess=False)
        self._preprocess = make_miner_preprocessor(trainer.processor)
        self._miner = CMAMAEMiner(
            detectors=detectors, seg_index=ns.seg_index,
            segment_probs=ns.segment_probs, seq_len=ns.seq_len,
            bank=self._bank, keep_threshold=self.keep_threshold,
            descriptor_dim=self.descriptor_dim, n_cells=self.n_cells,
            learning_rate=self.learning_rate, n_emitters=self.n_emitters,
            emitter_batch_size=self.emitter_batch_size, n_warmup=self.n_warmup,
            novelty_weight=self.novelty_weight, seed=self.mine_seed,
        )

    def on_epoch_end(self, nepoch, trainer):
        """
        Run a mining round if ``nepoch`` is a mining epoch.

        Parameters
        ----------
        nepoch : int
            Epoch just trained.
        trainer : SageVanillaTraining
            The trainer.
        """
        if not self._should_mine(nepoch):
            return
        if self._miner is None:
            self._lazy_init(trainer)
        self._mine(nepoch, trainer)

    def attach_for_resume(self, trainer):
        """
        Give the noise sampler the hard starts of the stored bank.

        Called once after a checkpoint is restored, since a rebuilt
        sampler starts without them. Does nothing while the bank holds
        no start.

        Parameters
        ----------
        trainer : SageVanillaTraining
            The trainer.
        """
        if self._miner is None:
            self._lazy_init(trainer)
        if self._bank.is_cold:
            return
        active = self._bank.active_start_indices(self.keep_threshold)
        trainer.noise_sampler.set_hard_bank(
            self._bank, active, hard_bias_prob=self._bias_for(active, trainer)
        )
        print(
            f"[HardMining] resume: re-biased sampler from bank — "
            f"{len(active):,} active / {self._bank.n_starts:,} starts",
            flush=True,
        )

    def _build_evaluate_fn(self, trainer):
        """
        The miner's window evaluation function, and its cleanup.

        Windows are scored by the uncompiled model in chunks of a batch;
        the diversity descriptor is the model's frontend embedding
        (``FrontendTap``), not the backend feature.

        Parameters
        ----------
        trainer : SageVanillaTraining
            The trainer, for its model and batch size.

        Returns
        -------
        tuple
            ``(evaluate_fn, cleanup)``: ``evaluate_fn(starts, segs,
            runs) -> (scores, embeddings)``; ``cleanup()`` restores
            the model's ``return_frontend_embedding`` flag.

        Raises
        ------
        TypeError
            If the model has no frontend tap.
        """
        eager = getattr(trainer.model, "_orig_mod", trainer.model)
        chunk = trainer.batch_size
        cleanup = lambda: None

        if hasattr(eager, "return_frontend_embedding"):
            # Through the model's flag, not a forward hook
            _prev = eager.return_frontend_embedding
            eager.return_frontend_embedding = True

            def cleanup(eager=eager, prev=_prev):
                """Restore the model's frontend-embedding flag."""
                eager.return_frontend_embedding = prev

            def run(net_input):
                """Ranking statistic and frontend embedding."""
                out = eager(net_input)
                score = (out[0] if isinstance(out, tuple) else out).reshape(-1).float()
                return score, eager.frontend_embedding.float()   # (B, D*C), unit/det
        else:
            raise TypeError(
                "Hard mining needs a model with the frontend tap "
                "(sage.architecture.network.common.FrontendTap)."
            )

        def evaluate_fn(starts, segs, runs):
            """Scores and frontend embeddings of the given windows."""
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
        """
        One round: mine, re-score the bank and update the noise sampler.

        A round already in the bank (a resume after a crash) is not
        mined again; the sampler's hard starts are refreshed from it.

        Parameters
        ----------
        nepoch : int
            Epoch just trained.
        trainer : SageVanillaTraining
            The trainer.
        """
        if self._bank.has_epoch(nepoch):
            active = self._bank.active_start_indices(self.keep_threshold)
            trainer.noise_sampler.set_hard_bank(
                self._bank, active, hard_bias_prob=self._bias_for(active, trainer)
            )
            print(
                f"[HardMining] epoch {nepoch}: already in bank (resume) — "
                f"skipped re-mine; re-biased {len(active):,} active",
                flush=True,
            )
            return

        was_training = trainer.model.training
        trainer.model.eval()
        evaluate_fn, cleanup = self._build_evaluate_fn(trainer)
        try:
            # One atomic round: a job stopped mid-round leaves the bank
            # as it was
            with self._bank.atomic_round():
                # Mine new hard starts and a diverse embedding subset
                _t0 = time.perf_counter()
                mstats = self._miner.mine(evaluate_fn, self.mine_iters, epoch=nepoch)
                _t_mine = time.perf_counter() - _t0
                # Re-score the whole bank with the current model, timed
                # apart since its cost grows with the bank
                _t1 = time.perf_counter()
                rstats = self._miner.reevaluate(evaluate_fn, model_epoch=nepoch)
                _t_reeval = time.perf_counter() - _t1
        finally:
            cleanup()                                 # reset the frontend-embed flag
            if was_training:
                trainer.model.train()

        # The sampler gets the currently hard starts (latest score >=
        # keep threshold); the bank keeps every start
        active = self._bank.active_start_indices(self.keep_threshold)
        trainer.noise_sampler.set_hard_bank(
            self._bank, active, hard_bias_prob=self._bias_for(active, trainer)
        )
        print(
            f"[HardMining] epoch {nepoch}: mined +{mstats['kept_starts']:,} starts "
            f"(+{mstats['kept_embeddings']} emb) | bank {mstats['bank_starts']:,} "
            f"starts, {mstats['bank_embeddings']:,} emb | active "
            f"{len(active):,}/{rstats.get('reeval_n', 0):,} | "
            f"bias {self._bias_for(active, trainer):.3f} | "
            f"mine {_t_mine:.1f}s reeval {_t_reeval:.1f}s",
            flush=True,
        )
        # How long the active starts have stayed hard
        try:
            ages = self._bank.hard_sample_ages(self.keep_threshold)
            act = ages[ages["currently_active"]] if len(ages) else ages
            if len(act):
                print(
                    f"[HardMining] age: active median {int(np.median(act['age_epochs']))} ep, "
                    f"oldest {int(act['age_epochs'].max())} ep, "
                    f"{int((act['streak_rounds'] >= 5).sum()):,} hard >=5 rounds straight",
                    flush=True,
                )
        except Exception as e:                                   # never break training
            print(f"[HardMining] age summary skipped: {e}", flush=True)


class EMACallback(Callback):
    """
    Per-step exponential moving average (EMA) of the model weights.

    Uses :class:`torch.optim.swa_utils.AveragedModel` with
    ``get_ema_multi_avg_fn``, updated every batch, buffers included.

    Parameters
    ----------
    model : nn.Module
        The uncompiled network: it shares parameter tensors with the
        compiled model, so each step reads the current weights.
    decay : float
        Weight of the running average at each step.
    """

    def __init__(self, model, decay):
        self.model = model
        # AveragedModel deep-copies the model; n_averaged starts at 0,
        # so the first update seeds the average with the live weights.
        self.ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True,
        )

    def on_batch_end(self, nbatch, nepoch, trainer):
        """
        Update the average with the current weights.

        Parameters
        ----------
        nbatch : int
            Batch index within the epoch.
        nepoch : int
            Epoch.
        trainer : SageVanillaTraining
            The trainer.
        """
        self.ema_model.update_parameters(self.model)

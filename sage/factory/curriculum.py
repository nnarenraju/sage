#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Curriculum training with hard-noise mining.

SageCurriculumTraining
    Wraps SageVanillaTraining and adds periodic mining passes using
    CMAMEMiner (GPS explorer) and CMAMEGAMiner (pattern refiner).
    Both miners write to a SharedHardNoiseBank that grows across all runs.
    The noise sampler is a MemmapNoiseSampler with hard_dataset_dir and
    hard_bias_prob — it handles hard/random mixing internally, so the
    training loop itself is unchanged.

    Schedule (all cadences are in epochs, 0-indexed):
        warmup_epochs      : train with random noise only (no mining)
        mine_explore_every : run CMAMEMiner every N epochs
        mine_refine_every  : run CMAMEGAMiner every M epochs (M < N)
        validate_every     : run validation every K epochs

    Example
    -------
        from sage.data.noise import (
            MemmapNoiseSampler, SharedHardNoiseBank,
            CMAMEMiner, CMAMEGAMiner,
        )
        from sage.factory import SageCurriculumTraining
        from sage.factory.training import SageVanillaTraining
        from sage.factory.validation import SageVanillaValidation

        bank = SharedHardNoiseBank.load("datasets/bank.npz")  # or create new

        noise_sampler = MemmapNoiseSampler(
            hard_dataset_dir = "datasets/",
            hard_bias_prob   = 0.6,
        )
        explorer = CMAMEMiner(explore_fraction=0.7, threshold=3.0)
        refiner  = CMAMEGAMiner(threshold=5.0, target_samples=5_000_000)

        vanilla_train = SageVanillaTraining(signal, noise_sampler, proc, model, ...)
        vanilla_val   = SageVanillaValidation(signal_val, noise_val, proc, model, ...)

        curriculum = SageCurriculumTraining(
            vanilla_training   = vanilla_train,
            vanilla_validation = vanilla_val,
            noise_sampler      = noise_sampler,
            signal_sampler     = signal_sampler,
            processor          = processor,
            model              = model,
            explorer           = explorer,
            refiner            = refiner,
            shared_bank        = bank,
            dataset_dir        = "datasets/",
            bank_path          = "datasets/bank.npz",
            total_epochs       = 80,
            warmup_epochs      = 10,
            mine_explore_every = 5,
            mine_refine_every  = 1,
            validate_every     = 5,
        )
        curriculum.run()
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sage.core.config import get_cfg
from sage.data.noise.qd_mining import SharedHardNoiseBank
from sage.data.noise.real_noise import _find_latest_hard_dataset
from sage.data.noise.lowfar_noise import StartTimeDataset


class SageCurriculumTraining:
    """
    Curriculum training with hard-noise mining.

    This class does not re-implement the training or validation loops —
    it calls ``vanilla_training(nepoch)`` and ``vanilla_validation(nepoch)``
    from SageVanillaTraining / SageVanillaValidation.  Mining is injected
    between epochs according to the schedule below.

    The ``noise_sampler`` must be a MemmapNoiseSampler initialised with
    ``hard_dataset_dir`` and ``hard_bias_prob``.  After each mining pass,
    ``noise_sampler.set_hard_dataset`` is called to swap in the new dataset;
    the sampler's prefetch thread picks it up on the very next batch without
    a restart.

    Mining schedule
    ---------------
    Epochs 0 … warmup_epochs-1 : random noise only (no mining).

    From epoch warmup_epochs onward:

        Every mine_explore_every epochs → run CMAMEMiner (GPS explorer).
            Finds new GPS regions + noise types.  ~70% of its budget is
            pure random exploration; ~30% re-validates GPS positions already
            in the bank with the current model.

        Every mine_refine_every epochs → run CMAMEGAMiner (pattern refiner).
            Re-evaluates stale bank templates; gradient-refines them to be
            harder; then mines more of the same patterns efficiently.

    When both mining passes are triggered on the same epoch, the explorer
    runs first (its findings enrich the bank for the refiner).

    Parameters
    ----------
    vanilla_training : SageVanillaTraining
        Pre-built training loop (model, loss, optimiser, scheduler already
        configured inside it).
    vanilla_validation : SageVanillaValidation or None
        Pre-built validation loop.  Pass ``None`` to skip validation.
    noise_sampler : MemmapNoiseSampler
        Must have ``hard_dataset_dir`` and ``hard_bias_prob`` set.
    signal_sampler : nn.Module
        Signal sampler from training — passed to miners so they see the
        same preprocessing pipeline (GWBatch state, coarse selector).
    processor : nn.Module
        Same preprocessor used in training.
    model : nn.Module
        The network.  Miners freeze it (eval mode) during mining passes.
    explorer : CMAMEMiner or None
        GPS-space quality-diversity miner.
    refiner : CMAMEGAMiner or None
        Template-bank pattern refiner.
    shared_bank : SharedHardNoiseBank
        Mutable shared state between miners.  Both miners read from and
        write to this object in-place.
    dataset_dir : str or Path
        Directory where versioned StartTimeDataset files are saved.
        Files are named ``hard_noise_epoch_NNNN.npz``.
    bank_path : str or Path
        Path where the SharedHardNoiseBank is saved after each mining pass.
    total_epochs : int
    warmup_epochs : int
        Epochs before any mining starts.  Model trains on random noise.
    mine_explore_every : int
        Run the GPS explorer every this many epochs.
    mine_refine_every : int
        Run the pattern refiner every this many epochs.
    validate_every : int
        Run validation every this many epochs.
    logger : HDF5LossLogger or None
        If provided, training and validation losses are logged each epoch.
    ckpt_mgr : CheckpointManager or None
        If provided, checkpoints are saved after each validation.
    """

    def __init__(
        self,
        vanilla_training,
        vanilla_validation,
        noise_sampler,
        signal_sampler,
        processor,
        model,
        explorer=None,
        refiner=None,
        shared_bank: SharedHardNoiseBank | None = None,
        dataset_dir: str | Path = "datasets/",
        bank_path:   str | Path = "datasets/bank.npz",
        total_epochs:        int = 80,
        warmup_epochs:       int = 10,
        mine_explore_every:  int = 5,
        mine_refine_every:   int = 1,
        validate_every:      int = 5,
        logger               = None,
        ckpt_mgr             = None,
        # Progressive hard-noise accumulation
        threshold_schedule         = None,
        accumulate:         bool   = True,
        max_total_samples:  int    = 20_000_000,
    ):
        self.vanilla_training   = vanilla_training
        self.vanilla_validation = vanilla_validation
        self.noise_sampler      = noise_sampler
        self.signal_sampler     = signal_sampler
        self.processor          = processor
        self.model              = model
        self.explorer           = explorer
        self.refiner            = refiner
        self.shared_bank        = shared_bank if shared_bank is not None \
                                  else SharedHardNoiseBank(K=32)
        self.dataset_dir        = Path(dataset_dir)
        self.bank_path          = Path(bank_path)
        self.total_epochs       = total_epochs
        self.warmup_epochs      = warmup_epochs
        self.mine_explore_every = mine_explore_every
        self.mine_refine_every  = mine_refine_every
        self.validate_every     = validate_every
        self.logger             = logger
        self.ckpt_mgr           = ckpt_mgr

        self.threshold_schedule = threshold_schedule
        self.accumulate         = accumulate
        self.max_total_samples  = max_total_samples

        self.cfg = get_cfg()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)

        # Load accumulated dataset on startup (resume support).
        # MemmapNoiseSampler already hot-swaps it in; here we keep a reference
        # so _update_accumulated can merge/filter incrementally.
        _latest = _find_latest_hard_dataset(self.dataset_dir)
        if _latest is not None:
            self._accumulated: StartTimeDataset | None = StartTimeDataset.load(_latest)
            print(
                f"[Curriculum] Loaded accumulated dataset: "
                f"{len(self._accumulated):,} windows from {_latest.name}"
            )
        else:
            self._accumulated = None

    # ------------------------------------------------------------------
    def _get_thresholds(self, epoch: int):
        """Return ``(explore_threshold, refine_threshold)`` for this epoch."""
        if self.threshold_schedule is not None:
            return self.threshold_schedule(epoch)
        return (
            self.explorer.threshold if self.explorer else None,
            self.refiner.threshold  if self.refiner  else None,
        )

    # ------------------------------------------------------------------
    def _update_accumulated(self, new_dataset: StartTimeDataset, threshold: float, epoch: int):
        """
        Merge *new_dataset* into the running accumulated hard dataset.

        Steps
        -----
        1. Filter the existing accumulated dataset to ``score >= threshold``
           (evicts windows that are no longer hard enough at the current bar).
        2. Merge the filtered base with *new_dataset* (which already contains
           only windows above the same threshold).
        3. Cap to ``max_total_samples`` keeping the highest-scored windows.
        4. Hot-swap the noise sampler and persist the versioned file to disk.

        When ``accumulate=False`` the call reduces to a plain
        ``set_hard_dataset`` — the same behaviour as before.
        """
        if not self.accumulate:
            self.noise_sampler.set_hard_dataset(
                new_dataset, epoch=epoch, save_dir=self.dataset_dir,
            )
            return

        # Filter existing to current threshold.
        if self._accumulated is not None and len(self._accumulated) > 0:
            base = self._accumulated.filter(threshold)
        else:
            base = None

        # Merge.
        if base is not None and len(base) > 0:
            merged = base.merge(new_dataset)
        else:
            merged = new_dataset

        # Cap: keep highest-scoring windows if over the limit.
        if len(merged) > self.max_total_samples:
            idx = np.argsort(merged.scores)[::-1][: self.max_total_samples]
            merged = StartTimeDataset(
                detectors       = merged.detectors,
                start_indices   = merged.start_indices[idx],
                segment_indices = merged.segment_indices[idx],
                gps_times       = merged.gps_times[idx],
                scores          = merged.scores[idx],
                bin_files       = merged.bin_files,
                sample_rate     = merged.sample_rate,
                seq_len         = merged.seq_len,
            )

        self._accumulated = merged

        pcts = np.percentile(self._accumulated.scores, [50, 75, 90, 99])
        print(
            f"[Curriculum] Accumulated: {len(self._accumulated):,} hard windows "
            f"(threshold≥{threshold:.2f}) | "
            f"score p50/p75/p90/p99 = "
            f"{pcts[0]:.2f}/{pcts[1]:.2f}/{pcts[2]:.2f}/{pcts[3]:.2f}"
        )

        self.noise_sampler.set_hard_dataset(
            self._accumulated, epoch=epoch, save_dir=self.dataset_dir,
        )

    # ------------------------------------------------------------------
    def _should_explore(self, epoch: int) -> bool:
        return (
            self.explorer is not None
            and epoch >= self.warmup_epochs
            and (epoch - self.warmup_epochs) % self.mine_explore_every == 0
        )

    def _should_refine(self, epoch: int) -> bool:
        return (
            self.refiner is not None
            and epoch >= self.warmup_epochs
            and (epoch - self.warmup_epochs) % self.mine_refine_every == 0
        )

    def _should_validate(self, epoch: int) -> bool:
        return (
            self.vanilla_validation is not None
            and ((epoch + 1) % self.validate_every == 0 or epoch == 0)
        )

    # ------------------------------------------------------------------
    def _run_explorer(self, epoch: int):
        """Run CMAMEMiner, accumulate results, update noise sampler + bank."""
        explore_thresh, _ = self._get_thresholds(epoch)
        if explore_thresh is not None:
            self.explorer.threshold = explore_thresh
        print(
            f"\n[Curriculum] Epoch {epoch}: Running GPS explorer "
            f"(CMAMEMiner, threshold={self.explorer.threshold:.2f}) …"
        )
        dataset = self.explorer.mine(
            model          = self.model,
            noise_sampler  = self.noise_sampler,
            processor      = self.processor,
            device         = self.cfg.device,
            signal_sampler = self.signal_sampler,
            shared_bank    = self.shared_bank,
        )
        if len(dataset) > 0:
            self._update_accumulated(dataset, self.explorer.threshold, epoch)
        self.shared_bank.save(self.bank_path)
        acc_n = len(self._accumulated) if self._accumulated is not None else 0
        print(
            f"[Curriculum] Explorer done: {len(dataset):,} new | "
            f"bank: {len(self.shared_bank):,} | "
            f"total accumulated: {acc_n:,}"
        )

    # ------------------------------------------------------------------
    def _run_refiner(self, epoch: int):
        """Run CMAMEGAMiner, accumulate results, update noise sampler + bank."""
        _, refine_thresh = self._get_thresholds(epoch)
        if refine_thresh is not None:
            self.refiner.threshold = refine_thresh
        print(
            f"\n[Curriculum] Epoch {epoch}: Running pattern refiner "
            f"(CMAMEGAMiner, threshold={self.refiner.threshold:.2f}) …"
        )
        dataset = self.refiner.mine(
            model          = self.model,
            noise_sampler  = self.noise_sampler,
            processor      = self.processor,
            device         = self.cfg.device,
            signal_sampler = self.signal_sampler,
            shared_bank    = self.shared_bank,
        )
        if len(dataset) > 0:
            self._update_accumulated(dataset, self.refiner.threshold, epoch)
        self.shared_bank.save(self.bank_path)
        acc_n = len(self._accumulated) if self._accumulated is not None else 0
        print(
            f"[Curriculum] Refiner done: {len(dataset):,} new | "
            f"bank: {len(self.shared_bank):,} | "
            f"total accumulated: {acc_n:,}"
        )

    # ------------------------------------------------------------------
    def run(self, start_epoch: int = 0):
        """
        Run the full training loop.

        Parameters
        ----------
        start_epoch : int
            Resume from this epoch (useful after checkpoint restore).
        """
        for epoch in range(start_epoch, self.total_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.total_epochs - 1}")
            print(f"{'='*60}")

            # Mining (explorer before refiner so refiner sees new patterns)
            if self._should_explore(epoch):
                self._run_explorer(epoch)

            if self._should_refine(epoch):
                self._run_refiner(epoch)

            # Training
            print(f"[Curriculum] Epoch {epoch}: Training …")
            self.vanilla_training(nepoch=epoch)

            if self.logger is not None:
                self.logger.log(
                    self.vanilla_training.loss_components,
                    epoch, split="training"
                )

            # Validation + checkpoint
            if self._should_validate(epoch):
                print(f"[Curriculum] Epoch {epoch}: Validating …")
                self.vanilla_validation(nepoch=epoch)

                if self.logger is not None:
                    self.logger.log(
                        self.vanilla_validation.loss_components,
                        epoch, split="validation"
                    )

                val_loss = self.vanilla_validation.loss_components[epoch][0].item()
                print(f"[Curriculum] Epoch {epoch}: val_loss = {val_loss:.6f}")

                if self.ckpt_mgr is not None:
                    self.ckpt_mgr.save(epoch=epoch, val_loss=val_loss)

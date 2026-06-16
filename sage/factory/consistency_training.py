#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training loop for the multi-detector consistency model.

Mirrors :class:`~sage.factory.training.SageVanillaTraining` for batch
construction and optimisation, but the model returns a
:class:`~sage.architecture.network.mscnn1d_att_resnet2d_cbam.ConsistencyOutput`
and the objective is the sum of two *separate* losses:

  * the existing classification + merged heteroscedastic-PE loss
    (:class:`BCEWithPEsigmaLoss`) on the merged ranking / PE heads, and
  * the per-detector :class:`ConsistencyNLLLoss` on the per-detector
    ``tc`` / ``mchirp`` heads.

The signal sampler MUST be built with ``append_per_det_tc=True`` so the targets
carry the per-detector arrival times after the class column:
``[pe..., class, tc_det0, tc_det1, ...]``.

Supervision masks (per detector) are, for now, derived from the class label —
i.e. the 2-class regime: matched-coherent signals are supervised on both
detectors, pure noise on neither. The 4-class mismatch / time-slide scheme is a
planned extension that will supply explicit per-detector masks here instead.
"""

import torch

from tqdm import tqdm
from contextlib import nullcontext

from sage.core.config import get_cfg
from sage.core.pipeline import GWBatch, Grid, ProcessingState
from .schedulers import ManageScheduler


class SageConsistencyTraining(torch.nn.Module):
    """Consistency-model training loop (BCE + merged-PE + per-detector NLL)."""

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        merged_loss,
        consistency_loss,
        optimiser,
        scheduler,
        scaler,
        num_iterations,
        num_epochs,
        consistency_weight: float = 1.0,
        masker=None,
        scheduler_mode: str = "batch",
    ):
        super().__init__()

        self.cfg = get_cfg()
        # Optional non-astrophysical (decoherent) sample generator (training
        # only). None -> all injections stay coherent (2-class regime).
        self.masker = masker

        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.model = model
        self.merged_loss = merged_loss.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )
        self.consistency_loss = consistency_loss.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )
        self.optimiser = optimiser
        self.scheduler = ManageScheduler(scheduler, scheduler_mode)
        self.scaler = scaler

        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.consistency_weight = float(consistency_weight)

        self.num_point_estimate = len(self.cfg.do_point_estimate)
        self.num_detectors = len(self.cfg.detectors)
        self.B = self.cfg.batch_size
        self.S = int(self.cfg.batch_size * self.cfg.class_balance)
        # target layout: [pe..., class | tc_det0..tc_det{D-1}]
        self.merged_width = self.num_point_estimate + 1
        self.full_width = self.merged_width + self.num_detectors

        # Auto-multiband config (inert unless the signal sampler exposes one).
        self._initial_state = getattr(
            signal_sampler, "output_state", ProcessingState(Grid.FD_UNIFORM)
        )
        self._selector = getattr(signal_sampler, "selector", None)
        self._freqs = self._selector.coarse_freqs if self._selector is not None else None
        self._coarse_indices = (
            self._selector.coarse_indices if self._selector is not None else None
        )

        # Logged per epoch: [total, merged_total, consistency_total].
        self.num_components = 3
        self.loss_components = torch.zeros(
            (num_epochs, self.num_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def forward(self, nepoch):
        self.model.train()
        device = self.cfg.device
        num_pe = self.num_point_estimate
        D = self.num_detectors
        B = self.B

        for nbatch in tqdm(range(self.num_iterations)):

            # ── 1. Sample ──────────────────────────────────────────────────
            signal_data, signal_targets = self.signal_sampler()   # (S, full_width)
            noise_data, noise_targets = self.noise_sampler()      # noise tgt (B, 1)

            if self._selector is not None:
                noise_data = self._selector(noise_data)

            S = signal_data.shape[0]
            per_det_tc = signal_targets[:, self.merged_width :].clone()  # (S, D)

            # ── 1b. Optionally decohere a fraction into non-astrophysical pairs.
            if self.masker is not None:
                signal_data, per_det_tc, signal_mask_S, is_coherent = self.masker(
                    signal_data, per_det_tc
                )
                signal_targets = signal_targets.clone()
                signal_targets[:, num_pe] = is_coherent              # class 0 if decohered
                signal_targets[:, self.merged_width :] = per_det_tc  # updated per-det tc
            else:
                signal_mask_S = torch.ones(
                    S, D, device=device, dtype=signal_targets.dtype
                )

            # ── 2. Pad noise targets to the full width ─────────────────────
            # [0..0 (num_pe) | class | 0..0 (D per-detector tc)]
            noise_full = torch.zeros(
                B, self.full_width, device=device, dtype=noise_targets.dtype
            )
            noise_full[:, num_pe : num_pe + 1] = noise_targets

            # ── 3. Random signal injection ─────────────────────────────────
            idx = torch.randperm(B, device=device)[: self.S]
            signal_pad = torch.zeros_like(noise_data)
            target_pad = torch.zeros(
                B, self.full_width, device=device, dtype=signal_targets.dtype
            )
            signal_pad[idx] = signal_data
            target_pad[idx] = signal_targets

            # Per-detector supervision mask (B, D): signal slots carry their
            # per-detector mask; pure-noise slots stay 0.
            per_det_mask = torch.zeros(
                B, D, device=device, dtype=signal_mask_S.dtype
            )
            per_det_mask[idx] = signal_mask_S

            x = noise_data + signal_pad
            targets = noise_full + target_pad

            # ── 4. Preprocess ──────────────────────────────────────────────
            batch = GWBatch(
                x,
                state=self._initial_state,
                freqs=self._freqs,
                coarse_indices=self._coarse_indices,
            )
            batch = self.processor(batch)
            net_input = batch.to_network_input()

            # ── 5. Forward + combined loss ─────────────────────────────────
            self.optimiser.zero_grad(set_to_none=True)

            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.autocast
                else nullcontext()
            ):
                out = self.model(net_input)

                # Existing classification + merged-PE loss (unchanged contract).
                merged = self.merged_loss(
                    (out.ranking_stat, out.point_estimates),
                    targets[:, : self.merged_width],
                )

                # Per-detector consistency NLL (per-detector supervision mask).
                tc_target = targets[:, self.merged_width :]            # (B, D) seconds
                mc_target = targets[:, 1]                              # (B,) std mchirp
                cons = self.consistency_loss(
                    out.mu_tc, out.log_sigma_tc, out.mu_mc, out.log_sigma_mc,
                    tc_target, mc_target, per_det_mask,
                )

                total = merged[0] + self.consistency_weight * cons[0]

            # ── 6. Backward + step ─────────────────────────────────────────
            if self.scaler is not None:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.clip_norm
            )

            if self.scaler is not None:
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                self.optimiser.step()

            self.scheduler.batch_step(nepoch, nbatch, self.num_iterations)
            self.loss_components[nepoch] += torch.stack(
                [total.detach(), merged[0].detach(), cons[0].detach()]
            )

        self.loss_components[nepoch] /= self.num_iterations

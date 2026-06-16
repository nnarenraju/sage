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

The signal sampler MUST be built with ``append_per_det_targets=True`` so the
targets carry the per-detector arrival times *and* chirp masses after the class
column: ``[pe..., class, tc_det0..tc_det{D-1}, mc_det0..mc_det{D-1}]``.

The batch is assembled as four classes. The signal sampler yields ``S`` coherent
injections plus an ``extra_batch`` pool; an optional
:class:`~sage.data.non_astrophysical.NonAstrophysicalMasker` turns the pool into
non-astrophysical (decoherent) pairs that are dropped into *noise* slots, so:

  - signal + signal   (coherent)        -> class 1, both detectors supervised
  - signal + noise     (non-astro)      -> class 0, signal detector supervised
  - signal + signal'   (non-astro)      -> class 0, both (each own truth)
  - noise  + noise     (pure noise)     -> class 0, neither supervised

The non-astrophysical pairs eat the noise budget (never the class-1 signal
budget) so the class balance is preserved. With no masker / ``extra_batch=0``
this reduces to the 2-class (coherent-signal vs noise) regime. TRAINING ONLY.
"""

import torch

from tqdm import tqdm
from contextlib import nullcontext

from sage.core.config import get_cfg, get_data_cfg
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
        # Window length (s) used to normalise the per-detector tc target to match
        # the model's window-normalised mu_tc.
        self.tc_scale = float(get_data_cfg().sample_length_in_s)
        # target layout: [pe..., class | tc_det0..tc_det{D-1} | mc_det0..mc_det{D-1}]
        self.merged_width = self.num_point_estimate + 1
        self.full_width = self.merged_width + 2 * self.num_detectors

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
        S = self.S
        mw = self.merged_width                  # num_pe + 1
        fw = self.full_width                    # mw + 2*D
        tc0, mc0 = mw, mw + D                    # per-detector tc / mc column offsets

        for nbatch in tqdm(range(self.num_iterations)):

            # ── 1. Sample ──────────────────────────────────────────────────
            # Signal sampler yields S coherent signals + `extra` pool signals.
            signal_data, signal_targets = self.signal_sampler()   # (S+extra, D, F)
            noise_data, noise_targets = self.noise_sampler()      # (B, D, F), (B, 1)

            if self._selector is not None:
                noise_data = self._selector(noise_data)

            coh_data = signal_data[:S]
            coh_tgt = signal_targets[:S]                          # class 1, full width
            coh_mask = torch.ones(S, D, device=device, dtype=signal_targets.dtype)

            # ── 1b. Non-astrophysical pool -> class-0 injections (training only).
            # The extra pool signals are decohered and dropped into noise slots,
            # so they eat the noise budget, never the class-1 signal budget.
            extra = 0
            if self.masker is not None and signal_data.shape[0] > S:
                pool_data = signal_data[S:]
                pool_tc = signal_targets[S:, tc0:mc0]            # (extra, D)
                pool_mc = signal_targets[S:, mc0 : mc0 + D]      # (extra, D)
                na_data, na_tc, na_mc, na_mask = self.masker(
                    pool_data, pool_tc, pool_mc
                )
                extra = na_data.shape[0]
                na_tgt = torch.zeros(
                    extra, fw, device=device, dtype=signal_targets.dtype
                )                                                # class col 0 (noise)
                na_tgt[:, tc0:mc0] = na_tc
                na_tgt[:, mc0 : mc0 + D] = na_mc

            # ── 2. Assemble B slots: S coherent (cls 1), `extra` non-astro
            #        (cls 0), the rest pure noise. ───────────────────────────
            perm = torch.randperm(B, device=device)
            coh_slots = perm[:S]
            na_slots = perm[S : S + extra]

            inj = torch.zeros_like(noise_data)
            targets = torch.zeros(B, fw, device=device, dtype=signal_targets.dtype)
            per_det_mask = torch.zeros(B, D, device=device, dtype=coh_mask.dtype)

            # pure-noise class label for every slot first (0), then overwrite the
            # injected slots with their own full-width targets.
            targets[:, num_pe : num_pe + 1] = noise_targets

            inj[coh_slots] = coh_data
            targets[coh_slots] = coh_tgt
            per_det_mask[coh_slots] = coh_mask

            if extra > 0:
                inj[na_slots] = na_data
                targets[na_slots] = na_tgt
                per_det_mask[na_slots] = na_mask

            x = noise_data + inj

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
                    targets[:, :mw],
                )

                # Per-detector consistency NLL (per-detector supervision mask).
                # tc is window-normalised to match the model's normalised mu_tc
                # (and put it on the same ~unit scale as the standardised mc).
                tc_target = targets[:, tc0:mc0] / self.tc_scale  # (B, D) in [0, 1]
                mc_target = targets[:, mc0 : mc0 + D]            # (B, D) std mchirp
                cons = self.consistency_loss(
                    out.mu_tc, out.sigma_tc, out.mu_mc, out.sigma_mc,
                    tc_target, mc_target, per_det_mask,
                )

                # Warm up the consistency term linearly over the first epoch so
                # the classifier and PE heads settle before the coherence loss
                # engages (avoids early heteroscedastic spikes dominating).
                if nepoch == 0:
                    cons_w = self.consistency_weight * (nbatch + 1) / self.num_iterations
                else:
                    cons_w = self.consistency_weight
                total = merged[0] + cons_w * cons[0]

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

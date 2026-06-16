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

import math

import torch
import torch._functorch.config

# The gradient-norm balancer calibrates by measuring per-loss gradients with
# extra backward(retain_graph=True) passes over the model's graph. torch.compile's
# donated-buffer optimisation frees/reuses those buffers and is incompatible with
# retain_graph, so a compiled model raises at the first calibration. Disabling it
# (small memory cost, no speed cost) keeps retain_graph working under compile.
torch._functorch.config.donated_buffer = False

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
        aux_weights=None,
        balance_target: float = 0.33,
        balance_every: int = 250,
        balance_decay: float = 0.7,
        balance_floor_frac: float = 0.1,
        balance_denom_floor: float = 0.1,
        balance_settle: int = 500,
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

        # ── Gradient-norm budget balancer ─────────────────────────────────
        # Every ``balance_every`` steps (after a ``balance_settle`` warm-up that
        # skips the huge step-0 gradient) we measure ‖g_bce‖ and each auxiliary
        # loss's ‖g_i‖, then set per-aux weights so that (a) the aux gradient
        # norms are EQUALISED among themselves and (b) their combined norm is
        # <= balance_target * BCE's gradient scale. The budget mostly *tracks*
        # BCE (aux saturates with it), with two small guards:
        #   * budget floor  B_ref = max(EMA(‖g_bce‖), floor_frac * g_bce_char)
        #     where g_bce_char is the settled scale captured at the first
        #     calibration — keeps aux from fully vanishing if BCE flattens,
        #     without the brittle peak-tracking;
        #   * per-aux denom floor g_i_eff = max(‖g_i‖, denom_floor * EMA(‖g_i‖))
        #     — a converged aux's weight saturates at a finite ceiling instead
        #       of 1/‖g_i‖ -> infinity.
        # w_i = (balance_target / n) * B_ref / g_i_eff.
        # balance_target <= 0 disables balancing (fixed consistency_weight path).
        self.balance_target = float(balance_target)
        self.balance_every = int(balance_every)
        self.balance_decay = float(balance_decay)        # EMA decay per calibration
        self.balance_floor_frac = float(balance_floor_frac)   # floor as frac of g_bce_char
        self.balance_denom_floor = float(balance_denom_floor)  # mu
        self.balance_settle = int(balance_settle)        # skip the init transient
        # Auxiliary loss layout in the (merged, cons) component vectors:
        #   merged = [total, bce, pe_reg, coupling];  cons = [total, tc, mc]
        self._aux_names = ["pe_reg", "coupling", "cons_tc", "cons_mc"]
        n_aux = len(self._aux_names)
        self._gstep = 0                             # global step (across epochs)
        self._ema_bce = None
        self._gbce_char = None                      # settled BCE-gradient scale
        self._ema_aux = [None] * n_aux
        # ``aux_weights`` (predefined, measured offline) -> FIXED-weight mode: no
        # live gradient calibration, so nothing is measured through the compiled
        # model. This is the production path. None -> live calibration (used by
        # the offline eager measurement that *derives* these weights).
        self._fixed = aux_weights is not None
        if self._fixed:
            assert len(aux_weights) == n_aux, "aux_weights must have one per aux loss"
            self._weights = [float(w) for w in aux_weights]
        else:
            self._weights = [0.0] * n_aux           # set by live calibration
        self._last_weights = list(self._weights)    # for inspection/logging

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

    def _grad_norm(self):
        """Global L2 norm of the gradients currently on the model parameters."""
        sq = [
            (p.grad.detach() ** 2).sum()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0

    @staticmethod
    def _grad_norm_of(grads):
        """L2 norm of a tuple of gradients (some entries may be None)."""
        sq = [(g.detach() ** 2).sum() for g in grads if g is not None]
        return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0

    def _ema(self, prev, new):
        d = self.balance_decay
        return new if prev is None else d * prev + (1.0 - d) * new

    def _calibrate_weights(self, net_input, recompute):
        """Recompute the per-aux weights from per-loss gradient norms.

        The norms are measured on a clean EAGER forward of the underlying module
        (``model._orig_mod`` when compiled): measuring through the *compiled*
        graph's retain_graph multi-backward is unreliable (donated buffers, fp16
        underflow, zero gradients). ``net_input`` is reused, so the only extra
        cost is one small forward + ``n+1`` ``autograd.grad`` calls every
        ``balance_every`` steps. Any non-finite measurement is skipped so it can
        never poison the running EMAs.
        """
        eager = getattr(self.model, "_orig_mod", self.model)
        params = [p for p in eager.parameters() if p.requires_grad]
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.cfg.autocast
            else nullcontext()
        ):
            out = eager(net_input)
            bce_term, aux_terms = recompute(out)
        g_bce = self._grad_norm_of(
            torch.autograd.grad(bce_term, params, retain_graph=True, allow_unused=True)
        )
        aux_norms = [
            self._grad_norm_of(
                torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            )
            for t in aux_terms
        ]

        # don't let a non-finite measurement poison the EMAs / weights.
        if not (math.isfinite(g_bce) and all(math.isfinite(x) for x in aux_norms)):
            return

        # budget reference: tracks BCE (EMA), floored at a fraction of the
        # settled BCE-gradient scale (captured once at the first calibration) so
        # aux doesn't fully vanish if BCE flattens — no brittle peak-tracking.
        self._ema_bce = self._ema(self._ema_bce, g_bce)
        if self._gbce_char is None:
            self._gbce_char = g_bce
        B_ref = max(self._ema_bce, self.balance_floor_frac * self._gbce_char)

        n = len(aux_terms)
        for i, g_i in enumerate(aux_norms):
            self._ema_aux[i] = self._ema(self._ema_aux[i], g_i)
            # denominator floor: a converged aux (g_i -> 0) gets a finite ceiling
            # weight instead of 1/g_i -> infinity.
            g_eff = max(g_i, self.balance_denom_floor * self._ema_aux[i])
            self._weights[i] = (self.balance_target / n) * B_ref / (g_eff + 1e-12)

        # inspection (last calibration): raw norms, budget, weighted aux norms.
        self._last_g_bce = g_bce
        self._last_aux_norms = list(aux_norms)
        self._last_B_ref = float(B_ref)

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

            # ── 5b. Combine: BCE + gradient-balanced auxiliary losses. ─────
            # Auxiliary terms (raw, pre-weight): pe_reg, coupling, cons_tc,
            # cons_mc. The balancer's per-aux weights (recomputed every
            # `balance_every` steps) equalise their gradient norms within a
            # budget of balance_target * BCE's characteristic scale.
            bce = merged[1]
            aux = [merged[2], merged[3], cons[1], cons[2]]

            if self._fixed or self.balance_target > 0.0:
                # Live-calibration mode only (the offline measurement): recompute
                # per-loss gradient norms on a clean eager forward. Fixed-weight
                # mode (production) never measures gradients -> compile-safe.
                if not self._fixed and (
                    self._gstep >= self.balance_settle
                    and self._gstep % self.balance_every == 0
                ):
                    def _recompute(o):
                        m = self.merged_loss(
                            (o.ranking_stat, o.point_estimates), targets[:, :mw]
                        )
                        c = self.consistency_loss(
                            o.mu_tc, o.sigma_tc, o.mu_mc, o.sigma_mc,
                            tc_target, mc_target, per_det_mask,
                        )
                        return m[1], [m[2], m[3], c[1], c[2]]
                    self._calibrate_weights(net_input, _recompute)
                # Warm up over the first epoch so BCE/PE settle before the aux
                # losses fully engage.
                warmup = (nbatch + 1) / self.num_iterations if nepoch == 0 else 1.0
                weights = [warmup * w for w in self._weights]
                self._last_weights = weights
                total = bce + sum(w * t for w, t in zip(weights, aux))
            else:
                # Balancing disabled: fall back to merged-loss internal weights
                # plus a fixed consistency weight.
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
            self._gstep += 1
            # Log [total, BCE, consistency-total] — BCE is the primary term and
            # the reference the aux losses are balanced against.
            self.loss_components[nepoch] += torch.stack(
                [total.detach(), bce.detach(), cons[0].detach()]
            )

        self.loss_components[nepoch] /= self.num_iterations

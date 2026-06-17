#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic gradient-norm budget balancer for multi-loss training.

This is the mechanism for combining a single primary loss (typically BCE) with
``n`` auxiliary losses by their *gradient norms* — it has nothing to do with any
particular model and is reusable by any job that optimises more than one loss.
It was extracted verbatim from the consistency training loop.

Every ``balance_every`` steps (after a ``balance_settle`` warm-up that skips the
huge step-0 gradient) we measure ``‖g_bce‖`` and each auxiliary loss's ``‖g_i‖``,
then set per-aux weights so that (a) the aux gradient norms are EQUALISED among
themselves and (b) their combined norm is ``<= balance_target * BCE's gradient
scale``. The budget mostly *tracks* BCE (aux saturates with it), with two small
guards:

  * budget floor  ``B_ref = max(EMA(‖g_bce‖), floor_frac * g_bce_char)`` where
    ``g_bce_char`` is the settled scale captured at the first calibration —
    keeps aux from fully vanishing if BCE flattens, without the brittle
    peak-tracking;
  * per-aux denom floor ``g_i_eff = max(‖g_i‖, denom_floor * EMA(‖g_i‖))`` — a
    converged aux's weight saturates at a finite ceiling instead of
    ``1/‖g_i‖ -> infinity``.

``w_i = (balance_target / n) * B_ref / g_i_eff``.

The caller supplies a ``recompute`` closure — ``recompute(model_output) ->
(bce_term, [aux_term, ...])`` — so the balancer stays agnostic about which losses
it is balancing.
"""

import math

import torch
import torch._functorch.config

from contextlib import nullcontext

# The balancer calibrates by measuring per-loss gradients with extra
# backward(retain_graph=True) passes over the model's graph. torch.compile's
# donated-buffer optimisation frees/reuses those buffers and is incompatible with
# retain_graph, so a compiled model raises at the first calibration. Disabling it
# (small memory cost, no speed cost) keeps retain_graph working under compile.
torch._functorch.config.donated_buffer = False


class GradientNormBalancer:
    """Combine ``bce + sum(w_i * aux_i)`` with gradient-norm-balanced weights.

    Parameters
    ----------
    n_aux : int
        Number of auxiliary losses being balanced against the primary (BCE).
    balance_target : float
        Budget cap: combined aux gradient norm ``<= balance_target * BCE scale``.
        ``<= 0`` is not used here (a job with no balancing simply passes no
        balancer); see the trainer's no-balancer path.
    balance_every : int
        Re-calibrate the weights every this many steps.
    balance_decay : float
        EMA decay applied to the measured gradient norms per calibration.
    balance_floor_frac : float
        Budget floor as a fraction of the settled BCE-gradient scale.
    balance_denom_floor : float
        Per-aux denominator floor (mu): caps a converged aux's weight.
    balance_settle : int
        Skip calibration for this many initial steps (the step-0 transient).
    aux_weights : sequence of float or None
        If given, FIXED-weight mode: no live gradient calibration (compile-safe
        production path). ``None`` -> live calibration (used to *derive* these
        weights offline).
    autocast : bool
        Whether the calibration forward runs under fp16 autocast (match training).
    aux_names : sequence of str or None
        Optional labels for the aux losses (inspection/logging only).
    """

    def __init__(
        self,
        n_aux,
        balance_target: float = 0.33,
        balance_every: int = 250,
        balance_decay: float = 0.7,
        balance_floor_frac: float = 0.1,
        balance_denom_floor: float = 0.1,
        balance_settle: int = 500,
        aux_weights=None,
        autocast: bool = False,
        aux_names=None,
    ):
        self.balance_target = float(balance_target)
        self.balance_every = int(balance_every)
        self.balance_decay = float(balance_decay)        # EMA decay per calibration
        self.balance_floor_frac = float(balance_floor_frac)   # floor as frac of g_bce_char
        self.balance_denom_floor = float(balance_denom_floor)  # mu
        self.balance_settle = int(balance_settle)        # skip the init transient
        self.autocast = bool(autocast)
        self.n_aux = int(n_aux)
        self.aux_names = (
            list(aux_names) if aux_names is not None
            else [f"aux{i}" for i in range(self.n_aux)]
        )
        self._gstep = 0                             # global step (across epochs)
        self._ema_bce = None
        self._gbce_char = None                      # settled BCE-gradient scale
        self._ema_aux = [None] * self.n_aux
        # ``aux_weights`` (predefined, measured offline) -> FIXED-weight mode: no
        # live gradient calibration, so nothing is measured through the compiled
        # model. This is the production path. None -> live calibration (used by
        # the offline eager measurement that *derives* these weights).
        self._fixed = aux_weights is not None
        if self._fixed:
            assert len(aux_weights) == self.n_aux, "aux_weights must have one per aux loss"
            self._weights = [float(w) for w in aux_weights]
        else:
            self._weights = [0.0] * self.n_aux      # set by live calibration
        self._last_weights = list(self._weights)    # for inspection/logging

    @staticmethod
    def _grad_norm_of(grads):
        """L2 norm of a tuple of gradients (some entries may be None)."""
        sq = [(g.detach() ** 2).sum() for g in grads if g is not None]
        return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0

    def _ema(self, prev, new):
        d = self.balance_decay
        return new if prev is None else d * prev + (1.0 - d) * new

    def _calibrate_weights(self, model, net_input, recompute):
        """Recompute the per-aux weights from per-loss gradient norms.

        The norms are measured on a clean EAGER forward of the underlying module
        (``model._orig_mod`` when compiled): measuring through the *compiled*
        graph's retain_graph multi-backward is unreliable (donated buffers, fp16
        underflow, zero gradients). ``net_input`` is reused, so the only extra
        cost is one small forward + ``n+1`` ``autograd.grad`` calls every
        ``balance_every`` steps. Any non-finite measurement is skipped so it can
        never poison the running EMAs.
        """
        eager = getattr(model, "_orig_mod", model)
        params = [p for p in eager.parameters() if p.requires_grad]
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.autocast
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

    def combine(self, bce, aux_terms, model, net_input, recompute, warmup: float = 1.0):
        """Return ``bce + sum(w_i * aux_i)``, recalibrating weights on schedule.

        Parameters
        ----------
        bce : torch.Tensor
            The primary (reference) loss term to backpropagate.
        aux_terms : list[torch.Tensor]
            The ``n_aux`` auxiliary loss terms (raw, pre-weight).
        model, net_input :
            The (possibly compiled) model and the current preprocessed input —
            reused for the clean eager calibration forward.
        recompute : callable
            ``recompute(model_output) -> (bce_term, [aux_term, ...])`` rebuilding
            the same loss terms from a fresh forward, for gradient measurement.
        warmup : float
            Multiplier applied to all aux weights this step (e.g. ramp 0->1 over
            the first epoch so BCE/PE settle before the aux losses fully engage).
        """
        if not self._fixed and (
            self._gstep >= self.balance_settle
            and self._gstep % self.balance_every == 0
        ):
            self._calibrate_weights(model, net_input, recompute)
        weights = [warmup * w for w in self._weights]
        self._last_weights = weights
        total = bce + sum(w * t for w, t in zip(weights, aux_terms))
        self._gstep += 1
        return total

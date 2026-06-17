#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loss adapters — a uniform protocol so :class:`SageVanillaTraining` can combine an
arbitrary main loss with auxiliary losses under a
:class:`~sage.architecture.custom_losses.GradientNormBalancer`.

Different losses have different call signatures (``BCEWithPEsigmaLoss`` takes a
``(ranking, point_estimates)`` tuple and sliced targets; ``ConsistencyNLLLoss``
takes per-detector tensors + a mask read from the batch context). An adapter
wraps a loss so the trainer can call every loss the same way —
``adapter(out, targets, ctx) -> stacked components`` — and declares which
component is the primary (reference / BCE) term and which are the auxiliary
terms to balance.
"""

from sage.core.config import get_cfg, get_data_cfg


class LossAdapter:
    """Base loss adapter.

    Subclasses implement :meth:`components` (the underlying loss call, returning a
    stacked component tensor whose ``[0]`` is the loss's own total) and set:

    primary_index : int or None
        Index of the primary (reference) component, e.g. BCE. ``None`` if this
        adapter contributes only auxiliary terms.
    aux_indices : tuple[int]
        Indices of the auxiliary components to balance against the primary.
    """

    primary_index = 0
    aux_indices = ()

    def components(self, out, targets, ctx):
        raise NotImplementedError

    def __call__(self, out, targets, ctx):
        return self.components(out, targets, ctx)


class MergedLossAdapter(LossAdapter):
    """Wraps ``BCEWithPEsigmaLoss`` (classification + merged heteroscedastic PE).

    Returns ``[total, bce, pe_reg, coupling]``; the primary is BCE and the aux
    terms are ``pe_reg`` and ``coupling``. ``targets[:, :mw]`` is the merged-width
    slice (PE params + class column), matching the loss's existing contract.
    """

    primary_index = 1            # bce
    aux_indices = (2, 3)         # pe_reg, coupling

    def __init__(self, loss):
        cfg = get_cfg()
        self.loss = loss.to(device=cfg.device, dtype=cfg.dtype)
        self.mw = len(cfg.do_point_estimate) + 1     # PE params + class column

    def components(self, out, targets, ctx):
        return self.loss(
            (out.ranking_stat, out.point_estimates), targets[:, : self.mw]
        )


class ConsistencyLossAdapter(LossAdapter):
    """Wraps ``ConsistencyNLLLoss`` (per-detector tc / mchirp NLL).

    Returns ``[total, cons_tc, cons_mc]`` — both are auxiliary terms (no primary).
    Reads the per-detector supervision ``per_det_mask`` from ``ctx`` (populated by
    the masking callback) and slices the per-detector tc / mc targets from the
    appended target columns. ``tc`` is window-normalised (``/ tc_scale``) to match
    the model's window-normalised ``mu_tc``.
    """

    primary_index = None
    aux_indices = (1, 2)         # cons_tc, cons_mc

    def __init__(self, loss):
        cfg = get_cfg()
        self.loss = loss.to(device=cfg.device, dtype=cfg.dtype)
        D = len(cfg.detectors)
        mw = len(cfg.do_point_estimate) + 1
        self.D = D
        self.tc0, self.mc0 = mw, mw + D              # per-det tc / mc column offsets
        self.tc_scale = float(get_data_cfg().sample_length_in_s)

    def components(self, out, targets, ctx):
        tc_target = targets[:, self.tc0 : self.mc0] / self.tc_scale   # (B, D) in [0,1]
        mc_target = targets[:, self.mc0 : self.mc0 + self.D]         # (B, D) std mc
        return self.loss(
            out.mu_tc, out.sigma_tc, out.mu_mc, out.sigma_mc,
            tc_target, mc_target, ctx["per_det_mask"],
        )

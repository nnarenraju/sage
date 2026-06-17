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

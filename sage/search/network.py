#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : network.py
Description   : Wrapper exposing a trained network's per-detector and shared halves.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The frontend feature cache needs to run half a network at a time: the per-detector half
once, and the shared half once per time slide. Nothing in ``sage.architecture`` offers
that split, and adding it there would be editing a trained model's class to suit the
search.

So the split is composed here, out of the model's **own submodules**. Not a copy of its
``forward`` -- a copy is a second implementation that drifts silently the moment the
architecture is refactored, and both halves would keep returning plausible numbers.
:class:`SplitNetwork` therefore checks itself at construction: running the two halves must
reproduce ``model.forward`` **bitwise** on a probe, or the wrapper refuses to exist.

That check is the whole point of the class. It converts "the search reaches into the
model's internals and hopes they still mean what they meant" into a statement verified
every time a campaign starts.
"""

from typing import Any, Optional, Sequence, Tuple

__all__ = ["SplitNetwork", "SeparabilityReport"]


# Submodule names the split is composed from. Read from the model rather than assumed, so
# an architecture that renames one fails at construction with the name it was looking for
# instead of scoring every window through a half-built graph.
_FRONTEND_PARTS: Tuple[str, ...] = ("norm", "frontend")
_BACKEND_PARTS: Tuple[str, ...] = (
    "backend",
    "avg_pool_1d",
    "flatten",
    "get_ranking_statistic",
    "point_estimate_layers",
)


class SeparabilityReport:
    """
    What a separability probe measured, rather than only whether it passed.

    ``worst_gap`` is the largest absolute change seen in a detector's frontend output when
    a *different* detector was perturbed. Zero means bitwise separable. It is reported
    rather than reduced to a boolean because the size of a violation says what it is: a
    value at the level of float32 rounding is a numerically-coupled reduction, and a large
    one is a genuine architectural dependence.
    """

    def __init__(self, separable: bool, worst_gap: float, worst_pair, n_detectors: int):
        self.separable = bool(separable)
        self.worst_gap = float(worst_gap)
        self.worst_pair = worst_pair
        self.n_detectors = int(n_detectors)

    def __bool__(self) -> bool:
        return self.separable

    def __repr__(self) -> str:
        if self.separable:
            return f"SeparabilityReport(separable, {self.n_detectors} detectors)"
        return (
            f"SeparabilityReport(coupled, detector {self.worst_pair[0]} -> "
            f"{self.worst_pair[1]} by {self.worst_gap})"
        )

    def as_dict(self) -> dict:
        """Flat form for provenance."""
        return {
            "separable": self.separable,
            "worst_gap": self.worst_gap,
            "worst_pair": list(self.worst_pair) if self.worst_pair else [],
            "n_detectors": self.n_detectors,
        }


class SplitNetwork:
    """
    A trained network, addressable as a per-detector half and a shared half.

    Wraps the model; does not subclass or modify it. Every call runs the model's own
    submodules, so there is one set of weights and one set of layers, not a
    reimplementation that happens to agree today.

    Parameters
    ----------
    model : nn.Module
        A trained Sage network.
    verify : bool
        Check at construction that ``backend(frontend(x))`` reproduces ``model(x)``
        bitwise. Leave it on. It is one forward pass on a two-window probe, and it is what
        makes reaching into the model's internals defensible: the architecture may be
        refactored, and this fails loudly when the split stops matching rather than
        quietly scoring every window through a graph that no longer is the network.
    sample_input : tensor, optional
        Probe for the verification, shaped ``(B, D, L)``. Built from the model's own
        detector count when omitted.

    Raises
    ------
    AttributeError
        The model does not expose the submodules the split is composed from.
    ValueError
        The two halves do not reproduce ``model.forward``.
    """

    def __init__(self, model, verify: bool = True, sample_input=None) -> None:
        import torch

        self.model = model
        self._probe_length: Optional[int] = None
        missing = [
            name
            for name in _FRONTEND_PARTS + _BACKEND_PARTS
            if not hasattr(model, name)
        ]
        if missing:
            raise AttributeError(
                f"{type(model).__name__} exposes none of {missing}, so its per-detector "
                "and shared halves cannot be addressed separately. The frontend cache "
                "needs that split; without it the search must score every slide from raw "
                "strain, which is what use_frontend_cache=False does"
            )
        self.n_detectors = int(
            getattr(model, "num_detectors", 0) or len(model.frontend) or 0
        )
        if self.n_detectors < 1:
            raise ValueError(
                f"{type(model).__name__} declares {self.n_detectors} detectors"
            )
        if verify:
            self.verify(sample_input)

    # ------------------------------------------------------------------ halves
    def whole(self, x):
        """
        The network's own forward. The authority the two halves are checked against.
        """
        return self.model(x)

    def frontend(self, x, detector: int):
        """
        One detector's frontend features, from the full batch.

        The normalisation is applied to the **whole** input first, exactly as
        ``model.forward`` does, and only then is the channel taken. Slicing before
        normalising would be a different computation -- and would make every model look
        separable, since a single channel cannot couple to its neighbours.

        Taking the full batch is not a limitation on caching. Under separability this
        value depends only on ``x[:, detector]``, which is precisely what
        :meth:`separability` measures, so a feature computed at zero lag stays valid for
        every slide that reads the same detector over the same samples.
        """
        import torch

        with torch.autocast(device_type=_device_type(x), enabled=False):
            normed = self.model.norm(x.float())
        return self.model.frontend[detector](normed[:, detector : detector + 1])

    def backend(self, features):
        """
        The shared half, on per-detector features concatenated along the channel axis.

        Returns ``(ranking_statistic, point_estimates)``, the same pair
        ``model.forward`` returns. The heads run with autocast disabled because the
        network's own ``forward`` runs them that way: a reduced-precision logit is
        quantised in steps coarse enough to be visible exactly where a false-alarm
        threshold sits.
        """
        import torch

        if not torch.is_tensor(features):
            features = torch.cat(list(features), dim=1)
        pooled = self.model.backend(features)
        pooled = self.model.flatten(self.model.avg_pool_1d(pooled))
        with torch.autocast(device_type=_device_type(pooled), enabled=False):
            pooled = pooled.float()
            statistic = self.model.get_ranking_statistic(pooled)
            raw = [layer(pooled) for layer in self.model.point_estimate_layers]
            mus = torch.cat([r[:, :1] for r in raw], dim=1)
            sigma = torch.cat([r[:, 1:] for r in raw], dim=1)
        return statistic, torch.cat([mus, sigma], dim=1)

    def split_forward(self, x):
        """The two halves composed: what :meth:`whole` must equal."""
        return self.backend([self.frontend(x, d) for d in range(self.n_detectors)])

    # ------------------------------------------------------------------ checks
    #: Probe lengths tried in turn, shortest first. A network's minimum input length is a
    #: property of its architecture rather than a convention -- a backend that downsamples
    #: repeatedly refuses an input that reaches a degenerate dimension part way through,
    #: and the production model needs 512 samples where a two-layer toy needs 64. The
    #: shortest length the model itself accepts is used, which keeps the check cheap
    #: without assuming a shape.
    _PROBE_LENGTHS: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)

    def _probe(self, sample_input=None):
        import torch

        if sample_input is not None:
            probe = sample_input.detach().clone()
            if not bool(torch.isfinite(probe).all()):
                raise ValueError(
                    "sample_input holds non-finite values, which no check here can use: "
                    "torch.equal treats NaN as unequal to itself, so a bitwise "
                    "comparison of two identical NaN outputs reports a difference and "
                    "the failure would be attributed to the architecture. Pass a finite "
                    "probe"
                )
            return probe
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        generator = torch.Generator().manual_seed(20260820)

        if self._probe_length is None:
            was_training = self.model.training
            self.model.eval()
            failure = None
            try:
                for length in self._PROBE_LENGTHS:
                    candidate = torch.randn(
                        2, self.n_detectors, length, generator=generator
                    ).to(device=device)
                    try:
                        with torch.no_grad():
                            self.model(candidate)
                    except RuntimeError as error:
                        failure = error
                        continue
                    self._probe_length = length
                    break
            finally:
                self.model.train(was_training)
            if self._probe_length is None:
                raise ValueError(
                    f"{type(self.model).__name__} accepted no probe shorter than "
                    f"{self._PROBE_LENGTHS[-1]} samples, so the split cannot be checked "
                    f"against its forward. Last failure: {failure}. Pass sample_input "
                    "with a shape this network does accept"
                )

        probe = torch.randn(
            2, self.n_detectors, self._probe_length, generator=generator
        )
        return probe.to(device=device)

    def verify(self, sample_input=None) -> None:
        """
        Assert the split reproduces ``model.forward`` bitwise.

        Bitwise, not ``allclose``. The halves run the same layers on the same weights in
        the same order, so agreement is exact or the composition is wrong; a tolerance
        here would accept a graph that had genuinely diverged from the network.
        """
        import torch

        probe = self._probe(sample_input)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                expected = self.whole(probe)
                actual = self.split_forward(probe)
        finally:
            self.model.train(was_training)

        pairs = list(zip(_as_tuple(expected), _as_tuple(actual)))
        for index, (want, got) in enumerate(pairs):
            if want.shape != got.shape:
                raise ValueError(
                    f"the split of {type(self.model).__name__} returns output {index} "
                    f"shaped {tuple(got.shape)} where forward returns "
                    f"{tuple(want.shape)}; the wrapper is composing a different graph "
                    "from the network's own"
                )
            if not torch.equal(want, got):
                gap = float((want - got).abs().max())
                raise ValueError(
                    f"the split of {type(self.model).__name__} does not reproduce its "
                    f"forward: output {index} differs by up to {gap}. The wrapper "
                    "composes the model's own submodules, so this means the architecture "
                    "changed shape and the split no longer is the network. Fix the "
                    "wrapper rather than loosening this check -- a search running the "
                    "wrong graph returns a ranking statistic for every window and no error"
                )

    def separability(self, sample_input=None) -> SeparabilityReport:
        """
        Measure whether a detector's frontend output depends on any other detector.

        Replace one detector's samples with a different realisation, re-run every *other*
        detector's frontend, and compare bitwise against the unperturbed baseline. Repeat
        for every detector. Replacement rather than a shift, because a time slide -- the
        operation the cache has to survive -- substitutes samples; an added constant
        leaves every shift-invariant statistic of the channel intact and a frontend
        coupled to one of those would pass.

The probe is deterministic random values rather than zeros or a constant. A
        constant input has zero variance, so a variance-normalising layer maps it to a
        degenerate output dominated by its epsilon: the frontend then sees almost nothing,
        and a coupling that only appears on structured input would be missed. (A constant
        probe does still catch a mean-coupled layer such as ``GroupNorm``, measured -- so
        this is about coverage, not about the check being useless.) Deterministic so that
        a campaign's separability verdict does not depend on a seed.

        What this licenses is the frontend cache. A separable frontend means a detector's
        features depend on that detector's samples alone, so they can be computed once and
        reused across every slide that re-pairs it -- the difference between scoring the
        background once and scoring it once per slide. It is a *measurement at one input*,
        not a proof for all inputs; the structural argument is what makes it convincing,
        and this is the guard against a refactor quietly invalidating it.
        """
        import torch

        if self.n_detectors < 2:
            raise ValueError(
                f"separability needs at least two detectors, found {self.n_detectors}"
            )
        probe = self._probe(sample_input)
        was_training = self.model.training
        self.model.eval()
        worst_gap = 0.0
        worst_pair = None
        try:
            with torch.no_grad():
                baseline = [
                    self.frontend(probe, d).detach().clone()
                    for d in range(self.n_detectors)
                ]
                generator = torch.Generator().manual_seed(20260821)
                for perturbed in range(self.n_detectors):
                    moved = probe.detach().clone()
                    # Replace the channel outright rather than offsetting it. The
                    # perturbation has to be the one the cache would actually apply, and
                    # that is a time slide: different samples in one detector, the others
                    # untouched. A uniform offset is not -- it leaves every shift- and
                    # scale-invariant statistic of the channel exactly where it was, so a
                    # frontend coupled to a neighbour's variance, its spectrum or its
                    # rank order returns bit-identical output and the gate reports
                    # separable. Substitution moves every statistic at once.
                    moved[:, perturbed] = torch.randn(
                        moved.shape[0],
                        moved.shape[2],
                        generator=generator,
                        dtype=moved.dtype,
                    ).to(device=moved.device)
                    for other in range(self.n_detectors):
                        if other == perturbed:
                            continue
                        after = self.frontend(moved, other)
                        if not torch.equal(after, baseline[other]):
                            gap = float((after - baseline[other]).abs().max())
                            # A non-finite gap is still a difference: the probe is finite
                            # by construction, so an output that is not is one this
                            # detector produced from another's samples. Recording it as
                            # "no coupling found" because NaN fails every comparison is
                            # the one way this check can fail open.
                            if not gap == gap or gap > worst_gap:
                                worst_gap, worst_pair = gap, (perturbed, other)
        finally:
            self.model.train(was_training)
        return SeparabilityReport(
            separable=worst_pair is None,
            worst_gap=worst_gap,
            worst_pair=worst_pair,
            n_detectors=self.n_detectors,
        )


def _as_tuple(value) -> Tuple:
    """A model output as a tuple, whether it returned one tensor or several."""
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _device_type(tensor) -> str:
    """Autocast device type for a tensor, without importing torch at module scope."""
    return tensor.device.type

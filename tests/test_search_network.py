#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_network.py
Description   : The wrapper that splits a trained network into its two halves.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

The search runs half a network at a time when the frontend cache is on. Nothing in
sage.architecture offers that split and the search does not add it there, so the split is
composed here out of the model's own submodules.

The composition is the risk: a copy of a network's forward drifts the moment the
architecture is refactored, and both halves keep returning plausible numbers. Every test
below exists to make that drift loud.
"""

import pytest

torch = pytest.importorskip("torch")

from sage.search.network import SplitNetwork
from tests.search_fixtures import ToyFrontendNet


class TestSplitReproducesForward:
    """The composition must be the network, not merely resemble it."""

    def test_split_equals_forward_bitwise(self):
        """
        Running the two halves reproduces ``model.forward`` exactly.

        Bitwise, not ``allclose``: the halves run the same layers on the same weights in
        the same order, so agreement is exact or the composition is wrong. A tolerance
        would accept a graph that had genuinely diverged from the network.
        """
        model = ToyFrontendNet(2, "instancenorm").eval()
        split = SplitNetwork(model, verify=True)
        probe = torch.randn(3, 2, 64)

        with torch.no_grad():
            expected = model(probe)
            actual = split.split_forward(probe)

        for want, got in zip(expected, actual):
            assert torch.equal(want, got)

    def test_verification_runs_at_construction(self):
        """
        The check is not opt-in at the call site; it happens when the wrapper is built.

        A campaign that constructs the wrapper has already proved the split is the
        network. Deferring it to first use would put the failure inside a scoring loop.
        """
        model = ToyFrontendNet(2, "instancenorm")
        SplitNetwork(model)  # would raise if the halves disagreed

    def test_drifted_backend_is_caught(self):
        """
        A model whose forward stops matching its own submodules is refused.

        This is the failure the wrapper exists to make loud: reaching into another
        module's internals is only defensible if something checks they still mean what
        they meant.
        """
        model = ToyFrontendNet(2, "instancenorm").eval()
        original = model.forward

        def drifted(x):
            statistic, point = original(x)
            return statistic + 1.0, point

        model.forward = drifted
        with pytest.raises(ValueError, match="does not reproduce its forward"):
            SplitNetwork(model)

    def test_missing_submodule_named(self):
        """A model without the parts says which ones it lacks."""
        with pytest.raises(AttributeError, match="norm"):
            SplitNetwork(torch.nn.Linear(4, 4))

    def test_probe_length_adapts_to_the_architecture(self):
        """
        The probe length is discovered, not assumed.

        A backend that downsamples repeatedly refuses an input that reaches a degenerate
        dimension part way through: the production network needs 512 samples where this
        toy needs 64. Assuming one length would make the wrapper unusable on the other.
        """
        split = SplitNetwork(ToyFrontendNet(2, "instancenorm"))

        assert split._probe_length in SplitNetwork._PROBE_LENGTHS


class TestSeparability:
    """Whether a detector's frontend output depends on any other detector."""

    def test_instancenorm_is_separable(self):
        """
        Per-channel normalisation leaves each detector's frontend independent.

        This is what licenses the frontend cache: a feature computed once is reusable
        across every slide that re-pairs the detector against different data.
        """
        report = SplitNetwork(ToyFrontendNet(2, "instancenorm")).separability()

        assert report.separable
        assert report.worst_gap == 0.0
        assert bool(report) is True

    def test_groupnorm_is_not_separable(self):
        """
        A normalisation spanning the detector axis couples every detector to every other.

        The negative control. Without it the measurement would pass on any model whose
        frontend happened to be an identity, and the cache would be enabled on a network
        it silently corrupts.
        """
        report = SplitNetwork(ToyFrontendNet(2, "groupnorm")).separability()

        assert not report.separable
        assert report.worst_gap > 0.0
        assert report.worst_pair is not None

    def test_shift_invariant_coupling_caught(self):
        """
        The perturbation has to be the one a slide applies. ``sharedscale`` couples every
        detector through a pooled standard deviation, which adding a constant to a
        channel leaves exactly where it was -- so an offset probe reports it separable
        and licenses a feature cache that is not valid.
        """
        from tests.search_fixtures import ToyFrontendNet

        model = ToyFrontendNet(2, "sharedscale").eval()
        report = SplitNetwork(model).separability()
        assert not report.separable
        assert report.worst_gap > 0.0

    def test_non_finite_probe_refused(self):
        """
        ``torch.equal`` calls NaN unequal to itself, so a probe carrying one makes every
        model look broken. Refused where it enters, and blamed on the probe.
        """
        import torch

        from tests.search_fixtures import ToyFrontendNet

        probe = torch.randn(2, 2, 64)
        probe[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            SplitNetwork(ToyFrontendNet(2, "instancenorm").eval(), sample_input=probe)

    def test_report_names_the_coupled_pair(self):
        """Which pair couples says where in the graph the coupling is."""
        report = SplitNetwork(ToyFrontendNet(3, "groupnorm")).separability()
        perturbed, other = report.worst_pair

        assert perturbed != other
        assert {perturbed, other} <= {0, 1, 2}
        assert "coupled" in repr(report)

    def test_probe_is_random_and_reproducible(self):
        """
        The default probe is structured and deterministic.

        Structured because a constant input has zero variance, so a variance-normalising
        layer maps it to a degenerate output dominated by its epsilon -- the frontend then
        sees almost nothing, and a coupling that only appears on real input would be
        missed. Deterministic because a campaign's separability verdict must not depend on
        a seed.
        """
        split = SplitNetwork(ToyFrontendNet(2, "instancenorm"))
        first, second = split._probe(), split._probe()

        assert torch.equal(first, second)
        assert float(first.std()) > 0.1

    def test_constant_probe_is_degenerate(self):
        """
        A constant probe still catches mean coupling, but tells you less.

        Recorded so the reason for the default is a measurement rather than folklore: a
        constant input does expose ``GroupNorm``, and the objection to it is coverage --
        the frontend is fed a near-zero signal, so anything that couples only on
        structured input goes unseen.
        """
        constant = torch.full((2, 2, 64), 3.0)

        assert SplitNetwork(ToyFrontendNet(2, "groupnorm")).separability(
            constant
        ).separable is False

    def test_single_detector_refused(self):
        """Separability is a statement about a pair; one detector has no pair."""
        with pytest.raises(ValueError, match="at least two detectors"):
            SplitNetwork(ToyFrontendNet(1, "instancenorm")).separability()

    def test_training_mode_restored(self):
        """The probe must not leave the model in eval mode behind the caller's back."""
        model = ToyFrontendNet(2, "instancenorm")
        model.train()
        SplitNetwork(model).separability()

        assert model.training

    def test_report_is_recordable(self):
        """The measurement goes into provenance as numbers, not as a boolean."""
        payload = SplitNetwork(ToyFrontendNet(2, "instancenorm")).separability().as_dict()

        assert payload["separable"] is True
        assert payload["n_detectors"] == 2
        assert payload["worst_gap"] == 0.0

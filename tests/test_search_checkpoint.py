#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_checkpoint.py
Description   : Checkpoint loading, stored-vs-live geometry, and the separability gate.

Created on 2026-08-19

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The separability check is the one that matters. The frontend feature cache reuses a
detector's features across every time slide, which is only legal if that detector's
frontend output depends on that detector alone. A config string saying ``instancenorm``
does not establish it and does not survive a refactor, so the property is measured on the
module graph -- bitwise, in both polarities, with a GroupNorm negative control that must
fail. Without that control the check passes on any model whose frontend is an identity.
"""

import pytest

torch = pytest.importorskip("torch")

from sage.search.checkpoint import (  # noqa: E402
    GEOMETRY_KEYS,
    LoadedCheckpoint,
    as_config,
    assert_separable,
    build_search_model,
    load_search_model,
    read_checkpoint,
    validate_geometry,
)
from tests.search_fixtures import (  # noqa: E402
    ToyFrontendNet,
    make_legacy_checkpoint,
    make_synthetic_checkpoint,
)


def _toy_factory(cfg, data_cfg):
    return ToyFrontendNet(len(cfg.detectors), cfg.norm_type)


class TestReading:
    """What a checkpoint has to carry to be usable."""

    def test_round_trip(self, tmp_path):
        """The stored configs and weights come back as written."""
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))

        assert ckpt.epoch == 7
        assert ckpt.norm_type == "instancenorm"
        assert ckpt.detectors == ("H1", "L1")
        assert ckpt.cfg["train_runs"] == ["O3b"]
        assert ckpt.data_cfg["sample_rate"] == 2048.0
        assert len(ckpt.sha256) == 64

    def test_digest_is_of_the_file(self, tmp_path):
        """
        Two checkpoints with identical weights still get different digests if the files
        differ, because the digest identifies the artefact a result came from.
        """
        first = read_checkpoint(make_synthetic_checkpoint(tmp_path / "a.pt", epoch=1))
        second = read_checkpoint(make_synthetic_checkpoint(tmp_path / "b.pt", epoch=2))

        assert first.sha256 != second.sha256

    def test_compile_prefix_stripped(self, tmp_path):
        """
        ``torch.compile`` prefixes every key, and left in place the weights load into
        nothing. Stripping it here is what the three ad-hoc loaders each re-implemented.
        """
        path = make_synthetic_checkpoint(tmp_path / "compiled.pt")
        payload = torch.load(path, weights_only=False)
        payload["model_state_dict"] = {
            f"_orig_mod.{name}": tensor
            for name, tensor in payload["model_state_dict"].items()
        }
        torch.save(payload, path)
        ckpt = read_checkpoint(path)

        assert ckpt.state_dict
        assert not any(name.startswith("_orig_mod.") for name in ckpt.state_dict)

    def test_legacy_object_config_refused(self, tmp_path):
        """
        A checkpoint storing a live config object is refused by name.

        Every checkpoint written before configs were flattened is in this state, and the
        native failure is a ModuleNotFoundError from inside pickle naming a module with
        nothing to do with the search.
        """
        path = make_legacy_checkpoint(tmp_path / "legacy.pt")
        with pytest.raises(ValueError, match="rather than a dict"):
            read_checkpoint(path)

    def test_missing_file_and_keys_refused(self, tmp_path):
        """An absent file, and a file that is not a Sage checkpoint, both name the fault."""
        with pytest.raises(FileNotFoundError):
            read_checkpoint(tmp_path / "absent.pt")

        path = tmp_path / "wrong.pt"
        torch.save({"cfg": {}, "data_cfg": {}}, path)
        with pytest.raises(ValueError, match="model_state_dict"):
            read_checkpoint(path)

    def test_detector_ordering_required(self, tmp_path):
        """
        Without a detector list the channel ordering is unknown.

        Reading the right detectors in the wrong order feeds each frontend the wrong
        strain, and every channel still holds real data, so nothing downstream notices.
        """
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        stripped = LoadedCheckpoint(
            path=ckpt.path, sha256=ckpt.sha256, state_dict=ckpt.state_dict,
            cfg={}, data_cfg={}, epoch=0, val_loss=0.0,
        )
        with pytest.raises(ValueError, match="no detector list"):
            _ = stripped.detectors


class TestGeometry:
    """Stored configuration against the live one."""

    def test_identical_config_has_no_mismatch(self, tmp_path):
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        assert validate_geometry(ckpt, ckpt.cfg, ckpt.data_cfg) == []

    def test_mismatch_raises_when_strict(self, tmp_path):
        """
        A geometry disagreement stops the run rather than being reported.

        The network produces a number for every window whatever it is fed, and nothing
        downstream can tell that number from a good one.
        """
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        live = dict(ckpt.data_cfg, sample_rate=4096.0)
        with pytest.raises(ValueError, match="different geometry"):
            validate_geometry(ckpt, ckpt.cfg, live)

        reported = validate_geometry(ckpt, ckpt.cfg, live, strict=False)
        assert any("sample_rate" in item for item in reported)

    def test_container_type_is_not_a_mismatch(self, tmp_path):
        """A detector list against a detector tuple describes the same network."""
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        live = dict(ckpt.data_cfg, detectors=("H1", "L1"))

        assert validate_geometry(ckpt, ckpt.cfg, live) == []

    def test_absence_is_reported_not_ignored(self, tmp_path):
        """
        A key present on one side only is a mismatch, not a pass.

        Treating an absent key as agreement is how a search silently runs a geometry
        nobody stated.
        """
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        reported = validate_geometry(ckpt, {}, {}, strict=False)

        assert len(reported) == len(GEOMETRY_KEYS)
        assert all("absent" in item for item in reported)


class TestSeparability:
    """The gate that decides whether the frontend cache is legal."""

    def test_instancenorm_is_separable(self):
        """Per-channel normalisation leaves each detector's frontend independent."""
        assert_separable(ToyFrontendNet(2, "instancenorm")) is None

    def test_groupnorm_is_refused(self):
        """
        The negative control, without which the check proves nothing.

        GroupNorm(1, D) spans the detector axis, so every frontend output depends on every
        input. A check that cannot fail here would also pass on a frontend that had been
        refactored into coupling.
        """
        with pytest.raises(ValueError, match="not separable"):
            assert_separable(ToyFrontendNet(2, "groupnorm"))

    def test_every_pair_is_checked(self):
        """
        All three detectors are perturbed, not just the first.

        A network separable in one direction and not another passes a single spot check,
        and that asymmetry is what a partially-refactored frontend produces.
        """
        with pytest.raises(ValueError, match="not separable"):
            assert_separable(ToyFrontendNet(3, "groupnorm"))
        assert_separable(ToyFrontendNet(3, "instancenorm"))

    def test_comparison_is_bitwise(self):
        """
        A tiny coupling fails, because the cache compounds it once per slide.

        The leak is injected into ``norm`` -- the one layer that sees every detector, and
        the only place a per-detector frontend can pick up its neighbours. Both
        ``forward`` and the split run it, so the wrapper's own composition check still
        passes and what is being measured is a genuine coupling rather than a broken
        split.

        The coupling is well below any ``allclose`` tolerance -- default ``rtol`` is
        1e-5 -- so a check written with approximate equality passes it, and the resulting
        background is subtly unlike the zero-lag it is compared against. It is kept above
        float32 epsilon (~1.2e-7) on purpose: a smaller leak is not representable and
        would be a no-op rather than a test.
        """
        model = ToyFrontendNet(2, "instancenorm")
        clean = model.norm

        class Leaky(torch.nn.Module):
            """Per-channel normalisation, plus 1e-6 of the other detector's level."""

            def __init__(self):
                super().__init__()
                self.inner = clean

            def forward(self, x):
                out = self.inner(x)
                bleed = 1e-6 * x.mean(dim=(1, 2), keepdim=True)
                return out + bleed

        model.norm = Leaky()
        with pytest.raises(ValueError, match="not separable"):
            assert_separable(model)

    def test_model_with_no_per_detector_path_refused(self):
        """
        A model that cannot isolate a detector cannot be proved separable.

        Two shapes are accepted: an explicit ``forward_frontend(x, detector)``, or the
        ``norm`` + ``frontend`` structure every Sage network has and states in its own
        ``forward`` -- which is the path the engine actually takes, so it is the one worth
        measuring. Anything else is refused rather than assumed separable.
        """
        with pytest.raises(ValueError, match="exposes none of"):
            assert_separable(torch.nn.Linear(4, 4))

    def test_groupnorm_refused_through_the_wrapper(self):
        """
        The negative control survives the move to the wrapper.

        Under ``GroupNorm(1, D)`` the normalisation spans the detector axis, so one
        detector's frontend output depends on every other's. Slicing the channel before
        normalising would hide exactly that and make every model look separable, which is
        why the wrapper normalises the whole input first, as ``forward`` does.
        """
        with pytest.raises(ValueError, match="separab|coupl|changed"):
            assert_separable(ToyFrontendNet(2, "groupnorm"))

    def test_training_mode_restored(self):
        """The probe must not leave the model in eval mode behind the caller's back."""
        model = ToyFrontendNet(2, "instancenorm")
        model.train()
        assert_separable(model)
        assert model.training


class TestBuilding:
    """Instantiating the architecture and loading the weights."""

    def test_factory_builds_and_loads(self, tmp_path):
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        model = build_search_model(ckpt, device="cpu", factory=_toy_factory)

        assert isinstance(model, ToyFrontendNet)
        assert not model.training

    def test_architecture_is_not_guessed(self, tmp_path):
        """
        A checkpoint that does not name its architecture is refused.

        Defaulting to whichever class is current would load old weights into a refactored
        one; a shape mismatch is the lucky outcome, since a class that changed only in
        behaviour loads cleanly and scores every window wrongly.
        """
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        with pytest.raises(ValueError, match="does not record which architecture"):
            build_search_model(ckpt, device="cpu")

    def test_mismatched_weights_refused(self, tmp_path):
        """
        Loading only the keys that match would leave a partly-random network.

        It would still return a ranking statistic for every window, which is the failure
        mode worth refusing rather than reporting.
        """
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        with pytest.raises(ValueError, match="do not match"):
            build_search_model(
                ckpt, device="cpu", factory=lambda cfg, dc: ToyFrontendNet(3, "instancenorm")
            )

    def test_config_wrapper_is_attribute_accessible(self, tmp_path):
        """The architecture reads its config by attribute, including with a default."""
        ckpt = read_checkpoint(make_synthetic_checkpoint(tmp_path / "best.pt"))
        cfg = as_config(ckpt.cfg)

        assert cfg.detectors == ["H1", "L1"]
        assert getattr(cfg, "use_blurpool", True) is True

    def test_load_search_model_proves_separability(self, tmp_path):
        """
        The one-call path validates geometry before building and separability before use.

        Doing it the other way round spends a GPU-hour discovering the configuration
        disagrees.
        """
        path = make_synthetic_checkpoint(tmp_path / "best.pt")
        ckpt = read_checkpoint(path)
        model, loaded = load_search_model(
            path, ckpt.cfg, ckpt.data_cfg, device="cpu",
            require_separable=True, factory=_toy_factory,
        )

        assert isinstance(model, ToyFrontendNet)
        assert loaded.sha256 == ckpt.sha256

    def test_load_refuses_a_coupled_model_when_separability_required(self, tmp_path):
        """A caller intending to use the feature cache is stopped before it scores."""
        path = make_synthetic_checkpoint(tmp_path / "gn.pt", norm_type="groupnorm")
        ckpt = read_checkpoint(path)
        with pytest.raises(ValueError, match="not separable"):
            load_search_model(
                path, ckpt.cfg, ckpt.data_cfg, device="cpu",
                require_separable=True, factory=_toy_factory,
            )


class TestSeparabilityDiagnostic:
    """The gate that decides whether the frontend cache may be used at all."""

    def test_controls_behave(self):
        """
        A gate that cannot fail proves nothing, and one that fails on everything proves
        nothing either. Both coupled frontends must be refused and the separable one
        accepted, before any verdict on a real network is worth reading.

        ``groupnorm`` couples through a normalisation group spanning the detector axis.
        ``sharedscale`` couples through a shift-invariant statistic, which is the case an
        additive perturbation cannot see.
        """
        from sage.diagnostics.diagnose_separability import controls

        verdicts = controls("cpu")

        assert verdicts["groupnorm"]["accepted"] is False
        assert verdicts["sharedscale"]["accepted"] is False
        assert verdicts["instancenorm"]["accepted"] is True

    def test_perturbation_reaches_the_network(self):
        """
        A cross-detector change of zero means nothing unless the probe moved something.
        Measured on the production network with real strain: the self change is 0.68
        while every cross change is exactly 0, so the zero is a property and not an
        inert probe.
        """
        import numpy as np
        import torch

        from sage.diagnostics.diagnose_separability import perturbation_report
        from tests.search_fixtures import ToyFrontendNet

        class _Engine:
            def __init__(self, model):
                self.model = model

            def forward_frontend(self, strain, detector):
                from sage.search.network import SplitNetwork

                with torch.inference_mode():
                    return SplitNetwork(self.model, verify=False).frontend(
                        torch.as_tensor(strain), detector
                    )

        rng = np.random.default_rng(4)
        strain = rng.normal(size=(8, 2, 64)).astype(np.float32)

        clean = perturbation_report(
            _Engine(ToyFrontendNet(num_detectors=2, norm_type="instancenorm")), strain
        )
        assert clean["separable"]
        assert clean["worst_self"] > 0.0

        coupled = perturbation_report(
            _Engine(ToyFrontendNet(num_detectors=2, norm_type="groupnorm")), strain
        )
        assert not coupled["separable"]
        assert coupled["worst_cross"] > 0.0

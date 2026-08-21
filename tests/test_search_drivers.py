#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_drivers.py
Description   : The stage entry points a campaign is actually driven through.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

Every stage exposes ``run(spec, **kwargs)`` and returns a report carrying a
``fingerprint``. The fingerprint is what decides whether re-running a stage invalidates
everything downstream of it, so a driver that omits it silently forces a full rebuild, and
one that computes it from something that does not track its product silently prevents a
needed one.

Exercised on the synthetic release: real strain is a separate, later question, and none of
the index or accounting logic below depends on it.
"""

import dataclasses
import shutil

import numpy as np
import pytest

from sage.search.spec import (
    DataSpec,
    EngineSpec,
    GeometrySpec,
    SearchSpec,
    SignificanceSpec,
    SlideSpec,
)

pytest.importorskip("h5py")


def _campaign(tmp_path, detectors=("H1", "L1"), release=None, **overrides):
    """
    A complete synthetic campaign: release, training prior and an out_dir.

    ``release`` shares one release between campaigns, which is what two arms over one
    observing run actually do -- an HL and an HLV search read the same strain.
    """
    from tests.search_fixtures import make_synthetic_release

    tmp_path.mkdir(parents=True, exist_ok=True)
    if release is None:
        release = make_synthetic_release(
            tmp_path / "release", detectors=detectors, chunk_s=64.0
        )
    prior = tmp_path / "gwconfig.yaml"
    prior.write_text(
        "priors:\n  tc:\n    name: uniform\n    min: 5.0\n    max: 7.0\n"
    )
    spec = SearchSpec(
        tag="synth",
        out_dir=tmp_path / "campaign",
        data=DataSpec(
            observing_run="O3a",
            detectors=tuple(detectors),
            release_dir=release,
            fiducial_dir=tmp_path,
            apply_cat1=False,
        ),
        engine=EngineSpec(checkpoint=tmp_path / "best.pt", gwconfig=prior),
        geometry=GeometrySpec(stride_samples=205, tc_source="gwconfig"),
        slides=SlideSpec(
            n_slides=4,
            reference_detector=detectors[0],
            min_separation_s=20.0,
            tau_max_s=1000.0,
            seed=5,
        ),
        significance=SignificanceSpec(
            n_exceedances=200, removal_modes=("inclusive",)
        ),
    )
    return dataclasses.replace(spec, **overrides)


def _background(spec, size=20000, seed=11):
    """A background standing in for the GPU stage, saved where `far` expects it."""
    from sage.search.background import BackgroundSet
    from sage.search.slides import SlidePlan
    from sage.search.triggers import histogram_stats

    stats = np.random.default_rng(seed).exponential(1.0, size) + 1.0
    plan = SlidePlan.load(spec.path("slides", "slide_plan.h5"))
    background = BackgroundSet(
        stats=stats,
        livetime_s=plan.background_livetime_s,
        n_slides=4,
        removal="inclusive",
        histogram=histogram_stats(stats, clustered=True),
    )
    background.save(spec.path("background", "bg_inclusive.h5"))
    return background


class TestEveryDriverReportsAFingerprint:
    """The cascade contract: a stage that cannot say whether its product moved."""

    def test_segments_grid_slides_carry_one(self, tmp_path):
        """
        A driver without a fingerprint invalidates its whole downstream chain on every
        re-run, whether or not anything changed. That is safe but wasteful; the point of
        the contract is that it need not be.
        """
        import sage.search.grid as grid
        import sage.search.segments as segments
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        for module in (segments, grid, slides):
            report = module.run(spec)
            assert report["fingerprint"]
            assert isinstance(report["fingerprint"], str)

    def test_fingerprint_tracks_the_configuration(self, tmp_path):
        """
        A knob that changes the product changes the fingerprint.

        Otherwise a re-run under a different ladder would report "nothing moved" and leave
        every downstream stage describing a background it no longer matches.
        """
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        first = slides.run(spec)["fingerprint"]
        other = dataclasses.replace(
            spec, slides=dataclasses.replace(spec.slides, seed=spec.slides.seed + 1)
        )
        second = slides.run(other)["fingerprint"]

        assert first != second

    def test_fingerprint_is_stable_across_reruns(self, tmp_path):
        """
        Re-running an unchanged stage reports the same fingerprint.

        This is the half that saves the campaign work: an idempotent re-run must be
        recognisable as one, or the contract only ever costs.
        """
        import sage.search.grid as grid

        spec = _campaign(tmp_path)

        assert grid.run(spec)["fingerprint"] == grid.run(spec)["fingerprint"]


class TestSegmentsDriver:
    """Coincident time and where the rest of it went."""

    def test_decomposition_closes(self, tmp_path):
        """
        Hosted plus the three losses equals the union, exactly.

        This is what makes the coverage report a check on the ownership sweep rather than
        a summary of it: a window start assigned outside the union, or two starts claiming
        the same second, fails to balance here.
        """
        import sage.search.segments as segments

        report = segments.run(_campaign(tmp_path))
        total = (
            report["hosted_s"]
            + report["lost_phase_restart_s"]
            + report["lost_window_fit_s"]
            + report["lost_boundary_holes_s"]
        )

        assert total == pytest.approx(report["union_s"], rel=0, abs=1e-6)

    def test_livetime_recorded_in_the_manifest(self, tmp_path):
        """Every rate divides by one of these numbers, so they are persisted."""
        import pathlib

        import sage.search.segments as segments
        from sage.search.manifest import RunManifest

        spec = _campaign(tmp_path)
        segments.run(spec)
        recorded = RunManifest(
            path=pathlib.Path(spec.path("manifest.h5"))
        ).summary()["livetime"]

        assert "O3a" in recorded
        assert recorded["O3a"]["coincident_livetime_s"] > 0

    def test_cat1_without_a_cache_refused(self, tmp_path):
        """
        Asking for a veto with no veto list is refused, never skipped.

        Proceeding would produce a livetime that describes unvetoed data while the
        provenance says it was vetoed -- the two disagree and nothing downstream can tell.
        """
        import sage.search.segments as segments

        spec = _campaign(tmp_path)
        spec = dataclasses.replace(
            spec, data=dataclasses.replace(spec.data, apply_cat1=True)
        )
        with pytest.raises(ValueError, match="cat1_cache_dir"):
            segments.run(spec)


class TestSlidesDriver:
    """The denominator every false-alarm rate divides by."""

    def test_plan_is_persisted(self, tmp_path):
        """
        The ladder is written, not rebuilt from a seed.

        It is the denominator: a ladder redrawn on a machine with a different numpy would
        give a background time nothing could reproduce.
        """
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        report = slides.run(spec)

        assert spec.path("slides", "slide_plan.h5").is_file()
        assert report["background_livetime_s"] > 0

    def test_keep_threshold_left_unset(self, tmp_path):
        """
        The threshold is not frozen here; `slides` cannot see the zero-lag histogram.

        Both stages depend only on `grid`, so they are submitted together. Stamping a
        threshold here would mean deriving it from something other than the complete
        foreground.
        """
        import sage.search.slides as slides
        from sage.search.slides import SlidePlan

        spec = _campaign(tmp_path)
        slides.run(spec)

        assert SlidePlan.load(spec.path("slides", "slide_plan.h5")).keep_threshold is None

    def test_livetime_is_measured_not_scaled(self, tmp_path):
        """
        Background time is summed from the plan, never `n * T_zerolag`.

        Asserted on the ladder, which is where the closed form is wrong: retention falls
        as a lag moves a detector's data off the end of the run, and the closed form
        assumes it does not.
        """
        import dataclasses

        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        spec = dataclasses.replace(
            spec, slides=dataclasses.replace(spec.slides, method="ladder")
        )
        report = slides.run(spec)
        naive = report["n_slides"] * report["foreground_livetime_s"]

        assert report["background_livetime_s"] < naive
        assert 0.0 < report["mean_slide_retention"] < 1.0

    def test_rolled_livetime_is_exact(self, tmp_path):
        """
        The roll's retention is exactly one, and that is a fact about the construction
        rather than the closed form sneaking back in: every lattice ordinal is hostable
        in every detector, so re-pairing ordinals moves nothing into a gap. The plan is
        still summed -- what changed is the answer, not where it comes from.
        """
        import sage.search.slides as slides

        report = slides.run(_campaign(tmp_path))

        assert report["mean_slide_retention"] == pytest.approx(1.0)
        assert report["background_livetime_s"] == pytest.approx(
            report["n_slides"] * report["foreground_livetime_s"]
        )


class TestFarDriver:
    """Counted rates, a fitted continuation, and the checks that judge it."""

    def _prepared(self, tmp_path):
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        slides.run(spec)
        _background(spec)
        return spec

    def test_curve_saved_per_mode(self, tmp_path):
        """One curve per removal mode the campaign asked for."""
        import sage.search.far as far

        spec = self._prepared(tmp_path)
        report = far.run(spec)

        assert set(report["curves"]) == {"inclusive"}
        assert spec.path("far", "far_curve_O3a_inclusive.h5").is_file()

    def test_shape_is_reported(self, tmp_path):
        """
        The fitted xi travels with the curve it produced.

        Reading it is how the extrapolation's character is known rather than assumed:
        positive is heavier than exponential and runs away above the threshold, negative
        is bounded above at `u - scale/xi`.
        """
        import sage.search.far as far

        detail = far.run(self._prepared(tmp_path))["checks"]["inclusive"]

        assert np.isfinite(detail["tail_shape"])
        assert detail["tail_n_exceedances"] == 200
        assert len(detail["ladder_shape"]) == len(detail["ladder_threshold"])

    def test_goodness_of_fit_reported_not_acted_on(self, tmp_path):
        """
        The p-values are recorded beside the fit; nothing branches on them.

        A driver that silently switched behaviour on a test outcome would make the
        published curve depend on a result nobody saw.
        """
        import sage.search.far as far

        detail = far.run(self._prepared(tmp_path))["checks"]["inclusive"]

        for name in ("ad_p_value", "ks_p_value", "lrt_p_value"):
            assert 0.0 <= detail[name] <= 1.0

    def test_curve_round_trips(self, tmp_path):
        """
        A reloaded curve answers identically, extrapolation included.

        The tail is embedded rather than referenced: a curve stored without it would lose
        the ability to separate "1 in 2 yr" from "1 in 100 yr", and would raise at the
        point of use one stage later.
        """
        import sage.search.far as far
        from sage.search.far import FarCurve

        spec = self._prepared(tmp_path)
        report = far.run(spec)
        curve = FarCurve.load(report["curves"]["inclusive"])
        probe = np.array([2.0, 4.0, 8.0, 20.0])

        assert curve.tail is not None
        assert np.all(np.isfinite(curve.far_of(probe)))
        assert np.all(np.isfinite(curve.far_extrapolated_of(probe)))

    def test_missing_background_named(self, tmp_path):
        """A mode the background stage never built is named, not guessed at."""
        import sage.search.far as far
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        slides.run(spec)
        with pytest.raises(FileNotFoundError, match="inclusive background"):
            far.run(spec)

    def test_small_background_is_not_fitted(self, tmp_path):
        """
        Too few events to leave the requested exceedances means no tail, stated as such.

        The alternative -- fitting anyway on whatever is there -- produces a shape from a
        handful of points and an extrapolation that looks like a measurement.
        """
        import sage.search.far as far
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        slides.run(spec)
        _background(spec, size=50)
        detail = far.run(spec)["checks"]["inclusive"]

        assert "tail_shape" not in detail
        assert "not fitted" in detail["tail"]


def _to_far(tmp_path, **kwargs):
    """A campaign carried as far as an inclusive FAR curve, without scoring anything."""
    import sage.search.far as far
    import sage.search.grid as grid
    import sage.search.segments as segments
    import sage.search.slides as slides

    spec = _campaign(tmp_path, **kwargs)
    segments.run(spec)
    grid.run(spec)
    slides.run(spec)
    _background(spec)
    far.run(spec)
    return spec


class TestTrialsDriver:
    """Which analyses were competing, and over exactly which seconds."""

    def test_single_arm_model(self, tmp_path):
        """
        A campaign with no siblings registers one arm, whose analysed livetime is the
        lattice's -- not the coincident livetime, which is larger and would credit the
        arm with chances at moments it could not have triggered on.
        """
        import sage.search.grid as grid
        import sage.search.trials as trials

        spec = _to_far(tmp_path)
        report = trials.run(spec)
        lattice = grid.run(spec)

        assert report["n_arms"] == 1
        assert report["arms"][spec.arm]["livetime_s"] == lattice["analysed_livetime_s"]
        assert (
            report["arms"][spec.arm]["livetime_s"] < lattice["coincident_livetime_s"]
        )

    def test_coverage_follows_the_lattice(self, tmp_path):
        """
        A time inside the analysed lattice is covered; one past its end is not. Coverage
        is what the factor counts, so this is the arithmetic behind every factor.
        """
        import numpy as np

        import sage.search.trials as trials
        from sage.search.trials import TrialsModel

        spec = _to_far(tmp_path)
        model = TrialsModel.load(trials.run(spec)["model"])
        intervals = model.segments[spec.arm].intervals

        inside = float(intervals[0].mean())
        outside = float(intervals[-1][1]) + 1.0e5
        assert model.coverage_at(np.array([inside])) == [(spec.arm,)]
        assert model.coverage_at(np.array([outside])) == [()]

    def test_sibling_arm_counts(self, tmp_path):
        """
        Two arms over the same run give a candidate in time both analysed two chances,
        and one in time only one of them analysed a single chance. That difference is
        the reason the factor is per candidate rather than per campaign.
        """
        import numpy as np

        import sage.search.trials as trials
        from sage.search.trials import build_records, trials_factor

        from tests.search_fixtures import make_synthetic_release

        # One release, two arms -- which is what an HL and an HLV search of one run are.
        release = make_synthetic_release(
            tmp_path / "release", detectors=("H1", "L1", "V1"), chunk_s=64.0
        )
        first = _to_far(tmp_path / "hl", release=release)
        second = _to_far(
            tmp_path / "hlv",
            detectors=("H1", "L1", "V1"),
            release=release,
            tag="synth_hlv",
            out_dir=tmp_path / "hlv" / "campaign",
            # Triple coincidence is a shorter stretch than double, so the ladder has to
            # be shorter too -- which is the real HLV/HL relationship in miniature.
            slides=SlideSpec(
                n_slides=2,
                reference_detector="H1",
                min_separation_s=10.0,
                tau_max_s=40.0,
                seed=5,
            ),
        )
        assert first.arm != second.arm

        model = trials.build_model(first, [second])
        assert len(model.internal_arms()) == 2

        both = float(model.segments[second.arm].intervals[0].mean())
        assert len(model.coverage_at(np.array([both]))[0]) == 2
        assert trials_factor(
            build_records({"name": ["c"], "gps": [both]}, model)[0], model
        ) == 2

    def test_sibling_without_far_is_refused(self, tmp_path):
        """
        An arm counts as a chance noise had only if its analysis actually ran. Counting
        one that did not would inflate every candidate's factor on the strength of a
        competing search that does not exist.
        """
        import sage.search.trials as trials

        spec = _to_far(tmp_path / "hl")
        unrun = _campaign(
            tmp_path / "other",
            detectors=("H1", "L1", "V1"),
            tag="other",
            out_dir=tmp_path / "other" / "campaign",
        )
        with pytest.raises(FileNotFoundError, match="has not completed its own"):
            trials.build_model(spec, [unrun])

    def test_cross_run_arm_is_refused(self, tmp_path):
        """
        Arms compete only over the same data. A factor built across observing runs would
        penalise a candidate for chances taken on entirely different seconds.
        """
        import sage.search.trials as trials

        spec = _to_far(tmp_path / "hl")
        other = dataclasses.replace(
            _to_far(tmp_path / "o3b", tag="synthb", out_dir=tmp_path / "o3b" / "camp"),
            data=dataclasses.replace(
                _campaign(tmp_path / "o3b2").data, observing_run="O3b"
            ),
        )
        with pytest.raises(ValueError, match="arms compete only over the same data"):
            trials.build_model(spec, [other])

    def test_fingerprint_tracks_the_intervals(self, tmp_path):
        """
        The analysed intervals are the product: they decide each candidate's factor one
        candidate at a time. A fingerprint over the arm count alone could not see a
        lattice that moved.
        """
        import sage.search.trials as trials

        spec = _to_far(tmp_path)
        before = trials.run(spec)["fingerprint"]
        assert trials.run(spec)["fingerprint"] == before

        shifted = dataclasses.replace(
            spec, trials=dataclasses.replace(spec.trials, convention="none")
        )
        assert trials.run(shifted)["fingerprint"] != before


class TestStagedInjectionTable:
    """The drawn parameter set is kept in the campaign that scored it."""

    def _stage(self, spec, columns, attrs, table, calls):
        from sage.search.injection.campaign import _staged_table

        def build():
            calls.append(1)
            return table

        return _staged_table(spec, 0, columns, attrs, build)

    def _fixture(self, tmp_path):
        import dataclasses

        import numpy as np

        spec = _campaign(tmp_path)
        # The staged table is reloaded onto the engine's device, which is a GPU in a real
        # campaign and absent on the login node this runs on.
        spec = dataclasses.replace(
            spec, engine=dataclasses.replace(spec.engine, device="cpu")
        )
        columns = ["mass1", "mass2", "tc"]
        attrs = {"n_draw": 100, "draw_seed": 7, "hyperposterior": "abc123"}
        table = np.array([[30.0, 20.0, 6.0], [40.0, 35.0, 6.5]], dtype=np.float64)
        return spec, columns, attrs, table

    def test_written_under_the_campaign(self, tmp_path):
        """
        Not a scratch file. The parameters behind ``p(x | signal)`` have to be readable
        after the job that drew them has exited, or the signal density cannot be checked
        against what was recovered.
        """
        spec, columns, attrs, table = self._fixture(tmp_path)
        self._stage(spec, columns, attrs, table, [])

        staged = spec.path("injections", "injection_table_00.h5")
        assert staged.is_file()
        assert staged.is_relative_to(spec.out_dir)

    def test_reused_when_provenance_matches(self, tmp_path):
        """Drawing is seeded, so a re-run scores the set it scored before."""
        import numpy as np

        spec, columns, attrs, table = self._fixture(tmp_path)
        calls = []
        self._stage(spec, columns, attrs, table, calls)
        again = self._stage(spec, columns, attrs, table, calls)

        assert len(calls) == 1
        assert np.allclose(np.asarray(again), table)

    def test_redrawn_when_population_differs(self, tmp_path):
        """
        A different hyperposterior is a different population under the same seed. Reusing
        the stored table would score one population under another's name.
        """
        spec, columns, attrs, table = self._fixture(tmp_path)
        calls = []
        self._stage(spec, columns, attrs, table, calls)
        self._stage(spec, columns, {**attrs, "hyperposterior": "def456"}, table, calls)

        assert len(calls) == 2

    def test_redrawn_when_columns_differ(self, tmp_path):
        """
        The table is positional, so a sampler whose column order changed would have its
        masses read as spins with no error anywhere.
        """
        spec, columns, attrs, table = self._fixture(tmp_path)
        calls = []
        self._stage(spec, columns, attrs, table, calls)
        self._stage(spec, ["tc", "mass1", "mass2"], attrs, table, calls)

        assert len(calls) == 2

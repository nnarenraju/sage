#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_stages.py
Description   : The stage graph: ordering, lookup, and agreement with the import graph.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The package has two orderings that must agree. The import graph decides when a module can
be *written*; the stage graph decides when it can be *run against numbers that mean
something*. They are maintained by hand in different files, so they drift, and the drift
is silent: a stage declared to run early whose module imports a late one still imports
fine and still passes its own unit tests.

The last class here checks them against each other.

Runs anywhere; needs no data, no GPU and no network.
"""

import ast
import pathlib

import pytest

from sage.search import stages as S


class TestRegistry:
    """The declared graph is well formed."""

    def test_stage_names_are_unique(self):
        names = [s.name for s in S.STAGES]
        assert len(names) == len(set(names))

    def test_every_dependency_names_a_real_stage(self):
        known = {s.name for s in S.STAGES}
        for stage in S.STAGES:
            for dep in stage.depends_on:
                assert dep in known, f"{stage.name} depends on unknown stage {dep!r}"

    def test_lookup_by_name(self):
        assert S.stage_by_name("far").name == "far"

    def test_unknown_name_suggests_alternatives(self):
        """A typo gets a message naming close matches, not a bare KeyError."""
        with pytest.raises(KeyError, match="far"):
            S.stage_by_name("fars")

    def test_tracks_partition_the_registry(self):
        core = S.track("core")
        followup = S.track("followup")
        assert {s.name for s in core}.isdisjoint({s.name for s in followup})
        assert len(core) + len(followup) == len(S.STAGES)

    def test_unknown_track_is_refused(self):
        with pytest.raises(ValueError):
            S.track("middle")

    def test_every_stage_has_a_module(self):
        """The dispatch table covers the graph, so no stage is unrunnable."""
        for stage in S.STAGES:
            assert stage.name in S.STAGE_MODULES, f"{stage.name} has no module"

    def test_stage_modules_are_importable_paths(self):
        """Each mapped module exists on disk."""
        import importlib.util

        for name, dotted in S.STAGE_MODULES.items():
            assert importlib.util.find_spec(dotted) is not None, f"{name} -> {dotted}"


class TestOrdering:
    """Resolution puts every stage after the things it consumes."""

    def test_order_is_topological(self):
        ordered = S.resolve_order([s.name for s in S.STAGES])
        position = {s.name: i for i, s in enumerate(ordered)}
        for stage in ordered:
            for dep in stage.depends_on:
                assert position[dep] < position[stage.name], (
                    f"{stage.name} is ordered before its dependency {dep}"
                )

    def test_dependencies_are_pulled_in(self):
        """Asking for a late stage brings everything it needs."""
        names = [s.name for s in S.resolve_order(["far"])]
        assert {"segments", "grid", "slides", "background", "far"} <= set(names)

    def test_dependencies_can_be_suppressed(self):
        """With dependencies off, only what was asked for comes back."""
        got = S.resolve_order(["far", "segments"], include_dependencies=False)
        assert {s.name for s in got} == {"far", "segments"}

    def test_suppressed_deps_ordered(self):
        got = [s.name for s in S.resolve_order(["far", "segments"], include_dependencies=False)]
        assert got.index("segments") < got.index("far")

    def test_result_has_no_duplicates(self):
        names = [s.name for s in S.resolve_order(["far", "background", "far"])]
        assert len(names) == len(set(names))

    def test_unknown_target_is_refused(self):
        with pytest.raises(KeyError):
            S.resolve_order(["nonexistent"])

    def test_empty_target_list(self):
        assert S.resolve_order([]) == []

    def test_cycle_is_detected(self, monkeypatch):
        """
        A dependency cycle raises rather than looping or truncating.

        Cannot happen in the declared graph, but the graph is edited by hand and a cycle
        would otherwise surface as a stage silently never running.
        """
        a = S.Stage("a", ("b",), "")
        b = S.Stage("b", ("a",), "")
        monkeypatch.setattr(S, "STAGES", (a, b))
        with pytest.raises(ValueError, match="cycle"):
            S.resolve_order(["a"])


class TestTrialsPlacement:
    """The trials stage sits where the candidate table can use it."""

    def test_trials_follows_far(self):
        order = [s.name for s in S.resolve_order(["candidates"])]
        assert order.index("far") < order.index("trials")

    def test_candidates_deps(self):
        deps = set(S.stage_by_name("candidates").depends_on)
        assert {"pastro", "trials"} <= deps


class TestGraphsAgree:
    """The import graph and the stage graph must not contradict each other."""

    def test_no_stage_module_imports_a_later_stage(self):
        """
        A stage's module may not import a module belonging to a later stage.

        This is the check that catches the two graphs drifting. A stage placed early
        whose module imports a late one still imports cleanly and still passes its own
        tests; the contradiction only shows up as a stage that cannot actually run when
        it is scheduled to.
        """
        import sage.search

        root = pathlib.Path(sage.search.__file__).parent
        ordered = S.resolve_order([s.name for s in S.STAGES])
        position = {s.name: i for i, s in enumerate(ordered)}
        # A module may own more than one stage -- sage.search.candidates owns both
        # `candidates` and the follow-up `retier` -- so this is a mapping to *every*
        # stage a module owns. Inverting STAGE_MODULES into a plain dict keeps only the
        # last, which made an import of an early stage look like an import of a late one
        # and reported a violation that was not there.
        module_to_stages = {}
        for stage_name_, dotted_ in S.STAGE_MODULES.items():
            module_to_stages.setdefault(dotted_, []).append(stage_name_)

        problems = []
        for stage_name, dotted in S.STAGE_MODULES.items():
            relative = dotted.removeprefix("sage.search.").replace(".", "/")
            path = root / f"{relative}.py"
            if not path.is_file():
                path = root / relative / "__init__.py"
            if not path.is_file():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported = None
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("sage.search"):
                            imported = alias.name
                if imported is None or imported not in module_to_stages:
                    continue
                others = [
                    name for name in module_to_stages[imported] if name != stage_name
                ]
                if not others:
                    continue
                # Legitimate if the module owns *any* stage scheduled earlier: the
                # import is then of something already built. Only when every stage it
                # owns comes later is the import genuinely out of order.
                if all(position[other] > position[stage_name] for other in others):
                    problems.append(
                        f"{stage_name} ({dotted}) imports {others} ({imported}), "
                        f"all of which are scheduled later"
                    )
        assert not problems, "stage graph contradicts the import graph:\n" + "\n".join(
            problems
        )


def _campaign_spec(tmp_path):
    """A minimal but valid spec rooted at a temporary campaign directory."""
    from sage.search.spec import DataSpec, EngineSpec, GeometrySpec, SearchSpec

    return SearchSpec(
        tag="o3a-HL",
        config_module="tests.test_search_stages",
        out_dir=tmp_path / "campaign",
        data=DataSpec(
            observing_run="O3a",
            detectors=("H1", "L1"),
            release_dir=tmp_path / "release",
            fiducial_dir=tmp_path / "fiducial",
        ),
        engine=EngineSpec(checkpoint=tmp_path / "best.pt"),
        geometry=GeometrySpec(tc_source="explicit", tc_lower_s=5.0, tc_upper_s=7.0),
    )


def _mark_done(spec, stage, spec_hash=None, fingerprint=None):
    """Record a stage as complete, as run_stage would."""
    from sage.search.manifest import RunManifest

    path = pathlib.Path(S.manifest_path(spec))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "spec_hash": spec_hash or spec.hash()}
    if fingerprint is not None:
        payload["fingerprint"] = fingerprint
    RunManifest(path=path).record_stage(stage, payload)


class TestCompletion:
    """What counts as a stage having run."""

    def test_fresh_campaign_has_nothing_complete(self, tmp_path):
        """A missing manifest is the state a campaign starts in, not an error."""
        spec = _campaign_spec(tmp_path)
        assert not S.is_complete(spec, "segments")
        assert [stage.name for stage in S.pending(spec)] == [
            stage.name for stage in S.track("core")
        ]

    def test_recorded_stage_is_complete(self, tmp_path):
        """A stage recorded under this configuration does not run again."""
        spec = _campaign_spec(tmp_path)
        _mark_done(spec, "segments")

        assert S.is_complete(spec, "segments")
        assert "segments" not in [stage.name for stage in S.pending(spec)]

    def test_other_configuration_does_not_count(self, tmp_path):
        """
        A product built under a different configuration is not a completed stage.

        Reusing one is silent and produces results that are wrong in a way nothing
        downstream can see, which is why the hash is checked rather than the file's
        existence.
        """
        spec = _campaign_spec(tmp_path)
        _mark_done(spec, "segments", spec_hash="a-different-configuration")

        assert not S.is_complete(spec, "segments")

    def test_unknown_stage_refused(self, tmp_path):
        """A misspelled stage must not silently report incomplete forever."""
        spec = _campaign_spec(tmp_path)
        with pytest.raises(KeyError):
            S.is_complete(spec, "backgruond")


class TestPending:
    """Which stages a resumed campaign still has to run."""

    def test_staleness_propagates_downstream(self, tmp_path):
        """
        A completed stage whose input is pending is itself pending.

        Its manifest entry cannot know that its input was rebuilt, so carrying the flag
        downstream is what makes a re-run of an early stage propagate instead of leaving a
        stale tail behind it.
        """
        spec = _campaign_spec(tmp_path)
        # `far` is recorded but `background`, which it depends on, is not.
        _mark_done(spec, "far")
        names = [stage.name for stage in S.pending(spec)]

        assert "background" in names
        assert "far" in names

    def test_completed_prefix_is_dropped(self, tmp_path):
        """Everything already done under this configuration drops out, in order."""
        spec = _campaign_spec(tmp_path)
        for stage in ("segments", "grid"):
            _mark_done(spec, stage)
        names = [stage.name for stage in S.pending(spec)]

        assert "segments" not in names and "grid" not in names
        assert names[0] in ("zerolag", "slides")

    def test_skip_does_not_make_dependants_pending(self, tmp_path):
        """
        Skipping a stage states its products are not wanted, so it is not staleness.

        A campaign that deliberately omits an arm must not have that omission cascade into
        re-running everything behind it.
        """
        spec = _campaign_spec(tmp_path)
        for stage in ("segments", "grid", "zerolag", "slides", "background", "far"):
            _mark_done(spec, stage)
        _mark_done(spec, "trials")
        names = [stage.name for stage in S.pending(spec, skip=("injections",))]

        assert "injections" not in names
        assert "trials" not in names

    def test_skipped_dependants_are_not_offered(self, tmp_path):
        """
        Nothing downstream of a skipped stage is returned, because it cannot run.

        ``sensitivity`` needs ``injections``; with the latter skipped its product was
        never built, so ``run_stage`` refuses it. Returning it would hand the caller a
        plan whose next step raises.
        """
        spec = _campaign_spec(tmp_path)
        for stage in ("segments", "grid", "zerolag", "slides", "background", "far"):
            _mark_done(spec, stage)
        names = [stage.name for stage in S.pending(spec, skip=("injections",))]

        assert "sensitivity" not in names
        assert "pastro" not in names
        assert "candidates" not in names
        assert "trials" in names

    def test_everything_returned_can_be_run(self, tmp_path):
        """
        The plan is executable in order: each stage's dependencies precede it or are done.

        This is the contract that makes `pending` a plan rather than a list of names --
        an entry `run_stage` would refuse is not work the caller can do.
        """
        spec = _campaign_spec(tmp_path)
        _mark_done(spec, "segments")
        plan = S.pending(spec, skip=("injections",))
        seen = {"segments"}
        for stage in plan:
            for dep in stage.depends_on:
                assert dep in seen or S.is_complete(spec, dep)
            seen.add(stage.name)

    def test_followup_can_exclude_the_core_track(self, tmp_path):
        """
        ``include_dependencies=False`` answers what is left of this track alone.

        The follow-up track depends on the core one, so by default completing it means
        running core stages and they are included. Asking for the track alone is a
        different question and gets a different answer.
        """
        spec = _campaign_spec(tmp_path)
        withdeps = [s.name for s in S.pending(spec, "followup")]
        alone = [s.name for s in S.pending(spec, "followup", include_dependencies=False)]

        assert "candidates" in withdeps
        assert "candidates" not in alone
        assert set(alone) <= {s.name for s in S.track("followup")}
        assert "dataquality" in alone

    def test_unknown_skip_refused(self, tmp_path):
        """A typo in skip would silently run a stage the caller meant to omit."""
        spec = _campaign_spec(tmp_path)
        with pytest.raises(ValueError, match="unknown stages"):
            S.pending(spec, skip=("injectons",))


class TestDescendants:
    """The dependency chain downstream of a stage."""

    def test_transitive_not_immediate(self):
        """
        The chain follows through intermediate stages, not just direct edges.

        ``release`` does not name ``far`` in its dependencies; it needs ``figures`` needs
        ``figdata`` needs ``store`` needs ``catalogue`` needs ``candidates`` needs
        ``pastro`` needs ``far``. Following only direct edges would leave the far end of
        that chain quietly stale.
        """
        names = [stage.name for stage in S.descendants("far")]

        assert "pastro" in names
        assert "release" in names
        assert "far" not in names
        assert "segments" not in names
        assert "grid" not in names

    def test_chain_crosses_tracks(self):
        """The follow-up track depends on the core one, so the chain must cross."""
        names = {stage.name for stage in S.descendants("candidates")}

        assert {"dataquality", "qscans", "retier", "event_pages"} <= names

    def test_terminal_stage_has_none(self):
        """Nothing is built from the release, so nothing is invalidated by rebuilding it."""
        assert S.descendants("release") == []

    def test_first_stage_reaches_everything(self):
        """Rebuilding the segments invalidates the whole campaign."""
        names = {stage.name for stage in S.descendants("segments")}

        assert names == {stage.name for stage in S.STAGES} - {"segments"}

    def test_returned_in_dependency_order(self):
        """The chain is a plan, so a stage never precedes something it depends on."""
        chain = S.descendants("grid")
        seen = set()
        for stage in chain:
            for dep in stage.depends_on:
                assert dep in seen or dep not in {s.name for s in chain}
            seen.add(stage.name)

    def test_unknown_stage_refused(self):
        """A typo would report an empty chain and invalidate nothing."""
        with pytest.raises(KeyError):
            S.descendants("grd")


class TestRunStage:
    """Dispatch, ordering and recording."""

    def test_dependencies_are_required(self, tmp_path):
        """
        A stage run out of order is refused rather than run on absent inputs.

        It would otherwise produce a product from inputs that do not exist, carrying a
        provenance block saying otherwise.
        """
        spec = _campaign_spec(tmp_path)
        with pytest.raises(ValueError, match="depends on"):
            S.run_stage(spec, "far")

    def test_missing_driver_names_its_module(self, tmp_path):
        """
        An unbuilt stage says which module owes a run() rather than failing obscurely.

        The stage is chosen at run time rather than named, so building one does not break
        a test about the ones still to come -- and once every stage has a driver the test
        skips rather than needing deletion, which is the honest end state for it.
        """
        import importlib
        import inspect

        unbuilt = None
        for stage in S.track("core"):
            module = importlib.import_module(S.STAGE_MODULES[stage.name])
            driver = getattr(module, "run", None)
            if driver is None or "raise NotImplementedError" in inspect.getsource(driver):
                unbuilt = stage.name
                break
        if unbuilt is None:
            pytest.skip("every core stage has a driver")

        spec = _campaign_spec(tmp_path)
        for stage in S.track("core"):
            if stage.name != unbuilt:
                _mark_done(spec, stage.name)
        with pytest.raises(NotImplementedError, match=S.STAGE_MODULES[unbuilt]):
            S.run_stage(spec, unbuilt)

    def test_success_records_the_spec_hash(self, tmp_path, monkeypatch):
        """
        A completed stage is recorded with the configuration it ran under.

        That record is what `is_complete` reads, so this closes the loop: running a stage
        makes it complete, and only for this configuration.
        """
        import sage.search.segments as segments_module

        spec = _campaign_spec(tmp_path)
        monkeypatch.setattr(
            segments_module, "run", lambda s, **kw: {"n_segments": 7}, raising=False
        )
        report = S.run_stage(spec, "segments")

        assert report == {"n_segments": 7}
        assert S.is_complete(spec, "segments")
        assert "segments" not in [stage.name for stage in S.pending(spec)]

    def test_rerun_invalidates_the_chain(self, tmp_path, monkeypatch):
        """
        Re-running a stage from the middle marks everything downstream for re-running.

        This is the case the spec hash cannot see: fix a bug in a stage's module, leave the
        configuration untouched, re-run it, and the hash is unchanged -- so without this
        every stage built on the old product still reports complete.
        """
        import sage.search.grid as grid_module

        spec = _campaign_spec(tmp_path)
        for stage in S.track("core"):
            _mark_done(spec, stage.name)
        assert S.pending(spec) == []

        monkeypatch.setattr(
            grid_module, "run", lambda s, **kw: {"n_windows": 11}, raising=False
        )
        S.run_stage(spec, "grid")
        names = [stage.name for stage in S.pending(spec)]

        assert S.is_complete(spec, "grid")
        assert S.is_complete(spec, "segments")
        assert names == [
            stage.name
            for stage in S.descendants("grid")
            if stage.track == "core"
        ]
        assert "zerolag" in names
        assert "far" in names

    def test_rerun_leaves_upstream_alone(self, tmp_path, monkeypatch):
        """
        Only the chain downstream is invalidated; inputs are untouched.

        A stage's dependencies produced what it just consumed. Invalidating them would
        turn every re-run into a full campaign rebuild from the first stage.
        """
        import sage.search.far as far_module

        spec = _campaign_spec(tmp_path)
        for stage in S.track("core"):
            _mark_done(spec, stage.name)
        monkeypatch.setattr(
            far_module, "run", lambda s, **kw: {"n_bins": 3}, raising=False
        )
        S.run_stage(spec, "far")

        for upstream in ("segments", "grid", "zerolag", "slides", "background"):
            assert S.is_complete(spec, upstream)
        assert not S.is_complete(spec, "pastro")

    def test_unchanged_product_does_not_invalidate(self, tmp_path, monkeypatch):
        """
        A re-run that produces the same product costs the campaign nothing.

        The chain exists because downstream reads this stage's output; it needs rebuilding
        only if that output moved. The driver reports a fingerprint, and a match is a
        measurement that it did not -- so the chain stays complete without anyone having
        to promise it would.
        """
        import sage.search.grid as grid_module

        spec = _campaign_spec(tmp_path)
        monkeypatch.setattr(
            grid_module,
            "run",
            lambda s, **kw: {"n_windows": 11, "fingerprint": "abc123"},
            raising=False,
        )
        _mark_done(spec, "segments")
        S.run_stage(spec, "grid")
        for stage in S.track("core"):
            _mark_done(spec, stage.name, fingerprint="abc123" if stage.name == "grid" else None)
        assert S.pending(spec) == []

        S.run_stage(spec, "grid")

        assert S.pending(spec) == []
        assert S.recorded_fingerprint(spec, "grid") == "abc123"

    def test_changed_product_invalidates(self, tmp_path, monkeypatch):
        """A different fingerprint is the output having moved, so the chain goes."""
        import sage.search.grid as grid_module

        spec = _campaign_spec(tmp_path)
        _mark_done(spec, "segments")
        monkeypatch.setattr(
            grid_module,
            "run",
            lambda s, **kw: {"fingerprint": "before"},
            raising=False,
        )
        S.run_stage(spec, "grid")
        for stage in S.track("core"):
            if stage.name not in ("segments", "grid"):
                _mark_done(spec, stage.name)
        assert S.pending(spec) == []

        monkeypatch.setattr(
            grid_module, "run", lambda s, **kw: {"fingerprint": "after"}, raising=False
        )
        S.run_stage(spec, "grid")
        names = [stage.name for stage in S.pending(spec)]

        assert "zerolag" in names
        assert "far" in names

    def test_no_fingerprint_cascades(self, tmp_path, monkeypatch):
        """
        A driver that reports no fingerprint invalidates the chain.

        The absence of a measurement is not a measurement of no change. It is also the
        correct default for a stage whose output is not reproducible -- anything seeded
        from the clock, or accumulated across a re-submitted array.
        """
        import sage.search.grid as grid_module

        spec = _campaign_spec(tmp_path)
        for stage in S.track("core"):
            _mark_done(spec, stage.name)
        monkeypatch.setattr(
            grid_module, "run", lambda s, **kw: {"n_windows": 11}, raising=False
        )
        S.run_stage(spec, "grid")

        assert "zerolag" in [stage.name for stage in S.pending(spec)]
        assert S.recorded_fingerprint(spec, "grid") is None

    def test_cascade_can_be_forced_either_way(self, tmp_path, monkeypatch):
        """
        The booleans override the measurement, for a fingerprint that cannot be trusted.

        ``False`` is a promise rather than a measurement: if it is wrong, every later
        product keeps a provenance block saying it was built under this configuration,
        which it was, from an input that has since been replaced, which it does not record.
        """
        import sage.search.grid as grid_module

        spec = _campaign_spec(tmp_path)
        for stage in S.track("core"):
            _mark_done(spec, stage.name)
        monkeypatch.setattr(
            grid_module, "run", lambda s, **kw: {"n_windows": 11}, raising=False
        )
        S.run_stage(spec, "grid", cascade=False)
        assert S.pending(spec) == []

        S.run_stage(spec, "grid", cascade=True)
        assert "zerolag" in [stage.name for stage in S.pending(spec)]

    def test_bad_cascade_refused(self, tmp_path, monkeypatch):
        """A misspelled cascade mode is not silently treated as one of the three."""
        import sage.search.grid as grid_module

        spec = _campaign_spec(tmp_path)
        _mark_done(spec, "segments")
        monkeypatch.setattr(
            grid_module, "run", lambda s, **kw: {"n_windows": 11}, raising=False
        )
        with pytest.raises(ValueError, match="cascade"):
            S.run_stage(spec, "grid", cascade="always")

    def test_crash_leaves_the_stage_incomplete(self, tmp_path, monkeypatch):
        """
        A re-run that dies part way through is not still recorded as complete.

        The driver has already overwritten some of its products by the time it fails, so
        the previous entry describes a product that no longer exists. Dropping the entry
        before dispatch makes the crash window read as "not run", which is what it is.
        """
        import sage.search.segments as segments_module

        spec = _campaign_spec(tmp_path)
        monkeypatch.setattr(
            segments_module, "run", lambda s, **kw: {"n_segments": 7}, raising=False
        )
        S.run_stage(spec, "segments")
        assert S.is_complete(spec, "segments")

        def explode(s, **kw):
            raise RuntimeError("died half way")

        monkeypatch.setattr(segments_module, "run", explode, raising=False)
        with pytest.raises(RuntimeError, match="died half way"):
            S.run_stage(spec, "segments")

        assert not S.is_complete(spec, "segments")
        assert "segments" in [stage.name for stage in S.pending(spec)]

    def test_driver_keys_are_not_overwritten(self, tmp_path, monkeypatch):
        """
        A driver reporting its own `stage` or `spec_hash` keeps them.

        They are the stage's own findings; silently replacing them with the driver's
        identity would discard a measurement and leave nothing saying so.
        """
        import sage.search.segments as segments_module
        from sage.search.manifest import RunManifest

        spec = _campaign_spec(tmp_path)
        monkeypatch.setattr(
            segments_module,
            "run",
            lambda s, **kw: {"stage": "measured-something", "spec_hash": "mine"},
            raising=False,
        )
        S.run_stage(spec, "segments")
        entry = RunManifest(path=pathlib.Path(S.manifest_path(spec))).summary()
        recorded = entry["stages"]["segments"]

        assert recorded["stage"] == "segments"
        assert recorded["spec_hash"] == str(spec.hash())
        assert recorded["report"]["stage"] == "measured-something"
        assert recorded["report"]["spec_hash"] == "mine"

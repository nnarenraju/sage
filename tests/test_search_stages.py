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

    def test_suppressed_dependencies_are_still_ordered(self):
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

    def test_candidates_depends_on_trials_and_pastro(self):
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
        module_to_stage = {v: k for k, v in S.STAGE_MODULES.items()}

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
                if imported is None or imported not in module_to_stage:
                    continue
                other = module_to_stage[imported]
                if other == stage_name:
                    continue
                if position[other] > position[stage_name]:
                    problems.append(
                        f"{stage_name} ({dotted}) imports {other} ({imported}), "
                        f"which is scheduled later"
                    )
        assert not problems, "stage graph contradicts the import graph:\n" + "\n".join(
            problems
        )

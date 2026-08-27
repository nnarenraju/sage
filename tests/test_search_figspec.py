#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_figspec.py
Description   : The figure declarations are well formed and reachable from the stages.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The declarations are a backward contract on the analysis stages: a figure that needs an
array obliges the stage named as its source to persist that array. So the thing worth
checking here is that every source is a real stage and every builder is a real module --
a declaration pointing at neither is a contract against nothing.

Runs anywhere; needs no data, no GPU and no network.
"""

import importlib.util

import pytest

from sage.search import stages as S
from sage.search.figdata import spec as F


class TestDeclarations:
    """The registry is populated and internally consistent."""

    def test_registry_is_populated(self):
        assert len(F.FIGURES) >= 20

    def test_keys_match_their_declarations(self):
        for key, decl in F.FIGURES.items():
            assert decl.key == key

    def test_every_figure_requires_something(self):
        for key, decl in F.FIGURES.items():
            assert decl.requires, key

    def test_requirements_unique(self):
        for key, decl in F.FIGURES.items():
            assert len(decl.requires) == len(set(decl.requires)), key

    def test_every_figure_has_a_title(self):
        for key, decl in F.FIGURES.items():
            assert decl.title.strip(), key

    def test_duplicate_declaration_is_refused(self):
        decl = next(iter(F.FIGURES.values()))
        with pytest.raises(ValueError, match="already declared"):
            F.declare(decl)

    def test_no_requirements_refused(self):
        with pytest.raises(ValueError, match="required arrays"):
            F.declare(
                F.FigureDecl(
                    key="empty", title="t", builder="build_meta", requires=(), sources=("far",)
                )
            )

    def test_no_source_refused(self):
        with pytest.raises(ValueError, match="source stage"):
            F.declare(
                F.FigureDecl(
                    key="sourceless", title="t", builder="build_meta",
                    requires=("x",), sources=(),
                )
            )


class TestReachability:
    """Every declaration points at things that exist."""

    def test_every_source_is_a_real_stage(self):
        """
        A source naming no stage is a contract against nothing.

        This is what ties the figure set to the pipeline: the stage named here is the one
        obliged to write the arrays listed in ``requires``.
        """
        known = {s.name for s in S.STAGES}
        for key, decl in F.FIGURES.items():
            for source in decl.sources:
                assert source in known, f"{key} names unknown stage {source!r}"

    def test_every_builder_module_exists(self):
        for key, decl in F.FIGURES.items():
            dotted = f"sage.search.figdata.{decl.builder}"
            assert importlib.util.find_spec(dotted) is not None, f"{key} -> {dotted}"

    def test_required_stages_resolves(self):
        needed = F.required_stages(["cumulative_vs_ifar"])
        assert "far" in needed and "candidates" in needed

    def test_required_stages_deduplicates(self):
        keys = ["cumulative_vs_ifar", "far_versus_statistic"]
        needed = F.required_stages(keys)
        assert len(needed) == len(set(needed))

    def test_unknown_figure_suggests_alternatives(self):
        with pytest.raises(KeyError, match="mass_plane"):
            F.resolve("mass_plan")


class TestGrouping:
    """The registry can be sliced the ways the pipeline needs."""

    def test_by_builder(self):
        assert F.by_builder("build_significance")
        assert all(d.builder == "build_significance" for d in F.by_builder("build_significance"))

    def test_by_stage_finds_consumers(self):
        """A stage can ask what it is obliged to write."""
        assert {d.key for d in F.by_stage("far")}

    def test_per_event_figures_are_separated(self):
        per_event = F.per_event_figures()
        assert per_event
        assert all(d.per_event for d in per_event)
        campaign = [d for d in F.FIGURES.values() if not d.per_event]
        assert len(campaign) > len(per_event)


class TestCoverage:
    """The set covers the analysis it is supposed to describe."""

    @pytest.mark.parametrize(
        "stage", ["far", "background", "sensitivity", "pastro", "trials", "catalogue"]
    )
    def test_each_major_stage_has_a_figure(self, stage):
        """Every stage producing a headline number is shown somewhere."""
        assert F.by_stage(stage), f"no figure draws on {stage}"

    def test_both_significance_views_are_declared(self):
        """The trials-corrected and uncorrected views are both published."""
        decl = F.resolve("trials_comparison")
        assert "ifar_yr" in decl.requires
        assert "ifar_trials_yr" in decl.requires

    def test_livetime_decomposition_is_declared(self):
        """The coverage decomposition is shown, since every rate divides by it."""
        decl = F.resolve("livetime_and_duty_cycle")
        assert "analysed_s" in decl.requires
        assert "lost_boundary_s" in decl.requires


class TestProvenance:
    """Every figure names where it comes from, or says it is not being built."""

    def test_every_figure_is_backed_or_deferred(self):
        """
        The search follows sgwc-1's procedure and takes from PyCBC only what has been
        agreed case by case. A figure with neither origin is one nobody has justified,
        and the ruling of 2026-08-20 is that those wait rather than being built.

        This is the rule made executable: adding a figure means naming its origin, or
        saying explicitly that it is deferred. Neither is a default.
        """
        unbacked = sorted(
            key
            for key, decl in F.FIGURES.items()
            if not decl.origin and not decl.deferred
        )
        assert not unbacked, (
            "these figures have no sgwc-1 or PyCBC counterpart and are not marked "
            f"deferred: {unbacked}"
        )

    def test_origins_name_a_known_reference(self):
        """
        An origin says which reference and where in it, so the claim is checkable. A
        bare "sgwc-1" would be an assertion; "sgwc-1: pastro.ipynb, ..." is a pointer.
        """
        for key, decl in F.FIGURES.items():
            if not decl.origin:
                continue
            assert decl.origin.startswith(("sgwc-1:", "pycbc:")), (
                f"{key} declares origin {decl.origin!r}, which names no reference"
            )
            assert len(decl.origin.split(":", 1)[1].strip()) > 3, (
                f"{key} names a reference but not where in it: {decl.origin!r}"
            )

    def test_deferred_figures_give_a_reason(self):
        """A deferred figure records why, so the decision survives the session."""
        for key, decl in F.FIGURES.items():
            if decl.deferred:
                assert len(decl.deferred) > 20, (
                    f"{key} is deferred without a usable reason: {decl.deferred!r}"
                )


def _is_built(module, name: str) -> bool:
    """
    Whether a builder function is implemented, not merely present.

    ``hasattr`` is true for a stub, so a coverage check written on it reports every
    declared figure as built the moment its placeholder exists -- which is the opposite
    of what the check is for.
    """
    import inspect

    function = getattr(module, name, None)
    if function is None:
        return False
    try:
        source = inspect.getsource(function)
    except OSError:
        return True
    return "raise NotImplementedError" not in source


class TestBuilderCoverage:
    """The declarations and the builder modules must not drift apart."""

    #: Figures we intend to build whose builder has not been written yet. A list rather
    #: than a tolerance: each entry is a specific piece of work, and the test fails when
    #: one lands so the list is trimmed instead of quietly outliving its reason.
    NOT_YET_BUILT = {
        # All four are sensitivity figures, and sensitivity needs found/missed against an
        # analysed timeline. sgwc-1's injection campaign has no timeline, so the
        # lattice-scheduled campaign that would give one is deferred -- and these wait on
        # it rather than on anything here.
        "injection_recovery",
        "sensitive_distance",
        "vt_versus_far",
        "range_over_time",
    }

    def test_declared_builder_functions_exist(self):
        """
        A figure we intend to build must have somewhere to build it.

        Dispatch is by ``builder_function``, which defaults to the figure key. Most
        builders name the function after the key; the ones that do not declare the name
        explicitly, because inferring it from a module's contents would pick a plausible
        neighbour whenever a name drifted.
        """
        import importlib

        missing = []
        for key, decl in F.FIGURES.items():
            if decl.deferred or key in self.NOT_YET_BUILT:
                continue
            try:
                module = importlib.import_module(f"sage.search.figdata.{decl.builder}")
            except ModuleNotFoundError:
                missing.append(f"{key} -> module {decl.builder}")
                continue
            if not _is_built(module, decl.builder_function):
                missing.append(f"{key} -> {decl.builder}.{decl.builder_function}")
        assert not missing, (
            "figures marked for building whose builder function does not exist: "
            f"{sorted(missing)}"
        )

    def test_the_unbuilt_list_shrinks(self):
        """
        Every entry of ``NOT_YET_BUILT`` must still be unbuilt. When one lands, this
        fails and the entry is removed -- so the list cannot outlive the work it records
        and quietly exempt a figure that is now checkable.
        """
        import importlib

        landed = []
        for key in sorted(self.NOT_YET_BUILT):
            decl = F.FIGURES[key]
            try:
                module = importlib.import_module(f"sage.search.figdata.{decl.builder}")
            except ModuleNotFoundError:
                continue
            if _is_built(module, decl.builder_function):
                landed.append(key)
        assert not landed, (
            f"these are built now and should leave NOT_YET_BUILT: {landed}"
        )

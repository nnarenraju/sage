#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_figdata.py
Description   : The figure-data layer: dispatch, products and the index.

Created on 2026-08-22

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Figure data is written by the analysis and read by a plotting function that computes
nothing, so a figure can be redrawn without rerunning the analysis and a plot cannot
disagree with what it depicts. What is checked here is the layer that keeps that true:
that a builder's output satisfies the contract the plotting function reads against, and
that a figure which was never built is reported as such rather than passed on.
"""

import dataclasses
import json

import numpy as np
import pytest

from sage.search import figdata
from sage.search.figdata.product import FigData
from sage.search.figdata.spec import FIGURES, FigureDecl


@pytest.fixture
def spec(tmp_path):
    """A campaign whose directory exists and holds nothing."""
    from sage.search.spec import DataSpec, SearchSpec

    return SearchSpec(
        tag="figs",
        config_module="figs",
        out_dir=tmp_path / "campaign",
        data=DataSpec(observing_run="O3a", detectors=("H1", "L1")),
    )


@pytest.fixture
def declared(monkeypatch):
    """A small figure set standing in for the declared one."""
    monkeypatch.setitem(
        FIGURES,
        "toy_figure",
        FigureDecl(
            key="toy_figure",
            title="Toy",
            builder="build_toy",
            requires=("x", "y"),
            sources=("far",),
            origin="sgwc-1",
        ),
    )
    return "toy_figure"


def _install_builder(monkeypatch, product, name="build_toy", function="toy_figure"):
    """Register a fake builder module the dispatch can import."""
    import sys
    import types

    module = types.ModuleType(f"sage.search.figdata.{name}")
    setattr(module, function, lambda spec: product)
    monkeypatch.setitem(sys.modules, f"sage.search.figdata.{name}", module)
    return module


class TestDispatch:
    """What gets built, and what does not."""

    def test_builds_and_saves(self, spec, declared, monkeypatch):
        """A declared figure's builder runs and its product lands under the campaign."""
        product = FigData(
            figure="toy_figure",
            arrays={"x": np.arange(4.0), "y": np.ones(4)},
        )
        _install_builder(monkeypatch, product)

        built = figdata.build(spec, figures=["toy_figure"])

        assert set(built) == {"toy_figure"}
        assert built["toy_figure"].is_file()
        assert built["toy_figure"].is_relative_to(spec.out_dir)

    def test_missing_array_refused(self, spec, declared, monkeypatch):
        """
        ``requires`` is the contract the plotting function reads against, so it is checked
        where the builder is called. Left to the drawing code, a dropped array fails a
        stage later with no clue which one is missing.
        """
        _install_builder(
            monkeypatch, FigData(figure="toy_figure", arrays={"x": np.arange(4.0)})
        )

        with pytest.raises(KeyError, match="y"):
            figdata.build(spec, figures=["toy_figure"])

    def test_deferred_is_skipped_not_attempted(self, spec, monkeypatch):
        """
        A deferred declaration is the design record for a figure with no producer.
        Dispatching to it would raise on a figure nobody has claimed is buildable.
        """
        monkeypatch.setitem(
            FIGURES,
            "later_figure",
            FigureDecl(
                key="later_figure",
                title="Later",
                builder="build_absent",
                requires=(),
                sources=(),
                deferred="needs parameter estimation",
            ),
        )

        assert figdata.build(spec, figures=["later_figure"]) == {}

    def test_undeclared_figure_refused(self, spec):
        """
        A mistyped key must not read as a figure that was simply never built; the two are
        indistinguishable once the build has finished.
        """
        with pytest.raises(KeyError, match="whatever"):
            figdata.build(spec, figures=["whatever"])

    def test_every_declared_builder_is_importable(self):
        """
        Every buildable declaration names a module and a function that exist. A
        declaration pointing at a name nobody wrote fails only when the figure is first
        built, which is at the end of a campaign.
        """
        import importlib

        missing = []
        for key, declaration in FIGURES.items():
            if declaration.deferred:
                continue
            try:
                module = importlib.import_module(
                    f"sage.search.figdata.{declaration.builder}"
                )
            except ImportError:
                missing.append(f"{key}: no module {declaration.builder}")
                continue
            if not hasattr(module, declaration.builder_function):
                missing.append(
                    f"{key}: {declaration.builder} has no {declaration.builder_function}"
                )
        assert not missing, missing


class TestVerify:
    """Whether a figure set is releasable, and which parts are not."""

    def test_absent_product_reported(self, spec, declared):
        """One verdict per figure, so the answer is *which* are missing."""
        assert figdata.verify(spec, figures=["toy_figure"]) == {"toy_figure": False}

    def test_present_product_passes(self, spec, declared, monkeypatch):
        """A product carrying its declared arrays verifies."""
        _install_builder(
            monkeypatch,
            FigData(figure="toy_figure", arrays={"x": np.arange(3.0), "y": np.zeros(3)}),
        )
        figdata.build(spec, figures=["toy_figure"])

        assert figdata.verify(spec, figures=["toy_figure"]) == {"toy_figure": True}

    def test_deferred_verifies(self, spec, monkeypatch):
        """
        Its absence is the intended state. Reporting it as a failure would make a correct
        release look broken, which is how a real failure stops being noticed.
        """
        monkeypatch.setitem(
            FIGURES,
            "later_figure",
            FigureDecl(
                key="later_figure", title="Later", builder="build_absent",
                requires=(), sources=(), deferred="needs PE",
            ),
        )

        assert figdata.verify(spec, figures=["later_figure"]) == {"later_figure": True}

    def test_product_under_another_name_fails(self, spec, declared, monkeypatch):
        """
        A product loaded under the wrong figure would be drawn by the wrong plotting
        function, which fails only if the arrays happen to disagree.
        """
        _install_builder(
            monkeypatch,
            FigData(figure="something_else", arrays={"x": np.arange(3.0), "y": np.zeros(3)}),
        )
        figdata.build(spec, figures=["toy_figure"])

        assert figdata.verify(spec, figures=["toy_figure"]) == {"toy_figure": False}


class TestManifest:
    """The index that makes a released figure set self-describing."""

    def test_records_origin_and_deferral(self, spec, tmp_path):
        """
        A figure with no ``origin`` is recorded as such rather than omitted: the search
        takes nothing sgwc-1 or PyCBC does not already do, so an unattributed figure is
        one to answer for, and dropping it from the index is how it stops being answered
        for.
        """
        written = figdata.manifest(spec, tmp_path / "figures.json")
        payload = json.loads(written.read_text())

        assert payload["n_figures"] == len(FIGURES)
        assert payload["n_deferred"] == sum(1 for f in FIGURES.values() if f.deferred)
        by_key = {entry["key"]: entry for entry in payload["figures"]}
        for key, declaration in FIGURES.items():
            assert by_key[key]["origin"] == (declaration.origin or None)
            assert by_key[key]["deferred"] == (declaration.deferred or None)

    def test_every_buildable_figure_is_attributed(self):
        """
        The ruling of 2026-08-20: a figure with no counterpart in sgwc-1 or PyCBC is
        deferred, not built. So anything buildable must name where it comes from.
        """
        unattributed = [
            key
            for key, declaration in FIGURES.items()
            if not declaration.deferred and not declaration.origin
        ]
        assert not unattributed, unattributed

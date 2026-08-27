#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Figure data products: the persisted input to every figure.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Each figure is built from a file containing exactly the numbers it shows, written by an
analysis stage and read by a plotting function that computes nothing. A figure can then
be redrawn without rerunning the analysis, the numbers behind it can be released
alongside it, and a plot cannot disagree with the analysis it depicts.

Builders here read analysis products and write these files; they never recompute an
analysis quantity that a stage already produced.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

from sage.search.figdata.product import FigData

__all__ = ["FigData", "build", "load", "manifest", "verify"]


def build(spec, figures: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """
    Build the requested figure data products from the analysis outputs.

    Dispatches to the builders named in the figure declarations. Those imports happen
    inside this function rather than at module scope: every builder imports
    :class:`~sage.search.figdata.product.FigData` from this package, so importing them
    eagerly here would make the package and its builders import each other.

    A deferred figure is skipped rather than attempted. Its declaration exists as the
    design record -- what it would show and what it would need -- and has no producer, so
    dispatching to it would raise on a figure nobody has claimed is buildable.

    A builder that raises stops the build. A figure set is read together, and one figure
    silently missing from a set is worse than a set that failed: the reader takes the
    remaining figures as the whole story.
    """
    import importlib

    from sage.search.figdata.spec import FIGURES

    wanted = list(figures) if figures is not None else sorted(FIGURES)
    unknown = [key for key in wanted if key not in FIGURES]
    if unknown:
        raise KeyError(
            f"no figure declared for {unknown}; declared figures are {sorted(FIGURES)}"
        )

    directory = spec.path("figdata")
    directory.mkdir(parents=True, exist_ok=True)

    built: Dict[str, Path] = {}
    for key in wanted:
        declaration = FIGURES[key]
        if declaration.deferred:
            continue
        module = importlib.import_module(
            f"sage.search.figdata.{declaration.builder}"
            if "." not in declaration.builder
            else declaration.builder
        )
        function = getattr(module, declaration.builder_function)
        product = function(spec)
        if product is None:
            continue
        # Checked here rather than trusted from the builder: `requires` is the contract
        # the plotting function reads against, and a builder that dropped a field would
        # otherwise fail inside the drawing code, a stage later and with no clue which
        # array was missing.
        product.require(*declaration.requires)
        built[key] = product.save(directory / f"{key}.h5")
    return built


def load(spec, figure: str) -> FigData:
    """
    Load one figure's data product.

    Refuses an undeclared key before touching the filesystem, so a mistyped figure is a
    named error rather than a missing file that reads as one never built.
    """
    from sage.search.figdata.spec import FIGURES

    if figure not in FIGURES:
        raise KeyError(
            f"no figure declared for {figure!r}; declared figures are {sorted(FIGURES)}"
        )
    return FigData.load(spec.path("figdata", f"{figure}.h5"))


def verify(spec, figures: Optional[Sequence[str]] = None) -> Dict[str, bool]:
    """
    Check every product exists and matches the spec it was built from.

    Reports one verdict per figure rather than raising on the first failure: the useful
    answer to "is this figure set releasable" is which figures are not, and a check that
    stops at the first tells you one of them.

    A deferred figure verifies as ``True``. It has no producer and its absence is the
    intended state, so reporting it as a failure would make a correct release look broken.
    """
    from sage.search.figdata.spec import FIGURES

    wanted = list(figures) if figures is not None else sorted(FIGURES)
    verdicts: Dict[str, bool] = {}
    for key in wanted:
        declaration = FIGURES.get(key)
        if declaration is None:
            verdicts[key] = False
            continue
        if declaration.deferred:
            verdicts[key] = True
            continue
        path = spec.path("figdata", f"{key}.h5")
        if not path.is_file():
            verdicts[key] = False
            continue
        try:
            product = FigData.load(path)
            product.require(*declaration.requires)
            verdicts[key] = product.figure == key
        except (KeyError, OSError, ValueError):
            # A product that cannot be opened, is missing an array, or was written under
            # another figure's name is not a product this figure can be drawn from.
            verdicts[key] = False
    return verdicts


def manifest(spec, path: str | Path) -> Path:
    """
    Write the index of figures, their data products and their provenance.

    The index is what makes a released figure set self-describing: every figure, the file
    its numbers are in, what it requires, which stages produced them, and where the figure
    itself comes from. A figure with no ``origin`` is recorded as such rather than
    omitted -- the search takes nothing that sgwc-1 or PyCBC does not already do, so an
    unattributed figure is one to answer for, and dropping it from the index is how it
    stops being answered for.
    """
    import json

    from sage.search.figdata.spec import FIGURES

    verdicts = verify(spec)
    entries = []
    for key in sorted(FIGURES):
        declaration = FIGURES[key]
        entries.append(
            {
                "key": key,
                "title": declaration.title,
                "product": f"{key}.h5" if not declaration.deferred else None,
                "requires": list(declaration.requires),
                "sources": list(declaration.sources),
                "origin": declaration.origin or None,
                "deferred": declaration.deferred or None,
                "present": bool(verdicts.get(key)),
            }
        )
    payload = {
        "tag": str(spec.tag),
        "spec_hash": spec.hash(),
        "n_figures": len(entries),
        "n_present": sum(1 for entry in entries if entry["present"]),
        "n_deferred": sum(1 for entry in entries if entry["deferred"]),
        "figures": entries,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1))
    return target


def run(spec, **kwargs) -> dict:
    """
    Stage driver: build every declared figure's data product and index them.

    Separate from drawing, so a figure can be redrawn without rerunning the analysis and
    the numbers behind it can be released alongside it.
    """
    from sage.search.fingerprint import combine

    built = build(spec, figures=kwargs.get("figures"))
    index = manifest(spec, spec.path("figdata", "figures.json"))
    verdicts = verify(spec)
    return {
        "manifest": str(index),
        "n_built": len(built),
        "n_verified": sum(1 for ok in verdicts.values() if ok),
        "n_figures": len(verdicts),
        "fingerprint": combine(len(built), sum(1 for ok in verdicts.values() if ok)),
    }

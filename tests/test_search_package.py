#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_package.py
Description   : Package-level contracts: declared exports resolve, and importing is cheap.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Two contracts that are cheap to state and expensive to discover the hard way.

A name in ``__all__`` that resolves to nothing is worse than no export at all: it reads
as a public API, survives review, and fails only when someone imports it. Three such
names existed in this subpackage before this test.

Importing ``sage.search`` must stay cheap. The stage registry and the specification are
read by tooling, by the figure layer and by anything inspecting a campaign, none of which
wants a CUDA context or an HDF5 library loaded as a side effect. Since the orchestrator
must eventually dispatch to modules that do import torch, the only way to keep this true
is for that dispatch to happen inside function bodies, and the only way to keep it true
in a year is to assert it.

Runs anywhere; needs no data, no GPU and no network.
"""

import subprocess
import sys

import pytest

SEARCH_PACKAGES = [
    "sage.search",
    "sage.search.injection",
    "sage.search.sensitivity",
    "sage.search.pastro",
    "sage.search.catalogue",
    "sage.search.characterize",
    "sage.search.figdata",
    "sage.search.release",
]


class TestDeclaredExports:
    """Every name a package advertises must resolve to something."""

    @pytest.mark.parametrize("package", SEARCH_PACKAGES)
    def test_all_names_resolve(self, package):
        """``__all__`` contains no name the package cannot produce."""
        import importlib

        pkg = importlib.import_module(package)
        missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
        assert not missing, f"{package}.__all__ names unresolvable: {missing}"

    @pytest.mark.parametrize("package", SEARCH_PACKAGES)
    def test_all_is_sorted_and_unique(self, package):
        """A duplicated export is a merge artefact; an unsorted one invites duplicates."""
        import importlib

        pkg = importlib.import_module(package)
        names = list(pkg.__all__)
        assert len(names) == len(set(names)), f"{package}.__all__ has duplicates"

    @pytest.mark.parametrize("package", SEARCH_PACKAGES)
    def test_unknown_attribute_raises_attribute_error(self, package):
        """A typo gets AttributeError, not ImportError or a silent None."""
        import importlib

        pkg = importlib.import_module(package)
        with pytest.raises(AttributeError):
            getattr(pkg, "definitely_not_a_real_export")

    @pytest.mark.parametrize("package", SEARCH_PACKAGES)
    def test_dir_includes_exports(self, package):
        """Tab completion and introspection see the lazy names."""
        import importlib

        pkg = importlib.import_module(package)
        listed = set(dir(pkg))
        assert set(pkg.__all__) <= listed


class TestImportCost:
    """Importing the package must not pull the heavy stack."""

    def test_import_does_not_pull_torch_or_h5py(self):
        """
        ``import sage.search`` leaves torch and h5py unloaded.

        Checked in a subprocess because this test session has already imported torch, so
        an in-process check would pass regardless of what the package does.
        """
        code = (
            "import sys; import sage.search; "
            "heavy = [m for m in ('torch', 'h5py') if m in sys.modules]; "
            "print(','.join(heavy))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert out.returncode == 0, out.stderr
        loaded = out.stdout.strip()
        assert not loaded, f"importing sage.search pulled: {loaded}"

    def test_every_submodule_imports(self):
        """
        No module in the subpackage is import-broken.

        Cheap to run and it catches a dangling import the moment it lands, rather than
        when some later layer first touches that module.
        """
        import importlib
        import pathlib
        import pkgutil

        import sage.search

        root = pathlib.Path(sage.search.__file__).parent
        failures = []
        for info in pkgutil.walk_packages([str(root)], prefix="sage.search."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
        assert not failures, "modules failed to import:\n" + "\n".join(failures)

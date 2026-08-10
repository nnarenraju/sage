#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : _lazy.py
Description   : Deferred re-exports for the search packages.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A package here advertises a small public surface while its modules pull heavy
dependencies: torch for the engine, h5py for the trigger shards, network clients for the
catalogue readers. Importing the package eagerly would load all of them to satisfy a
caller who wanted one name, and would make ``import sage.search`` cost a CUDA context.

So each package declares a table of ``name -> where it lives`` and the import happens on
first access. The cost is paid by whoever uses the name, and only once.
"""

import importlib
import sys
from typing import Callable, Dict, Tuple, Union

# Either "module", meaning the attribute has the same name as the export, or
# ("module", "attribute") where the public name and the internal one differ.
Target = Union[str, Tuple[str, str]]


def lazy_exports(package: str, table: Dict[str, Target]) -> Tuple[Callable, Callable]:
    """
    Build the ``__getattr__`` and ``__dir__`` a package needs to export lazily.

    Parameters
    ----------
    package : str
        Importing package, normally ``__name__``.
    table : dict
        Public name to the module it lives in, or to a ``(module, attribute)`` pair when
        the exported name differs from the internal one.

    Returns
    -------
    tuple
        ``(__getattr__, __dir__)``, to be assigned at module scope.

    Notes
    -----
    An unknown name raises ``AttributeError``, not ``ImportError``, so a typo behaves the
    way an ordinary missing attribute does. A name that is declared but whose module
    cannot provide it raises ``AttributeError`` too, naming both, since that means the
    table has drifted from the code.
    """

    def __getattr__(name: str):
        try:
            target = table[name]
        except KeyError:
            raise AttributeError(
                f"module {package!r} has no attribute {name!r}"
            ) from None
        module_name, attribute = (target, name) if isinstance(target, str) else target
        module = importlib.import_module(f"{package}.{module_name}")
        try:
            return getattr(module, attribute)
        except AttributeError:
            raise AttributeError(
                f"{package}.{module_name} does not define {attribute!r}, "
                f"which {package} exports as {name!r}"
            ) from None

    def __dir__():
        # Defining __dir__ replaces the default listing of the module namespace, so the
        # names that are already bound have to be added back alongside the lazy ones.
        bound = vars(sys.modules[package]) if package in sys.modules else {}
        return sorted(set(table) | set(bound))

    return __getattr__, __dir__

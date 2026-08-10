#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Ingest published event catalogues into one internal schema.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Catalogues differ in units, frames, significance conventions and naming, so each source
is normalised on ingest and its conventions recorded with it. Significance values are
not interchangeable between groups, and the internal record keeps enough provenance that
a comparison never silently places two incompatible numbers on the same axis.
"""

from sage.search._lazy import lazy_exports

_EXPORTS = {
    "CatalogueCache": "cache",
    "CatalogueEvent": "record",
    "Conventions": "record",
    "ExternalCatalogue": "record",
    "REGISTRY": "external",
    "load_all": "external",
    "load_catalogue": "external",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

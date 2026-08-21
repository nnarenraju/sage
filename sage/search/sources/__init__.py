#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : __init__.py
Description   : Per-release fetchers and handlers for external data.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

**Every module in here is expected to rot, and that is the design.**

External data -- population hyperposteriors, sub-threshold lists, another group's
catalogue, an iDQ archive -- is published in a layout that serves the paper that released
it. The layout changes between releases, sometimes completely: the GWTC-3 population
release ships a bilby result JSON, and the GWTC-4.0 one ships ``popsummary`` HDF5 for the
same model. Code written against either is wrong about the other, and code written to
handle both is wrong about the third.

So each release gets **its own module**, named for the release rather than for the
quantity, and a new release gets a **new file** rather than an edit to an old one. The
old module keeps working against the data it was written for, which is what makes an
analysis re-runnable after the source has moved on.

What does *not* change is the far side. Each module converts its release into one of the
canonical forms below, and nothing outside this package knows which release it came from:

``hyperposterior``
    A flat mapping of hyperparameter name to float, plus provenance, written as JSON and
    read by :func:`sage.search.injection.campaign._hyperposterior`. The Power-Law + Peak
    contract is the fourteen names :mod:`sage.search.injection.population` reads:
    ``alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m, mu_chi, sigma_chi, amax,
    xi_spin, sigma_spin, lamb``.

``event times``
    Anything with times in it -- an external catalogue, a sub-threshold list, a glitch
    list -- becomes an :class:`~sage.search.catalogue.record.ExternalCatalogue` through
    :func:`sage.search.catalogue.eventlist.from_times`, which already accepts GPS, UTC or
    event names.

Two rules keep the rot contained:

1. **Release-specific dependencies live here and nowhere else.** ``bilby``,
   ``popsummary`` and a release's directory layout are this package's problem. The
   handler writes the canonical form once, into the campaign's export directory, and the
   search reads that -- so a compute node running the campaign needs none of them.
2. **Provenance travels with the value.** Each module states its record, DOI and the file
   it read, and stamps them onto what it writes. A hyperparameter with no source is a
   number someone typed.
"""

from sage.search._lazy import lazy_exports

_EXPORTS: dict = {}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

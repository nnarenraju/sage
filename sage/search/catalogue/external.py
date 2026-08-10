#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : external.py
Description   : Independent catalogues published outside the collaboration releases.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Several independent groups have published candidates absent from the collaboration
releases, so a claim that a Sage candidate is new has to be checked against them too.

Formats vary. Some groups publish structured trigger files with a shared column layout;
others publish only tables in the paper, which are transcribed here with the source
version pinned, since near-threshold values move between revisions.

Each source records the region it actually searched. Where a candidate falls outside
that region, its absence from that catalogue carries no information and is reported as
uncovered rather than as a non-detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from sage.search.catalogue.record import Conventions, ExternalCatalogue


@dataclass(frozen=True)
class CatalogueSource:
    """A registered external catalogue and how to read it."""

    key: str
    loader: Callable
    conventions: Conventions
    reference: str
    version: str
    location: str = ""
    transcribed: bool = False


REGISTRY: Dict[str, CatalogueSource] = {}

EXCLUDED: tuple = ()


def register(source: CatalogueSource) -> None:
    """Add a source, refusing any name on the exclusion list."""
    raise NotImplementedError


def is_excluded(name: str) -> bool:
    """
    Whether a catalogue name is excluded from all comparisons.

    Matching is on whole words so that ordinary text containing the same letters is not
    caught by accident.
    """
    raise NotImplementedError


def load_catalogue(key: str, cache) -> ExternalCatalogue:
    """Load one registered catalogue."""
    raise NotImplementedError


def load_all(cache, keys: Optional[Sequence[str]] = None) -> Dict[str, ExternalCatalogue]:
    """Load every registered catalogue, or a chosen subset."""
    raise NotImplementedError


def read_trigger_file(path: str | Path, conventions: Conventions) -> ExternalCatalogue:
    """
    Read a structured trigger file in the shared column layout.

    Significance is stored as a rate in some releases and as its inverse in others, so
    the convention is taken from the source record rather than inferred from the column.
    """
    raise NotImplementedError


def read_transcribed_table(path: str | Path, conventions: Conventions) -> ExternalCatalogue:
    """Read a table transcribed from a publication, with its version pinned."""
    raise NotImplementedError


def dedup_across_sources(
    catalogues: Dict[str, ExternalCatalogue],
    precedence: Sequence[str],
    tolerance_s: float = 1.0,
) -> ExternalCatalogue:
    """
    Merge catalogues into one event list.

    Where the same event appears in several sources, the parameters are taken from the
    first source in the precedence order that provides them.
    """
    raise NotImplementedError

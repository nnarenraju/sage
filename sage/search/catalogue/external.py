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
    """
    Add a source, refusing any name on the exclusion list.

    The registry holds *how to obtain* each source, not a parser for it: a loader is a
    callable returning an :class:`~sage.search.catalogue.record.ExternalCatalogue`, and
    the short wrappers that produce one are written when the source's data is local.
    """
    if source.key in REGISTRY:
        raise ValueError(
            f"catalogue {source.key!r} is already registered as {REGISTRY[source.key]}"
        )
    if is_excluded(source.key):
        raise ValueError(
            f"{source.key!r} is on the exclusion list and may not be registered"
        )
    REGISTRY[source.key] = source


def is_excluded(name: str) -> bool:
    """
    Whether a catalogue name is excluded from all comparisons.

    Matching is on whole words so that ordinary text containing the same letters is not
    caught by accident.

    ``EXCLUDED`` is empty and is expected to stay so. A group whose *methods* are not a
    reference here still publishes results, and results are compared like anyone else's;
    excluding a source from a comparison because of how it was produced would make the
    comparison a statement about our opinion of them rather than about the data.
    """
    import re

    text = str(name)
    return any(
        re.search(rf"\b{re.escape(entry)}\b", text, flags=re.IGNORECASE)
        for entry in EXCLUDED
    )


def load_catalogue(key: str, cache) -> ExternalCatalogue:
    """Load one registered catalogue through its own loader."""
    if key not in REGISTRY:
        import difflib

        close = difflib.get_close_matches(str(key), sorted(REGISTRY), n=3)
        hint = f"; did you mean {close}" if close else ""
        raise KeyError(
            f"no catalogue registered as {key!r}. Registered: {sorted(REGISTRY)}{hint}"
        )
    return REGISTRY[key].loader(cache)


def load_all(cache, keys: Optional[Sequence[str]] = None) -> Dict[str, ExternalCatalogue]:
    """
    Load every registered catalogue, or a chosen subset.

    A source that fails to load raises rather than being skipped. A comparison quietly
    missing one catalogue reports every candidate it would have matched as new, which is
    the most interesting possible answer and the one most likely to be wrong.
    """
    return {
        key: load_catalogue(key, cache)
        for key in (keys if keys is not None else sorted(REGISTRY))
    }


def read_trigger_file(path: str | Path, conventions: Conventions) -> ExternalCatalogue:
    """
    Deferred. Read an external source through
    :func:`sage.search.catalogue.eventlist.read_event_times` instead.

    There is deliberately no per-source parser here. Every external comparison reduces to
    times and significances, and everything else about a source's file layout is
    discarded before the comparison sees it -- so a parser written against one release is
    a thing to maintain against every future one, for a part of the file this analysis
    does not use. All of these sources restructure between releases: Zenodo rearranges,
    trigger tables gain columns, iDQ's archive layout is not promised to be stable.

    When a source is actually needed and its data is sitting locally, a short wrapper is
    written *then*, against the layout in hand, and it produces times and calls
    :func:`~sage.search.catalogue.eventlist.from_times`. That wrapper is cheap to write
    and cheap to throw away, which is the correct lifetime for code that tracks somebody
    else's release format.
    """
    raise NotImplementedError(
        "per-source parsers are deliberately not built. Use "
        "sage.search.catalogue.eventlist.read_event_times for a table of times, or "
        "from_times() from a short wrapper written against the data you have locally"
    )


def read_transcribed_table(path: str | Path, conventions: Conventions) -> ExternalCatalogue:
    """
    Deferred, in favour of :func:`sage.search.catalogue.eventlist.read_event_times`.

    That reader already accepts what a person produces when copying a table out of a
    paper: any column order, comma- or whitespace-separated, with or without a header,
    and times written as GPS, as a UTC string, or as an event name that carries one.
    Converting times by hand was the step this replaces, and it was where the
    transcription errors were.
    """
    raise NotImplementedError(
        "use sage.search.catalogue.eventlist.read_event_times, which accepts GPS, UTC "
        "strings and event names directly"
    )


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

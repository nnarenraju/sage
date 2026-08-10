#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : store.py
Description   : The queryable store holding every recorded fact about every candidate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A search produces facts about candidates from many stages: significance, probability,
data quality, consistency tests, parameter estimates, localisation, catalogue matches
and sensitivity context. Scattered across per-stage files these are awkward to relate,
and answering an ordinary question ends up meaning reading several files and joining
them by hand.

This module keeps one relational database per campaign. Stages write into it; every
later question is a query. Anything recorded about an event can be retrieved by name,
and any set of events can be selected by an arbitrary condition over any recorded
quantity, including conditions spanning stages.

The backend is SQLite: a single portable file, queryable with plain SQL, readable by
external tools, and requiring nothing beyond the standard library. Results are returned
as data frames for interactive use. Bulk arrays such as spectrograms and posterior
samples stay in their own files; the database records where they are, so a candidate's
record resolves to everything about it without duplicating large data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

TABLES: Tuple[str, ...] = (
    "campaign",
    "runs",
    "arms",
    "events",
    "triggers",
    "significance",
    "trials",
    "pastro",
    "dataquality",
    "consistency",
    "followup",
    "parameters",
    "skymaps",
    "catalogue_events",
    "catalogue_matches",
    "catalogue_coverage",
    "injections",
    "injection_recovery",
    "sensitivity",
    "background",
    "livetime",
    "artefacts",
    "provenance",
)


@dataclass
class EventRecord:
    """Everything recorded about one candidate, gathered across stages."""

    name: str
    gps: float
    fields: Dict[str, Any] = field(default_factory=dict)
    dataquality: Dict[str, Any] = field(default_factory=dict)
    consistency: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    catalogue: Dict[str, Any] = field(default_factory=dict)
    artefacts: Dict[str, Path] = field(default_factory=dict)

    def summary(self) -> str:
        """Readable one-screen summary of the candidate."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Flat mapping of every recorded quantity."""
        raise NotImplementedError

    def to_markdown(self) -> str:
        """Readable table form, for a report or a message."""
        raise NotImplementedError

    def artefact(self, kind: str) -> Path:
        """Resolve a stored artefact, such as a spectrogram or posterior file."""
        raise NotImplementedError


class SearchStore:
    """
    The campaign database.

    Opened once per campaign and shared by every stage. Writes are idempotent on the
    natural key of each table, so re-running a stage replaces its rows rather than
    duplicating them.

    Examples
    --------
    Retrieve one candidate with everything known about it::

        store.event("SGW230814_230901")

    Select a set by any condition over any recorded quantity::

        store.select(where="pastro > 0.9 AND dq_vetoed = 0 AND ifar_yr > 100")

    Ask a question spanning stages::

        store.query('''
            SELECT e.name, s.ifar_yr, p.pastro, c.catalogue, c.dt_s
            FROM events e
            JOIN significance s USING (name)
            JOIN pastro p       USING (name)
            LEFT JOIN catalogue_matches c USING (name)
            WHERE p.pastro > 0.5
            ORDER BY s.ifar_yr DESC
        ''')
    """

    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        raise NotImplementedError

    @classmethod
    def open(cls, spec, read_only: bool = False) -> "SearchStore":
        """Open (creating if needed) the store for a campaign."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the connection."""
        raise NotImplementedError

    def __enter__(self) -> "SearchStore":
        """Context-manager entry."""
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        """Context-manager exit."""
        raise NotImplementedError

    # -- schema -------------------------------------------------------------

    def initialise(self) -> None:
        """Create the schema and its indices."""
        raise NotImplementedError

    def schema(self, table: Optional[str] = None) -> str:
        """Show the schema, for one table or all of them."""
        raise NotImplementedError

    def describe(self) -> str:
        """Readable overview: tables, row counts and what each holds."""
        raise NotImplementedError

    def columns(self, table: Optional[str] = None) -> Dict[str, Tuple[str, ...]]:
        """Available column names, for discovering what can be queried."""
        raise NotImplementedError

    # -- writing ------------------------------------------------------------

    def put(self, table: str, rows, key: Optional[Sequence[str]] = None) -> int:
        """Insert or replace rows, keyed on the table's natural key."""
        raise NotImplementedError

    def put_events(self, candidates) -> int:
        """Record the candidate table."""
        raise NotImplementedError

    def put_significance(self, candidates, far_curve) -> int:
        """Record false-alarm rate, inverse rate and p-value per candidate."""
        raise NotImplementedError

    def put_pastro(self, table) -> int:
        """Record astrophysical probability and its credible interval."""
        raise NotImplementedError

    def put_dataquality(self, reports) -> int:
        """Record each data-quality task's outcome per candidate."""
        raise NotImplementedError

    def put_consistency(self, results) -> int:
        """Record each consistency test's outcome per candidate."""
        raise NotImplementedError

    def put_parameters(self, pe_results, level: float = 0.9) -> int:
        """Record parameter medians and credible intervals, with their provenance."""
        raise NotImplementedError

    def put_catalogue(self, catalogues, matches, coverage) -> int:
        """Record external events, their matches and each catalogue's coverage."""
        raise NotImplementedError

    def put_sensitivity(self, results) -> int:
        """Record sensitivity at each threshold and reference point."""
        raise NotImplementedError

    def put_artefact(self, name: str, kind: str, path: str | Path, **attrs) -> None:
        """Record where a bulk artefact for a candidate is stored."""
        raise NotImplementedError

    def put_provenance(self, stage: str, attrs: Mapping[str, Any]) -> None:
        """Record how a stage's rows were produced."""
        raise NotImplementedError

    # -- reading ------------------------------------------------------------

    def event(self, name: str) -> EventRecord:
        """Everything recorded about one candidate."""
        raise NotImplementedError

    def events(self, names: Optional[Sequence[str]] = None, where: Optional[str] = None) -> List[EventRecord]:
        """Full records for a set of candidates."""
        raise NotImplementedError

    def at_gps(self, gps: float, tolerance_s: float = 1.0) -> List[EventRecord]:
        """Candidates near a time, for cross-checking an externally reported event."""
        raise NotImplementedError

    def select(
        self,
        where: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """
        Select candidates by a condition over any recorded quantity.

        The condition is evaluated against a joined view spanning every per-candidate
        table, so quantities from different stages can be combined in one expression
        without writing the joins.
        """
        raise NotImplementedError

    def query(self, sql: str, params: Optional[Sequence] = None):
        """Run arbitrary SQL and return the result as a data frame."""
        raise NotImplementedError

    def table(self, name: str):
        """Read one table in full."""
        raise NotImplementedError

    def joined(self):
        """The per-candidate view that :meth:`select` filters over."""
        raise NotImplementedError

    # -- comparison and export ---------------------------------------------

    def compare(self, names: Sequence[str], columns: Optional[Sequence[str]] = None):
        """Place several candidates side by side on the same quantities."""
        raise NotImplementedError

    def comparison_matrix(self, tolerance_s: float = 1.0):
        """
        Candidates against catalogues.

        Distinguishes found, searched but not found, and outside a catalogue's searched
        region, since the last says nothing about the candidate.
        """
        raise NotImplementedError

    def tier_counts(self):
        """How many candidates fall in each inclusion tier."""
        raise NotImplementedError

    def export(self, path: str | Path, fmt: str = "csv", where: Optional[str] = None) -> Path:
        """Write a selection out as csv, markdown, latex or hdf5."""
        raise NotImplementedError

    def to_pandas(self, table: Optional[str] = None):
        """Whole table, or the joined view, as a data frame."""
        raise NotImplementedError


def open_store(spec, read_only: bool = False) -> SearchStore:
    """Open the campaign store for a spec."""
    raise NotImplementedError


def build_from_products(spec, overwrite: bool = False) -> SearchStore:
    """
    Populate a store from the products a campaign has already written.

    Lets a store be rebuilt from stage outputs without re-running the analysis.
    """
    raise NotImplementedError

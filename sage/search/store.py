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

import datetime
import difflib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

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

# Bumped whenever the schema changes in a way an existing file cannot answer. Stored in
# the database itself, so an old campaign store is refused rather than half-read.
SCHEMA_VERSION: int = 1

# The per-candidate view :meth:`SearchStore.select` filters over, and the name to use for
# it in hand-written SQL.
VIEW_NAME: str = "candidates"

# Stamped onto every row of every table but ``provenance``, whose own columns are the
# stamp. Any row can then be traced to the stage that wrote it and the spec it was
# produced under, without joining anything.
BOOKKEEPING: Tuple[Tuple[str, str], ...] = (
    ("stage", "TEXT"),
    ("spec_hash", "TEXT"),
    ("written_utc", "TEXT"),
)

EXPORT_FORMATS: Tuple[str, ...] = ("csv", "markdown", "latex", "hdf5")


class UnknownName(ValueError):
    """
    A name the store does not hold, reported with its near neighbours.

    Interactive querying is mostly typing column names from memory. A bare ``KeyError``
    on a mistyped one says the query found nothing, which is indistinguishable from a
    correct query over an empty stage; naming the near neighbours turns the mistake into
    the answer.
    """


class UnknownTable(UnknownName):
    """A table absent from the schema."""


class UnknownColumn(UnknownName):
    """A column absent from the table or view being queried."""


class UnknownEvent(UnknownName):
    """A candidate absent from the store."""


class UnknownArtefact(UnknownName):
    """An artefact kind not recorded for a candidate."""


@dataclass(frozen=True)
class TableSpec:
    """One table's columns, its natural key and what it holds."""

    columns: Tuple[Tuple[str, str], ...]
    key: Tuple[str, ...]
    note: str

    def column_names(self) -> Tuple[str, ...]:
        """Declared columns, in schema order, without the bookkeeping stamp."""
        return tuple(name for name, _ in self.columns)


# Column names are unique across the per-candidate tables, so the joined view can carry
# them unqualified and a condition can name any quantity without knowing its stage.
SCHEMA: Dict[str, TableSpec] = {
    "campaign": TableSpec(
        columns=(
            ("tag", "TEXT NOT NULL"),
            ("spec_hash", "TEXT"),
            ("config_module", "TEXT"),
            ("observing_run", "TEXT"),
            ("detectors", "TEXT"),
            ("out_dir", "TEXT"),
            ("created_utc", "TEXT"),
        ),
        key=("tag",),
        note="the campaign this store belongs to",
    ),
    "runs": TableSpec(
        columns=(
            ("observing_run", "TEXT NOT NULL"),
            ("detectors", "TEXT"),
            ("gps_start", "REAL"),
            ("gps_end", "REAL"),
            ("n_segments", "INTEGER"),
            ("coincident_s", "REAL"),
            ("analysed_s", "REAL"),
        ),
        key=("observing_run",),
        note="observing runs searched, and their spans",
    ),
    "arms": TableSpec(
        columns=(
            ("arm", "TEXT NOT NULL"),
            ("detectors", "TEXT"),
            ("observing_run", "TEXT"),
            ("livetime_s", "REAL"),
            ("internal", "INTEGER DEFAULT 1"),
            ("far_curve_path", "TEXT"),
            ("note", "TEXT"),
        ),
        key=("arm",),
        note="analyses competing for the trials factor",
    ),
    "events": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("gps", "REAL"),
            ("tc_gps", "REAL"),
            ("stat", "REAL"),
            ("arm", "TEXT"),
            ("observing_run", "TEXT"),
            ("detectors", "TEXT"),
            ("slide_id", "INTEGER DEFAULT 0"),
            # Undetermined until a tier stage assigns one, and never promoted here:
            # this table records what it is given and otherwise holds no verdict. The
            # value is ``candidates.TIER_UNDETERMINED``, written out rather than
            # imported because the tier stages run after the store is written and a
            # module import of theirs would invert the stage graph.
            ("tier", "INTEGER DEFAULT -1"),
            ("tier_trials", "INTEGER DEFAULT -1"),
            ("tier_provisional", "INTEGER DEFAULT 1"),
            ("note", "TEXT"),
        ),
        key=("name",),
        note="the candidate list; every other per-candidate table keys on it",
    ),
    "triggers": TableSpec(
        columns=(
            ("arm", "TEXT NOT NULL"),
            ("slide_id", "INTEGER NOT NULL"),
            ("gps", "REAL NOT NULL"),
            ("stat", "REAL"),
            ("block_id", "INTEGER"),
            ("segment_index", "INTEGER"),
            ("first_local", "INTEGER"),
            ("detectors", "TEXT"),
        ),
        key=("arm", "slide_id", "gps"),
        note="clustered triggers, zero-lag and slid",
    ),
    "significance": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("far_per_yr", "REAL"),
            ("ifar_yr", "REAL"),
            ("far_hierarchical_per_yr", "REAL"),
            ("p_value", "REAL"),
            ("n_louder", "REAL"),
            ("background_livetime_s", "REAL"),
        ),
        key=("name",),
        note="single-arm false-alarm rate and p-value",
    ),
    "trials": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("n_trials", "INTEGER"),
            ("trials_convention", "TEXT"),
            ("covered_by", "TEXT"),
            ("found_by", "TEXT"),
            ("best_arm", "TEXT"),
            ("far_trials_per_yr", "REAL"),
            ("ifar_trials_yr", "REAL"),
            ("p_value_trials", "REAL"),
        ),
        key=("name",),
        note="trials-corrected significance, carried beside the uncorrected",
    ),
    "pastro": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("pastro", "REAL"),
            ("pastro_lo", "REAL"),
            ("pastro_hi", "REAL"),
            ("pastro_category", "TEXT"),
            ("pastro_model", "TEXT"),
            ("mchirp", "REAL"),
            ("mchirp_sigma", "REAL"),
        ),
        key=("name",),
        note="astrophysical probability and its credible interval",
    ),
    "dataquality": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("task", "TEXT NOT NULL"),
            ("detector", "TEXT"),
            ("p_value", "REAL"),
            ("passed", "INTEGER"),
            ("threshold", "REAL"),
            ("detail", "TEXT"),
        ),
        key=("name", "task"),
        note="one row per data-quality task per candidate",
    ),
    "consistency": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("test", "TEXT NOT NULL"),
            ("passed", "INTEGER"),
            ("value", "REAL"),
            ("available", "INTEGER DEFAULT 1"),
            ("detail", "TEXT"),
        ),
        key=("name", "test"),
        note="one row per consistency test per candidate",
    ),
    "followup": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("detector", "TEXT NOT NULL"),
            ("snr", "REAL"),
            ("chisq", "REAL"),
            ("chisq_dof", "INTEGER"),
            ("reduced_chisq", "REAL"),
            ("mass1", "REAL"),
            ("mass2", "REAL"),
            ("spin1z", "REAL"),
            ("spin2z", "REAL"),
            ("dt_s", "REAL"),
            ("template_id", "TEXT"),
        ),
        key=("name", "detector"),
        note="matched-filter follow-up, per detector",
    ),
    "parameters": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("parameter", "TEXT NOT NULL"),
            ("median", "REAL"),
            ("lower", "REAL"),
            ("upper", "REAL"),
            ("level", "REAL"),
            ("waveform", "TEXT"),
            ("sampler", "TEXT"),
            ("source", "TEXT"),
        ),
        key=("name", "parameter"),
        note="parameter medians and credible intervals",
    ),
    "skymaps": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("skymap_path", "TEXT"),
            ("area_50_deg2", "REAL"),
            ("area_90_deg2", "REAL"),
            ("distance_mean_mpc", "REAL"),
            ("distance_std_mpc", "REAL"),
            ("skymap_source", "TEXT"),
        ),
        key=("name",),
        note="localisation summaries",
    ),
    "catalogue_events": TableSpec(
        columns=(
            ("catalogue", "TEXT NOT NULL"),
            ("event_name", "TEXT NOT NULL"),
            ("gps", "REAL"),
            ("far_per_yr", "REAL"),
            ("ifar_yr", "REAL"),
            ("p_astro", "REAL"),
            ("network_snr", "REAL"),
            ("mass1", "REAL"),
            ("mass2", "REAL"),
            ("chirp_mass", "REAL"),
            ("redshift", "REAL"),
            ("luminosity_distance", "REAL"),
            ("chi_eff", "REAL"),
            ("posterior_url", "TEXT"),
        ),
        key=("catalogue", "event_name"),
        note="events published by external catalogues",
    ),
    "catalogue_matches": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("catalogue", "TEXT NOT NULL"),
            ("match_name", "TEXT"),
            ("dt_s", "REAL"),
            ("matched", "INTEGER DEFAULT 0"),
        ),
        key=("name", "catalogue"),
        note="candidate against catalogue, matched on GPS time",
    ),
    "catalogue_coverage": TableSpec(
        columns=(
            ("catalogue", "TEXT NOT NULL"),
            ("name", "TEXT NOT NULL"),
            ("covered", "INTEGER"),
            ("reason", "TEXT"),
        ),
        key=("catalogue", "name"),
        note="whether a catalogue searched where a candidate lies",
    ),
    "injections": TableSpec(
        columns=(
            ("stream", "INTEGER NOT NULL"),
            ("injection_id", "INTEGER NOT NULL"),
            ("gps", "REAL"),
            ("mass1", "REAL"),
            ("mass2", "REAL"),
            ("spin1z", "REAL"),
            ("spin2z", "REAL"),
            ("distance_mpc", "REAL"),
            ("inclination", "REAL"),
            ("ra", "REAL"),
            ("dec", "REAL"),
            ("polarization", "REAL"),
            ("optimal_snr", "REAL"),
            ("draw_pdf", "REAL"),
            ("population", "TEXT"),
        ),
        key=("stream", "injection_id"),
        note="the injection set, as drawn",
    ),
    "injection_recovery": TableSpec(
        columns=(
            ("stream", "INTEGER NOT NULL"),
            ("injection_id", "INTEGER NOT NULL"),
            ("arm", "TEXT NOT NULL"),
            ("found", "INTEGER"),
            ("stat", "REAL"),
            ("far_per_yr", "REAL"),
            ("dt_s", "REAL"),
            ("name", "TEXT"),
        ),
        key=("stream", "injection_id", "arm"),
        note="which injections each arm recovered",
    ),
    "sensitivity": TableSpec(
        columns=(
            ("population", "TEXT NOT NULL"),
            ("far_threshold_per_yr", "REAL NOT NULL"),
            ("arm", "TEXT NOT NULL"),
            ("vt", "REAL"),
            ("vt_err", "REAL"),
            ("n_effective", "REAL"),
            ("n_found", "INTEGER"),
            ("n_generated", "INTEGER"),
            ("analysis_time_s", "REAL"),
            ("relative_error", "REAL"),
            ("plottable", "INTEGER"),
            ("sensitive_volume", "REAL"),
            ("range_mpc", "REAL"),
        ),
        key=("population", "far_threshold_per_yr", "arm"),
        note="sensitive volume-time per threshold and reference point",
    ),
    "background": TableSpec(
        columns=(
            ("arm", "TEXT NOT NULL"),
            ("idx", "INTEGER NOT NULL"),
            ("stat", "REAL"),
            ("n_louder", "REAL"),
            ("far_per_yr", "REAL"),
            ("livetime_s", "REAL"),
            ("n_slides", "INTEGER"),
        ),
        key=("arm", "idx"),
        note="the measured background, as a FAR curve",
    ),
    "livetime": TableSpec(
        columns=(
            ("arm", "TEXT NOT NULL"),
            ("observing_run", "TEXT NOT NULL"),
            ("zerolag_s", "REAL"),
            ("background_s", "REAL"),
            ("coincident_s", "REAL"),
            ("analysed_s", "REAL"),
            ("n_windows", "INTEGER"),
            ("n_slides", "INTEGER"),
        ),
        key=("arm", "observing_run"),
        note="the time behind every rate this store quotes",
    ),
    "artefacts": TableSpec(
        columns=(
            ("name", "TEXT NOT NULL"),
            ("kind", "TEXT NOT NULL"),
            ("path", "TEXT"),
            ("bytes", "INTEGER"),
            ("sha256", "TEXT"),
            ("attrs", "TEXT"),
        ),
        key=("name", "kind"),
        note="where each candidate's bulk data lives",
    ),
    "provenance": TableSpec(
        columns=(
            ("stage", "TEXT NOT NULL"),
            ("spec_hash", "TEXT"),
            ("sage_version", "TEXT"),
            ("git_hash", "TEXT"),
            ("config_module", "TEXT"),
            ("created_utc", "TEXT"),
            ("attrs", "TEXT"),
        ),
        key=("stage",),
        note="how each stage's rows were produced",
    ),
}

# Beyond the natural keys, which SQLite indexes already. Time lookups and the
# per-candidate gathers are the two access patterns that are not key lookups.
INDICES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("events", ("gps",)),
    ("triggers", ("gps",)),
    ("catalogue_events", ("gps",)),
    ("injections", ("gps",)),
    ("dataquality", ("name",)),
    ("consistency", ("name",)),
    ("followup", ("name",)),
    ("parameters", ("name",)),
    ("catalogue_matches", ("name",)),
    ("catalogue_coverage", ("name",)),
    ("artefacts", ("name",)),
    ("injection_recovery", ("name",)),
)

# Tables holding at most one row per candidate, joined straight onto ``events`` in the
# view. Earlier entries win a name collision, though the schema avoids them.
VIEW_JOINS: Tuple[str, ...] = ("significance", "trials", "pastro", "skymaps")

# Tables holding several rows per candidate, folded into one row by an aggregate. A
# verdict is the conjunction over its tasks, so the summary is derived rather than
# stored twice and cannot disagree with the rows behind it.
VIEW_AGGREGATES: Tuple[str, ...] = (
    """
    SELECT name,
           MAX(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS dq_vetoed,
           MIN(p_value)                                AS dq_p_value,
           COUNT(*)                                    AS dq_tasks,
           SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS dq_failed
    FROM dataquality GROUP BY name
    """,
    """
    SELECT name,
           COUNT(*)                                    AS consistency_tests,
           SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS consistency_failed
    FROM consistency WHERE available = 1 GROUP BY name
    """,
    # The closest match, taken from one row rather than assembled from two aggregates.
    # MIN(catalogue) beside MIN(ABS(dt_s)) reports the alphabetically first catalogue
    # next to the smallest separation, and for a candidate matching more than one
    # catalogue that pair describes no match anyone recorded. SQLite's bare-column rule
    # makes the other columns of the MIN() row available, which is what picks a row.
    """
    SELECT name,
           COUNT(*)                          AS n_catalogue_matches,
           catalogue                         AS catalogue_match,
           dt_s                              AS catalogue_dt_s,
           MIN(ABS(dt_s))                    AS catalogue_abs_dt_s
    FROM catalogue_matches WHERE matched = 1 GROUP BY name
    """,
    # Matched-filter follow-up is per detector, so the view carries the network summary
    # a condition would actually be written against. Without it a whole stage's output
    # is unreachable through select(), which reports the quantity as unknown rather than
    # as recorded elsewhere.
    """
    SELECT name,
           COUNT(*)                          AS followup_detectors,
           SQRT(SUM(snr * snr))              AS network_snr,
           MAX(reduced_chisq)                AS max_reduced_chisq,
           MAX(ABS(dt_s))                    AS max_followup_dt_s
    FROM followup GROUP BY name
    """,
    """
    SELECT name,
           COUNT(*)                          AS catalogues_checked,
           SUM(CASE WHEN covered = 1 THEN 1 ELSE 0 END) AS catalogues_covering
    FROM catalogue_coverage GROUP BY name
    """,
)

# Words a condition may contain that are not column names.
_SQL_WORDS = frozenset(
    """
    and or not null is in like glob between case when then else end escape collate
    nocase select distinct from where group by having order asc desc limit offset
    union all exists cast as on using join left right inner outer cross true false
    abs min max sum count avg round coalesce ifnull length lower upper substr
    """.split()
)

# An identifier, refusing anything preceded by a word character, a dot or a currency
# sign: the exponent of ``1e9`` and the tail of a qualified name are not names to check.
_IDENTIFIER = re.compile(r"(?<![\w.$])([A-Za-z_][A-Za-z_0-9]*)")
_STRING_LITERAL = re.compile(r"'[^']*'")


def _utc_now() -> str:
    """Current UTC instant, to the second."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _error(kind: str, name: str, known: Iterable[str], error: type) -> UnknownName:
    """Build the near-neighbour error for an unrecognised name."""
    options = sorted(str(item) for item in known)
    close = difflib.get_close_matches(str(name), options, n=4, cutoff=0.4)
    if close:
        hint = "did you mean " + ", ".join(repr(item) for item in close) + "?"
    elif len(options) <= 16:
        hint = "known: " + ", ".join(options)
    else:
        hint = f"known: {', '.join(options[:16])} ... ({len(options)} in all)"
    return error(f"unknown {kind} {str(name)!r}; {hint}")


def _identifiers(expression: str) -> Iterable[str]:
    """Yield the bare identifiers of a SQL expression, ignoring literals and calls."""
    text = _STRING_LITERAL.sub("''", str(expression))
    for match in _IDENTIFIER.finditer(text):
        tail = text[match.end() :].lstrip()
        if tail.startswith("("):
            continue
        yield match.group(1)


def _encode(value: Any) -> Any:
    """
    Render a Python value as something SQLite can store.

    Numpy scalars, paths and containers all arrive from the stages that produce them;
    the alternative to converting here is an adapter registration that applies process
    wide and surprises anything else using sqlite3.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (str, bytes, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _encode(value.item())
    if isinstance(value, (Mapping, list, tuple, set, np.ndarray)):
        return json.dumps(_jsonable(value), sort_keys=True, default=str)
    return str(value)


def _jsonable(value: Any) -> Any:
    """Recursively convert a container into JSON-serialisable form."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _unnull(value: Any) -> Any:
    """
    Read a data frame's stand-ins for a null back as ``None``.

    A frame has no null for a numeric column and uses NaN, and no null for a nullable
    integer and uses ``pd.NA``. Both are values, not absences: ``x is None`` is False for
    them and ``x == x`` raises or is False. Left alone, whether an unrecorded quantity
    reads as missing would depend on the dtype pandas inferred, which depends on the
    other candidates in the store.
    """
    return None if value is None or _is_na(value) else value


def _optional_float(value):
    """A float, or ``None`` for an absent one -- never a nan."""
    if value is None or _is_na(value):
        return None
    return float(value)


def _columnar_of(product) -> Dict[str, Any]:
    """
    A product's columns, whatever shape it arrived in.

    Stage products are variously a table object carrying a ``columns`` mapping, a plain
    mapping of arrays, a sequence of row mappings, or a data frame. Adapters should not
    each learn all four.
    """
    columns = getattr(product, "columns", None)
    if isinstance(columns, Mapping):
        return dict(columns)
    if isinstance(product, Mapping):
        return dict(product)
    if hasattr(product, "to_dict") and hasattr(product, "columns"):
        return {str(name): product[name].tolist() for name in product.columns}
    return {}


def _table_rows(product, wanted: Sequence[str], aliases: Optional[Mapping[str, str]] = None):
    """
    Rows for one table, taking the columns it holds and leaving the rest.

    Filtering rather than refusing a surplus column is what lets a product and the store
    evolve apart: a candidate table gains a column for a new convention long before
    anyone decides it belongs here. A *missing* column is left absent rather than filled
    with a null, so a stage that has not run reads as unrecorded rather than as measured
    and empty.
    """
    columns = _columnar_of(product)
    if not columns:
        return []
    aliases = dict(aliases or {})
    taken = {}
    for name in wanted:
        source = aliases.get(name, name)
        if source in columns:
            taken[name] = columns[source]
    if not taken:
        return []
    length = max(len(list(values)) for values in taken.values())
    rows = []
    for index in range(length):
        row = {}
        for name, values in taken.items():
            value = list(values)[index]
            if not _is_na(value):
                row[name] = value
        if row:
            rows.append(row)
    return rows


def _rows_of(product, wanted: Sequence[str]):
    """Rows from a sequence of mappings or a columnar product, keeping known columns."""
    if isinstance(product, Sequence) and not isinstance(product, (str, bytes)):
        return [
            {k: v for k, v in dict(row).items() if k in set(wanted) and not _is_na(v)}
            for row in product
        ]
    return _table_rows(product, wanted)


def _long_rows(reports, label: str, wanted: Sequence[str]):
    """
    One row per (candidate, ``label``), from the nested form these stages report in.

    Accepts ``{name: {label_value: {field: value}}}`` as well as a flat sequence of rows,
    because a per-candidate stage naturally accumulates the first and a collated one the
    second.
    """
    if reports is None:
        return []
    if isinstance(reports, Mapping):
        rows = []
        for name, entries in reports.items():
            for value, fields in dict(entries).items():
                row = {"name": str(name), label: str(value)}
                row.update(
                    {
                        k: v
                        for k, v in dict(fields).items()
                        if k in set(wanted) and not _is_na(v)
                    }
                )
                rows.append(row)
        return rows
    return _rows_of(reports, wanted)


def _is_na(value: Any) -> bool:
    """Whether a scalar is one of pandas' missing markers, without importing pandas."""
    if isinstance(value, (str, bytes)) or isinstance(value, (list, tuple, dict)):
        return False
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _is_columnar(value: Any) -> bool:
    """Whether a mapping value looks like a column rather than a scalar."""
    return isinstance(value, (list, tuple, np.ndarray)) and not isinstance(
        value, (str, bytes)
    )


def _as_rows(rows: Any) -> List[Dict[str, Any]]:
    """
    Normalise the accepted row containers into a list of mappings.

    Stages hold their results as columnar dicts, record lists or data frames depending on
    what produced them, and converting at the boundary keeps that choice theirs.
    """
    if rows is None:
        return []
    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        return [dict(record) for record in rows.to_dict("records")]
    if isinstance(rows, Mapping):
        values = list(rows.values())
        if not values:
            return []
        columnar = [key for key, value in rows.items() if _is_columnar(value)]
        if columnar:
            # A mapping with any column in it is a columnar write. Scalars alongside
            # are broadcast across it -- ``{"name": [...], "arm": "HL"}`` means what it
            # looks like. Falling through to the single-row branch instead wrote one row
            # whose natural key was a JSON-encoded list, putting a corrupt primary key
            # into a declared column without a word of complaint.
            lengths = {len(rows[key]) for key in columnar}
            if len(lengths) != 1:
                sizes = {key: len(rows[key]) for key in columnar}
                raise ValueError(f"columns of unequal length: {sizes}")
            count = lengths.pop()
            return [
                {
                    key: (value[index] if key in set(columnar) else value)
                    for key, value in rows.items()
                }
                for index in range(count)
            ]
        return [dict(rows)]
    return [dict(row) for row in rows]


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
        lines = [f"{self.name}  gps={self.gps}"]
        recorded = {
            key: value for key, value in self.fields.items() if value is not None
        }
        for key in sorted(recorded):
            if key in ("name", "gps"):
                continue
            lines.append(f"  {key:<24} {recorded[key]}")
        for label, block in (
            ("data quality", self.dataquality),
            ("consistency", self.consistency),
            ("parameters", self.parameters),
            ("catalogue", self.catalogue),
        ):
            if block:
                lines.append(f"  {label}: {', '.join(sorted(block))}")
        if self.artefacts:
            lines.append(f"  artefacts: {', '.join(sorted(self.artefacts))}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Flat mapping of every recorded quantity."""
        flat: Dict[str, Any] = {"name": self.name, "gps": self.gps}
        flat.update(self.fields)
        blocks = (
            ("dataquality", self.dataquality),
            ("consistency", self.consistency),
            ("parameters", self.parameters),
            ("catalogue", self.catalogue),
        )
        for label, block in blocks:
            for entry, values in block.items():
                if isinstance(values, Mapping):
                    for key, value in values.items():
                        flat[f"{label}.{entry}.{key}"] = value
                else:
                    flat[f"{label}.{entry}"] = values
        for kind, path in self.artefacts.items():
            flat[f"artefacts.{kind}"] = str(path)
        return flat

    def to_markdown(self) -> str:
        """Readable table form, for a report or a message."""
        flat = self.to_dict()
        rows = [(key, flat[key]) for key in flat if flat[key] is not None]
        return _markdown_table(("quantity", "value"), rows)

    def artefact(self, kind: str) -> Path:
        """
        Resolve a stored artefact, such as a spectrogram or posterior file.

        Refuses a path that has since disappeared instead of returning it. The store
        records locations rather than contents, so a moved or purged scratch directory
        is the ordinary failure, and it has to name the path it expected.
        """
        if kind not in self.artefacts:
            raise _error("artefact kind", kind, self.artefacts, UnknownArtefact)
        path = Path(self.artefacts[kind])
        if not path.exists():
            raise FileNotFoundError(
                f"the {kind!r} artefact recorded for {self.name} is not at {path}; the "
                "store records where bulk data lives, so the file has moved or been "
                "removed since it was recorded"
            )
        return path


def _markdown_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a markdown table without a third-party formatter."""
    body = [[("" if value is None else str(value)) for value in row] for row in rows]
    widths = [len(str(name)) for name in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |",
    ]
    for row in body:
        lines.append(
            "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        )
    return "\n".join(lines) + "\n"


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
        """
        Connect to a store file, creating it and its schema unless read-only.

        A read-only store is never created: an empty one answers every question with
        nothing, which reads as a campaign that produced nothing rather than as a
        mistyped path.
        """
        self.path = Path(path)
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"no campaign store at {self.path}; a read-only store is never "
                    "created, since an empty one answers every question with nothing"
                )
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA foreign_keys = OFF")
        if self.read_only:
            # Checked here too, not only in initialise(): read-only is the mode used to
            # inspect an old campaign, which is exactly the case a schema change makes
            # unreadable. Leaving the check to the writable path meant the one mode that
            # meets an old store was the one that half-read it, failing later as a bare
            # sqlite3 error naming a column rather than the file.
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                self._conn.close()
                raise RuntimeError(
                    f"{self.path} reports schema version {version}, but this is version "
                    f"{SCHEMA_VERSION}; it was written by another version of Sage, or "
                    "is not a campaign store. Rebuild it from its products rather than "
                    "reading it under a schema it does not have"
                )
        else:
            # Stages write in whatever order the campaign runs them, so the schema is
            # created on open rather than left for the first writer to remember.
            self.initialise()

    @classmethod
    def open(cls, spec, read_only: bool = False) -> "SearchStore":
        """
        Open (creating if needed) the store for a campaign.

        The campaign row is refreshed on every writable open, so the file always states
        which spec it belongs to even if no stage has run yet.
        """
        if not hasattr(spec, "out_dir"):
            raise TypeError(
                f"expected a SearchSpec, got {type(spec).__name__}; open a store by "
                "path with SearchStore(path)"
            )
        store = cls(Path(spec.out_dir) / "store.sqlite", read_only=read_only)
        if not read_only:
            store.put(
                "campaign",
                {
                    "tag": getattr(spec, "tag", "") or Path(spec.out_dir).name,
                    "spec_hash": spec.hash() if hasattr(spec, "hash") else None,
                    "config_module": getattr(spec, "config_module", None),
                    "observing_run": getattr(spec.data, "observing_run", None),
                    "detectors": ",".join(getattr(spec.data, "detectors", ()) or ()),
                    "out_dir": str(spec.out_dir),
                    "created_utc": _utc_now(),
                },
            )
        return store

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def __enter__(self) -> "SearchStore":
        """Context-manager entry."""
        return self

    def __exit__(self, *exc) -> None:
        """Context-manager exit."""
        self.close()

    # -- schema -------------------------------------------------------------

    def initialise(self) -> None:
        """
        Create the schema and its indices.

        Idempotent, and safe on a store that already holds rows. No foreign keys are
        declared: stages complete in an order the campaign chooses, and a per-candidate
        row must be writable before or after the candidate list it belongs to.
        """
        self._require_writable()
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise RuntimeError(
                f"{self.path} was written with schema version {version}, but this is "
                f"version {SCHEMA_VERSION}; rebuild the store from its products rather "
                "than mixing schemas"
            )
        with self._conn:
            for table in TABLES:
                spec = SCHEMA[table]
                columns = list(spec.columns)
                if table != "provenance":
                    declared = set(spec.column_names())
                    columns += [c for c in BOOKKEEPING if c[0] not in declared]
                body = ",\n    ".join(f'"{name}" {kind}' for name, kind in columns)
                key = ", ".join(f'"{name}"' for name in spec.key)
                self._conn.execute(
                    f'CREATE TABLE IF NOT EXISTS "{table}" (\n    {body},\n'
                    f"    PRIMARY KEY ({key})\n)"
                )
            for table, columns in INDICES:
                name = f"ix_{table}_{'_'.join(columns)}"
                target = ", ".join(f'"{column}"' for column in columns)
                self._conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ({target})'
                )
            self._conn.execute(f'DROP VIEW IF EXISTS "{VIEW_NAME}"')
            self._conn.execute(
                f'CREATE VIEW "{VIEW_NAME}" AS {self._view_body()}'
            )
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def schema(self, table: Optional[str] = None) -> str:
        """Show the schema, for one table or all of them."""
        if table is not None:
            self._spec(table)
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()
            return "" if row is None or row[0] is None else f"{row[0]};"
        rows = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
        ).fetchall()
        return "\n\n".join(f"{row[0]};" for row in rows)

    def describe(self) -> str:
        """Readable overview: tables, row counts and what each holds."""
        counts = {table: self.count(table) for table in TABLES}
        populated = sum(1 for value in counts.values() if value)
        width = max(len(table) for table in TABLES)
        lines = [
            f"SearchStore: {self.path}",
            f"schema version {SCHEMA_VERSION}, {len(TABLES)} tables, "
            f"{populated} populated",
            "",
        ]
        for table in TABLES:
            lines.append(
                f"  {table:<{width}}  {counts[table]:>8d} rows  {SCHEMA[table].note}"
            )
        return "\n".join(lines)

    def table_columns(self, table: str) -> Tuple[str, ...]:
        """
        A table's declared columns, from the schema rather than from the open file.

        Read by the stage adapters to decide what of a product belongs in the store. From
        the schema, so it answers before the file exists and does not depend on which
        stages have run.
        """
        return tuple(name for name, _ in self._spec(table).columns)

    def columns(self, table: Optional[str] = None) -> Dict[str, Tuple[str, ...]]:
        """Available column names, for discovering what can be queried."""
        if table is not None:
            return {table: self._columns_of(table)}
        names = {name: self._columns_of(name) for name in TABLES}
        names[VIEW_NAME] = self._columns_of(VIEW_NAME)
        return names

    def count(self, table: str) -> int:
        """Number of rows in a table."""
        self._spec(table)
        return int(self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

    # -- writing ------------------------------------------------------------

    def put(self, table: str, rows, key: Optional[Sequence[str]] = None) -> int:
        """
        Insert rows, merging into any that already carry the same natural key.

        Merging rather than appending is what makes a stage re-runnable: a stage that is
        run twice, or resumed after an interruption, must leave the store holding one row
        per candidate and not two, or every count and every rate downstream is doubled.

        Merging rather than *replacing* is what lets stages share a table. Only the
        columns a write actually names are assigned; everything else on an existing row
        is left as it stands. Several stages fill one row in turn -- significance writes
        the rate, vetting writes the tier -- and a replacing write would blank whichever
        columns it did not happen to carry. A tier is the sharp case: it is demoted by
        vetting and never promoted, so restoring its ``TIER_UNDETERMINED`` default would
        undo a judgement rather than merely lose a value.

        Accepts a sequence of mappings, a columnar mapping of equal-length arrays, a
        single mapping, or a data frame. Columns absent from a row are left null on a new
        row and untouched on an existing one, so a stage may fill in what it knows and
        leave the rest to another.

        Parameters
        ----------
        key : sequence of str, optional
            Replace whole groups sharing these columns instead of merging on the natural
            key: every stored row matching a written row's values is deleted first, so
            this is the way to re-run a stage whose output has a different number of rows
            than it did before. It clears rather than merges, by construction, and the
            natural key still has to be unique in the result.

        Returns
        -------
        int
            Number of rows written.
        """
        self._require_writable()
        spec = self._spec(table)
        records = _as_rows(rows)
        if not records:
            return 0

        known = self._columns_of(table)
        for record in records:
            for column in record:
                if column not in known:
                    raise _error(f"column of {table}", column, known, UnknownColumn)
        for column in spec.key:
            # `_unnull` and not `is None`: a data frame is one of the documented input
            # forms and carries a missing key as NaN, which slipped this guard and
            # surfaced as a raw IntegrityError from sqlite instead of the explanation.
            missing = [
                index
                for index, r in enumerate(records)
                if _unnull(r.get(column)) is None
            ]
            if missing:
                raise ValueError(
                    f"{table} rows {missing[:5]} do not set {column!r}, part of the "
                    f"natural key {spec.key}; a row without its key cannot be replaced "
                    "on a re-run and would accumulate"
                )
        seen: Dict[Tuple, int] = {}
        for index, record in enumerate(records):
            identity = tuple(_encode(record[column]) for column in spec.key)
            if identity in seen:
                raise ValueError(
                    f"rows {seen[identity]} and {index} of this write share the {table} "
                    f"key {dict(zip(spec.key, identity))}; one of them would be lost"
                )
            seen[identity] = index

        self._stamp(table, records)
        if key is not None:
            for column in key:
                if column not in known:
                    raise _error(f"column of {table}", column, known, UnknownColumn)

        with self._conn:
            if key is not None:
                self._replace_group(table, tuple(key), known, records)
            self._merge(table, spec.key, known, records)
        return len(records)

    def _replace_group(
        self,
        table: str,
        columns: Tuple[str, ...],
        known: Tuple[str, ...],
        records: List[Dict[str, Any]],
    ) -> None:
        """
        Delete every stored row sharing a written row's values in ``columns``.

        The clearing half of a widened write. Rows the write goes on to insert come back;
        rows of the same group that this write no longer produces do not, which is the
        point -- a re-fitted stage emitting fewer rows must not leave the surplus behind.
        """
        condition = " AND ".join(f'"{name}" IS ?' for name in columns)
        self._conn.executemany(
            f'DELETE FROM "{table}" WHERE {condition}',
            [tuple(_encode(record.get(name)) for name in columns) for record in records],
        )

    def _merge(
        self,
        table: str,
        key_columns: Tuple[str, ...],
        known: Tuple[str, ...],
        records: List[Dict[str, Any]],
    ) -> None:
        """
        Insert rows, assigning only the columns each one names.

        Rows are grouped by the exact set of columns they carry, so a write in which one
        row omits a column another sets does not assign that column null on the first.
        Each group is then one ``executemany``; the usual case is a single group.
        """
        groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        for record in records:
            present = tuple(name for name in known if name in record)
            groups.setdefault(present, []).append(record)

        conflict = ", ".join(f'"{name}"' for name in key_columns)
        for columns, group in groups.items():
            target = ", ".join(f'"{name}"' for name in columns)
            placeholders = ", ".join("?" for _ in columns)
            assigned = [name for name in columns if name not in key_columns]
            action = (
                "DO UPDATE SET "
                + ", ".join(f'"{name}" = excluded."{name}"' for name in assigned)
                if assigned
                else "DO NOTHING"
            )
            self._conn.executemany(
                f'INSERT INTO "{table}" ({target}) VALUES ({placeholders}) '
                f"ON CONFLICT ({conflict}) {action}",
                [tuple(_encode(record[name]) for name in columns) for record in group],
            )

    def put_events(self, candidates) -> int:
        """
        Record the candidate table.

        The columns are taken as the candidate table declares them, filtered to those the
        ``events`` table holds. Filtering rather than failing on a surplus column is what
        lets the two evolve apart: the candidate table gains a column for a new
        significance convention long before anyone decides it belongs in the store.
        """
        rows = _table_rows(candidates, self.table_columns("events"))
        return self.put("events", rows)

    def put_significance(self, candidates, far_curve) -> int:
        """
        Record false-alarm rate, inverse rate and p-value per candidate.

        The per-candidate rates come from the candidate table, which is where they were
        assigned; the curve supplies the livetimes they were measured against. Those
        livetimes are the denominator of every rate here, so they are stored beside the
        rates rather than left to be looked up -- a rate whose exposure has to be found
        elsewhere is a rate that will be quoted against the wrong one.
        """
        rows = _table_rows(candidates, self.table_columns("significance"))
        written = self.put("significance", rows)
        if far_curve is not None:
            self.put_provenance(
                "significance",
                {
                    "background_livetime_s": float(far_curve.background_livetime_s),
                    "foreground_livetime_s": float(far_curve.foreground_livetime_s),
                    "removal": str(far_curve.removal),
                    "ifar_cap_yr": float(far_curve.ifar_cap_yr),
                },
            )
        return written

    def put_pastro(self, table) -> int:
        """
        Record astrophysical probability and its credible interval.

        Named ``pastro`` in the store against ``p_astro`` in the candidate table: the
        store's column names are what a condition is written in at a prompt, and the
        underscore was a recurring mistype. Translated here rather than renamed there,
        because the candidate table's names follow the papers.
        """
        wanted = self.table_columns("pastro")
        rows = _table_rows(
            table,
            wanted,
            aliases={"pastro": "p_astro", "pastro_lo": "p_astro_lo",
                     "pastro_hi": "p_astro_hi"},
        )
        return self.put("pastro", rows)

    def put_dataquality(self, reports) -> int:
        """
        Record each data-quality task's outcome per candidate.

        One row per candidate and task, keyed on both. Written with ``key=["name"]`` so a
        re-run that drops a task leaves no orphan row behind: the set of tasks is part of
        what a re-run changes, and merging cannot remove.
        """
        rows = _long_rows(reports, "task", self.table_columns("dataquality"))
        return self.put("dataquality", rows, key=["name"]) if rows else 0

    def put_consistency(self, results) -> int:
        """
        Record each consistency test's outcome per candidate.

        As :meth:`put_dataquality`, keyed on the test rather than the task.
        """
        rows = _long_rows(results, "test", self.table_columns("consistency"))
        return self.put("consistency", rows, key=["name"]) if rows else 0

    def put_parameters(self, pe_results, level: float = 0.9) -> int:
        """
        Record parameter medians and credible intervals, with their provenance.

        ``level`` is stored on the row rather than assumed by a reader: a median with an
        interval whose credibility is not stated is a number nobody can quote.

        The waveform is likewise carried per row. Estimates from two waveform families
        sit in the same table and differ systematically, and a table that cannot say
        which produced a row invites them to be averaged.
        """
        rows = _long_rows(pe_results, "parameter", self.table_columns("parameters"))
        for row in rows:
            row.setdefault("level", float(level))
        return self.put("parameters", rows, key=["name"]) if rows else 0

    def put_catalogue(self, catalogues, matches, coverage) -> int:
        """
        Record external events, their matches and each catalogue's coverage.

        Three tables, because they answer three different questions and only the first
        two are about candidates. ``catalogue_coverage`` is what separates "this
        catalogue looked here and found nothing" from "this catalogue never looked", and
        without it a comparison presents the second as the first.
        """
        written = 0
        events = []
        for key, catalogue in dict(catalogues or {}).items():
            for event in getattr(catalogue, "events", ()) or ():
                events.append(
                    {
                        "catalogue": str(key),
                        "event_name": str(event.name),
                        "gps": float(event.gps),
                        "far_per_yr": _optional_float(event.far_per_yr),
                        "pastro": _optional_float(event.p_astro),
                    }
                )
        if events:
            written += self.put("catalogue_events", events, key=["catalogue"])
        if matches:
            written += self.put(
                "catalogue_matches",
                _rows_of(matches, self.table_columns("catalogue_matches")),
                key=["name"],
            )
        if coverage:
            written += self.put(
                "catalogue_coverage",
                _rows_of(coverage, self.table_columns("catalogue_coverage")),
                key=["catalogue"],
            )
        return written

    def put_sensitivity(self, results) -> int:
        """
        Record sensitivity at each threshold and reference point.

        One row per (threshold, reference point) rather than a single summary number: a
        sensitive volume quoted without the false-alarm rate it was measured at cannot be
        compared with anyone else's.
        """
        rows = _rows_of(results, self.table_columns("sensitivity"))
        return self.put("sensitivity", rows, key=["arm"]) if rows else 0

    def put_artefact(self, name: str, kind: str, path: str | Path, **attrs) -> None:
        """
        Record where a bulk artefact for a candidate is stored.

        The file itself is not read: recording is cheap and happens as a stage writes,
        which may be before the file is closed. Existence is checked when the path is
        resolved, by :meth:`EventRecord.artefact`.
        """
        target = Path(path)
        size = target.stat().st_size if target.is_file() else None
        self.put(
            "artefacts",
            {
                "name": name,
                "kind": kind,
                "path": str(target),
                "bytes": size,
                "sha256": attrs.pop("sha256", None),
                "attrs": dict(attrs) if attrs else None,
            },
        )

    def put_provenance(self, stage: str, attrs: Mapping[str, Any]) -> None:
        """
        Record how a stage's rows were produced.

        Rows already written by this stage are stamped with the spec hash recorded here,
        so provenance may be written before or after the rows it describes and a row is
        traceable either way.
        """
        payload = dict(attrs or {})
        row = {
            "stage": stage,
            "spec_hash": payload.get("spec_hash"),
            "sage_version": payload.get("sage_version"),
            "git_hash": payload.get("git_hash"),
            "config_module": payload.get("config_module"),
            "created_utc": payload.get("created_utc") or _utc_now(),
            "attrs": payload,
        }
        self.put("provenance", row)
        spec_hash = row["spec_hash"]
        if spec_hash is None:
            return
        with self._conn:
            for table in TABLES:
                if table == "provenance":
                    continue
                # Every row of the stage, not only the unstamped ones. A stage owns its
                # rows outright, so when it is re-run under a different configuration the
                # rows it just rewrote belong to the new one. Restricting the backfill to
                # NULLs left those rows claiming the previous hash while the provenance
                # table moved on -- two answers for one stage, and the stale one on the
                # row a reader would trust.
                self._conn.execute(
                    f'UPDATE "{table}" SET spec_hash = ? WHERE stage = ?',
                    (_encode(spec_hash), stage),
                )

    # -- reading ------------------------------------------------------------

    def event(self, name: str) -> EventRecord:
        """Everything recorded about one candidate."""
        return self.events(names=[name])[0]

    def events(
        self, names: Optional[Sequence[str]] = None, where: Optional[str] = None
    ) -> List[EventRecord]:
        """
        Full records for a set of candidates.

        Every stage that has written something contributes; the stages that have not are
        simply absent from the record rather than an error, so a store part-way through a
        campaign answers the same questions as a finished one.

        A name the store does not hold is refused, while a name it holds that the
        condition excludes is merely absent: the two are different mistakes and only the
        first is worth interrupting for.
        """
        frame = self.select(where=where)
        rows = frame.to_dict("records")
        if names is not None:
            wanted = [str(item) for item in names]
            recorded = {
                row[0] for row in self._conn.execute("SELECT name FROM events")
            }
            missing = [item for item in wanted if item not in recorded]
            if missing:
                raise _error("candidate", missing[0], recorded, UnknownEvent)
            by_name = {str(row["name"]): row for row in rows}
            rows = [by_name[item] for item in wanted if item in by_name]
        if not rows:
            return []

        # The rows arrive through a data frame, which has no null for a numeric column
        # and substitutes NaN. Whether a quantity this candidate lacks reads as None
        # would then depend on whether some *other* candidate has it -- with one
        # candidate the column stays object-typed and the null survives, with two it
        # becomes float and turns into NaN. Every `is None` filter downstream, including
        # the ones in EventRecord.summary and to_markdown, is silently dead in that
        # state and the summary fills with `nan` lines.
        rows = [{key: _unnull(value) for key, value in row.items()} for row in rows]

        selected = [str(row["name"]) for row in rows]
        blocks = {
            "dataquality": self._gather("dataquality", "task", selected),
            "consistency": self._gather("consistency", "test", selected),
            "parameters": self._gather("parameters", "parameter", selected),
            "catalogue": self._gather("catalogue_matches", "catalogue", selected),
        }
        artefacts = self._gather("artefacts", "kind", selected)
        out = []
        for row in rows:
            name = str(row["name"])
            out.append(
                EventRecord(
                    name=name,
                    gps=row.get("gps"),
                    fields={k: v for k, v in row.items() if k != "name"},
                    dataquality=blocks["dataquality"].get(name, {}),
                    consistency=blocks["consistency"].get(name, {}),
                    parameters=blocks["parameters"].get(name, {}),
                    catalogue=blocks["catalogue"].get(name, {}),
                    artefacts={
                        kind: Path(entry["path"])
                        for kind, entry in artefacts.get(name, {}).items()
                        if entry.get("path")
                    },
                )
            )
        return out

    def at_gps(self, gps: float, tolerance_s: float = 1.0) -> List[EventRecord]:
        """
        Candidates near a time, for cross-checking an externally reported event.

        Time, not name: the same event carries second-level differences in its name
        between catalogues, so a name lookup both misses real associations and invents
        false ones.
        """
        rows = self._conn.execute(
            f'SELECT name FROM "{VIEW_NAME}" WHERE ABS(gps - ?) <= ? '
            "ORDER BY ABS(gps - ?)",
            (float(gps), float(tolerance_s), float(gps)),
        ).fetchall()
        return self.events(names=[row[0] for row in rows])

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

        A stage that has not run contributes nulls rather than removing the candidate,
        so a condition naming it selects nothing while conditions over the stages that
        did run keep working.
        """
        available = self._columns_of(VIEW_NAME)
        lowered = {name.lower() for name in available}
        for expression in (where, order_by):
            if not expression:
                continue
            for token in _identifiers(expression):
                if token.lower() in _SQL_WORDS or token.lower() in lowered:
                    continue
                raise _error("column", token, available, UnknownColumn)
        if columns:
            for column in columns:
                if column.lower() not in lowered:
                    raise _error("column", column, available, UnknownColumn)
            target = ", ".join(f'"{column}"' for column in columns)
        else:
            target = "*"

        sql = f'SELECT {target} FROM "{VIEW_NAME}"'
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql)

    def query(self, sql: str, params: Optional[Sequence] = None):
        """Run arbitrary SQL and return the result as a data frame."""
        import pandas as pd

        try:
            cursor = self._conn.execute(str(sql), tuple(params or ()))
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc
        names = [column[0] for column in (cursor.description or ())]
        return pd.DataFrame(cursor.fetchall(), columns=names)

    def table(self, name: str):
        """Read one table in full."""
        self._spec(name)
        return self.query(f'SELECT * FROM "{name}"')

    def joined(self):
        """The per-candidate view that :meth:`select` filters over."""
        return self.select()

    # -- comparison and export ---------------------------------------------

    def compare(self, names: Sequence[str], columns: Optional[Sequence[str]] = None):
        """
        Place several candidates side by side on the same quantities.

        Candidates transposed against quantities, which is the orientation a handful of
        candidates is actually read in -- a column each, so the eye runs down a row to
        compare one quantity. An unknown name is refused rather than dropped: a candidate
        silently missing from a comparison reads as one that did not stand out.
        """
        records = [self.event(name) for name in names]
        if columns is None:
            seen: List[str] = []
            for record in records:
                for field, value in record.fields.items():
                    if value is not None and field not in seen:
                        seen.append(field)
            columns = seen
        columns = [c for c in columns if c != "name"]

        import pandas as pd

        frame = pd.DataFrame(
            {
                record.name: [record.fields.get(column) for column in columns]
                for record in records
            },
            index=list(columns),
        )
        frame.index.name = "quantity"
        return frame

    def comparison_matrix(self):
        """
        Candidates against catalogues.

        Distinguishes found, searched but not found, and outside a catalogue's searched
        region, since the last says nothing about the candidate.

        Cells are ``"found"``, ``"not found"`` and ``"not searched"`` rather than 1/0/nan.
        A matrix read by eye is the one place the three-way distinction is most easily
        lost: a blank cell reads as a zero, and a zero asserts a non-detection on the
        strength of a catalogue that never looked. Naming the third state makes that
        impossible to misread and impossible to average.

        Coverage is per candidate rather than per time span, because a catalogue's reach
        is not only temporal -- a search covering the same days over a narrower chirp-mass
        range has not searched where a heavy candidate lies. The reason is carried on the
        coverage row for exactly that case.
        """
        import pandas as pd

        events = self.table("events")
        coverage = self.table("catalogue_coverage")
        matches = self.table("catalogue_matches")
        catalogues = sorted(
            set(coverage["catalogue"].tolist() if len(coverage) else [])
            | set(matches["catalogue"].tolist() if len(matches) else [])
        )
        index = pd.Index(events["name"].tolist() if len(events) else [], name="name")
        if not len(index) or not catalogues:
            return pd.DataFrame(index=index, columns=catalogues, dtype=object)

        matched = {
            (str(row["name"]), str(row["catalogue"]))
            for _, row in matches.iterrows()
            if row.get("matched")
        }
        # Absent from the coverage table means the question was never recorded, which is
        # not the same as recorded uncovered. Treated as searched, because the matches
        # table is then the only evidence there is -- and a campaign that records matches
        # without recording coverage is the ordinary case before the coverage stage runs.
        uncovered = {
            (str(row["name"]), str(row["catalogue"]))
            for _, row in coverage.iterrows()
            if not row.get("covered")
        }

        cells = {
            catalogue: [
                "not searched"
                if (name, catalogue) in uncovered
                else ("found" if (name, catalogue) in matched else "not found")
                for name in index
            ]
            for catalogue in catalogues
        }
        return pd.DataFrame(cells, index=index, dtype=object)

    def tier_counts(self):
        """
        How many candidates fall in each inclusion tier.

        Both views: the tier assigned from a single arm's significance and the one after
        the trials correction. Reporting only the corrected count would hide the
        candidates the correction moved, which is the set worth looking at.
        """
        import pandas as pd

        frame = self.table("events")
        if not len(frame):
            return pd.DataFrame(columns=["tier", "n", "n_trials_corrected"])
        plain = frame["tier"].value_counts()
        corrected = frame["tier_trials"].value_counts()
        tiers = sorted(set(plain.index) | set(corrected.index))
        return pd.DataFrame(
            {
                "tier": tiers,
                "n": [int(plain.get(tier, 0)) for tier in tiers],
                "n_trials_corrected": [int(corrected.get(tier, 0)) for tier in tiers],
            }
        )

    def export(self, path: str | Path, fmt: str = "csv", where: Optional[str] = None) -> Path:
        """Write a selection out as csv, markdown, latex or hdf5."""
        frame = self.select(where=where)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        kind = str(fmt).lower()
        if kind == "csv":
            frame.to_csv(target, index=False)
        elif kind in ("md", "markdown"):
            target.write_text(
                _markdown_table(list(frame.columns), frame.itertuples(index=False)),
                encoding="utf-8",
            )
        elif kind in ("tex", "latex"):
            target.write_text(frame.to_latex(index=False), encoding="utf-8")
        elif kind in ("h5", "hdf5"):
            _write_hdf5(target, frame)
        else:
            raise ValueError(
                f"unknown export format {fmt!r}; expected one of {EXPORT_FORMATS}"
            )
        return target

    def to_pandas(self, table: Optional[str] = None):
        """Whole table, or the joined view, as a data frame."""
        return self.joined() if table is None else self.table(table)

    # -- internals ----------------------------------------------------------

    def _require_writable(self) -> None:
        """Refuse a write on a store opened read-only, before SQLite does."""
        if self.read_only:
            raise PermissionError(
                f"the store at {self.path} was opened read-only; open it writable to "
                "record anything"
            )

    def _spec(self, table: str) -> TableSpec:
        """The specification of a table, refusing an unknown name with suggestions."""
        if table not in SCHEMA:
            raise _error("table", table, TABLES, UnknownTable)
        return SCHEMA[table]

    def _columns_of(self, table: str) -> Tuple[str, ...]:
        """Column names of a table or of the view, as SQLite reports them."""
        if table != VIEW_NAME:
            self._spec(table)
        rows = self._conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return tuple(row[1] for row in rows)

    def _view_body(self) -> str:
        """
        Build the joined per-candidate view from the schema.

        Generated rather than written out, so a column added to a stage's table is
        queryable through :meth:`select` without a second edit that can be forgotten.
        """
        bookkeeping = {name for name, _ in BOOKKEEPING}
        taken = {"name"}
        parts = ["e.name AS name"]
        joins = []
        for index, table in enumerate(("events",) + VIEW_JOINS):
            alias = "e" if table == "events" else f"t{index}"
            for column in SCHEMA[table].column_names():
                if column in taken or column in bookkeeping:
                    continue
                taken.add(column)
                parts.append(f'{alias}."{column}" AS "{column}"')
            if table != "events":
                joins.append(f'LEFT JOIN "{table}" {alias} ON {alias}.name = e.name')
        for index, aggregate in enumerate(VIEW_AGGREGATES):
            alias = f"a{index}"
            joins.append(f"LEFT JOIN ({aggregate.strip()}) {alias} ON {alias}.name = e.name")
            for column in _aggregate_columns(aggregate):
                if column in taken:
                    continue
                taken.add(column)
                parts.append(f'{alias}."{column}" AS "{column}"')
        select = ",\n       ".join(parts)
        return f'SELECT {select}\nFROM "events" e\n' + "\n".join(joins)

    def _stamp(self, table: str, records: List[Dict[str, Any]]) -> None:
        """Attach the writing stage, its spec hash and the time to each row."""
        if table == "provenance":
            return
        now = _utc_now()
        hashes: Dict[str, Any] = {}
        for record in records:
            stage = record.get("stage") or table
            if stage not in hashes:
                row = self._conn.execute(
                    "SELECT spec_hash FROM provenance WHERE stage = ?", (stage,)
                ).fetchone()
                hashes[stage] = None if row is None else row[0]
            record["stage"] = stage
            if record.get("spec_hash") is None:
                record["spec_hash"] = hashes[stage]
            record["written_utc"] = now

    def _gather(
        self, table: str, entry: str, names: Sequence[str]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Read a one-to-many per-candidate table, grouped by candidate and entry."""
        if not names:
            return {}
        placeholders = ", ".join("?" for _ in names)
        cursor = self._conn.execute(
            f'SELECT * FROM "{table}" WHERE name IN ({placeholders})', tuple(names)
        )
        columns = [column[0] for column in cursor.description]
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            out.setdefault(str(record["name"]), {})[str(record[entry])] = record
        return out

    def _translate(self, exc: sqlite3.OperationalError) -> Exception:
        """Turn SQLite's bare 'no such column' into the near-neighbour error."""
        message = str(exc)
        match = re.search(r"no such column:\s*([\w.\"]+)", message)
        if match:
            name = match.group(1).strip('"').split(".")[-1]
            known = set(self._columns_of(VIEW_NAME))
            for table in TABLES:
                known.update(self._columns_of(table))
            return _error("column", name, known, UnknownColumn)
        match = re.search(r"no such table:\s*([\w.\"]+)", message)
        if match:
            name = match.group(1).strip('"').split(".")[-1]
            return _error("table", name, TABLES, UnknownTable)
        return exc


def _aggregate_columns(sql: str) -> Tuple[str, ...]:
    """Output names of an aggregate subquery, taken from its ``AS`` labels."""
    return tuple(re.findall(r"\bAS\s+(\w+)", sql))


def _write_hdf5(path: Path, frame) -> None:
    """
    Write a selection as one dataset per column.

    Column-per-dataset rather than a compound table: an external reader can pull one
    quantity out without knowing the rest of the schema, and a null in a text column
    does not force a dtype on the others.
    """
    import h5py

    with h5py.File(path, "w") as handle:
        for column in frame.columns:
            values = frame[column].to_numpy()
            try:
                handle.create_dataset(column, data=values.astype(float))
            except (TypeError, ValueError):
                text = ["" if value is None else str(value) for value in values]
                handle.create_dataset(
                    column, data=text, dtype=h5py.string_dtype(encoding="utf-8")
                )


def open_store(spec, read_only: bool = False) -> SearchStore:
    """
    Open the campaign store for a spec.

    Delegates to :meth:`SearchStore.open` rather than resolving a path of its own. Two
    entry points naming two files would give a campaign two stores, and the one a reader
    opened would depend on which name they had happened to see.
    """
    return SearchStore.open(spec, read_only=read_only)


def build_from_products(spec, overwrite: bool = False) -> SearchStore:
    """
    Populate a store from the products a campaign has already written.

    Lets a store be rebuilt from stage outputs without re-running the analysis, which is
    what makes the store safe to change: its schema is a presentation of products that
    already exist on disk, so a schema change costs a rebuild rather than a campaign.

    Products absent from the campaign are skipped rather than refused. The store is
    queried while a campaign is running and most of it is empty most of the time; a
    rebuild that insisted on the whole chain would be unusable exactly when it is most
    wanted.
    """
    import json

    store = open_store(spec)
    if overwrite:
        # Rebuilt from products, so discarding is safe and is what `overwrite` asks for.
        store.close()
        Path(store.path).unlink(missing_ok=True)
        store = SearchStore.open(spec)
    store.initialise()

    store.put(
        "campaign",
        [
            {
                "tag": str(spec.tag),
                "spec_hash": spec.hash(),
                "config_module": str(spec.config_module),
                "observing_run": str(spec.data.observing_run),
            }
        ],
    )
    store.put(
        "arms",
        [{"arm": spec.arm, "detectors": ",".join(spec.data.detectors)}],
    )

    candidates_path = spec.path("candidates", "candidates.h5")
    if candidates_path.is_file():
        from sage.search.candidates import CandidateTable

        table = CandidateTable.load(candidates_path, allow_undetermined=True)
        store.put_events(table)
        store.put_significance(table, None)
        store.put_pastro(table)
        store.put_provenance(
            "events", {"spec_hash": spec.hash(), "source": str(candidates_path)}
        )

    livetime = spec.path("slides", "slide_plan.h5")
    if livetime.is_file():
        from sage.search.slides import SlidePlan

        plan = SlidePlan.load(livetime)
        store.put(
            "livetime",
            [
                {
                    "arm": spec.arm,
                    "observing_run": str(spec.data.observing_run),
                    "foreground_s": float(plan.foreground_livetime_s),
                    "background_s": float(plan.background_livetime_s),
                    "n_slides": sum(1 for s in plan.slides if s.slide_id != 0),
                }
            ],
            key=["arm"],
        )

    report = spec.path("catalogue", "comparison.json")
    if report.is_file():
        payload = json.loads(report.read_text())
        store.put_provenance("catalogue", {"sources": ",".join(payload.get("sources", []))})

    return store


def run(spec, **kwargs) -> dict:
    """
    Stage driver: build the campaign store from what the campaign has written.

    Reads products rather than recomputing anything, so re-running this stage cannot
    change a result -- it can only change how one is presented. That is the property that
    makes the schema safe to alter after a campaign has finished.
    """
    from sage.search.fingerprint import combine

    store = build_from_products(spec, overwrite=bool(kwargs.get("overwrite", False)))
    try:
        counts = {table: store.count(table) for table in TABLES}
        filled = {name: n for name, n in counts.items() if n}
        return {
            "store": str(store.path),
            "n_events": counts.get("events", 0),
            "tables_filled": len(filled),
            "rows": sum(counts.values()),
            "fingerprint": combine(
                counts.get("events", 0), sum(counts.values()), len(filled)
            ),
        }
    finally:
        store.close()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : triggers.py
Description   : The canonical trigger shard schema, writer and histogram algebra.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One schema serves zero-lag, background and injection runs; ``slide_id`` and ``inj_id``
distinguish them. Two histograms are stored per shard: ``counts_windows`` over every
analysed window and ``counts_clustered`` over cluster representatives. Only the
clustered counts are a valid background count; the FAR layer refuses the other.
Overflow and underflow are stored as separate scalars so histograms from any subset
add exactly, since a raw fp32 logit is unbounded.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

STAT_HIST_LO: float = -40.0
STAT_HIST_HI: float = 60.0
STAT_HIST_NBINS: int = 40000

TRIGGER_COLUMNS: Tuple[str, ...] = (
    "stat",
    "gps",
    "tc_gps",
    "tc_sigma",
    "mchirp",
    "mchirp_sigma",
    "segment_index",
    "local_start",
    "slide_id",
    "inj_id",
)

# On-disk dtype of every schema column. Declared here rather than taken from whatever the
# producer handed over, so shards written by different stages hold the same types and a
# merge cannot promote a column silently. A cast that would lose information -- a float
# time into an integer index -- is refused rather than performed.
COLUMN_DTYPES: Dict[str, type] = {
    "stat": np.float64,
    "gps": np.float64,
    "tc_gps": np.float64,
    "tc_sigma": np.float64,
    "mchirp": np.float64,
    "mchirp_sigma": np.float64,
    "segment_index": np.int64,
    "local_start": np.int64,
    "slide_id": np.int64,
    "inj_id": np.int64,
}

# Layout inside a shard.
#: Attributes two shards must agree on before their triggers may be put together.
#:
#: ``keep_threshold`` is here because it decides which windows became rows at all. Two
#: rungs of a ladder thresholded at different values hold different *fractions* of their
#: own tails, and concatenating them gives a background whose count belongs to no single
#: threshold -- while every column, every dtype and every provenance field still matches.
#: ``spec_hash`` does not cover it: the hash is deliberately blind to code, so a fix in
#: the decoder or the histogram leaves it identical.
COMPATIBILITY_KEYS: Tuple[str, ...] = (
    "spec_hash",
    "observing_run",
    "arm",
    "clustered",
    "keep_threshold",
)

_TRIGGER_GROUP = "triggers"
_HISTOGRAM_GROUP = "histogram"
_STREAM_GROUP = "stream"
_BLOCK_DATASET = "completed_blocks"


@dataclass
class TriggerTable:
    """An in-memory trigger set with its provenance."""

    columns: Dict[str, np.ndarray]
    attrs: Dict[str, object]

    def __post_init__(self) -> None:
        """Refuse a table whose columns are not one value per trigger."""
        self.columns = {name: np.asarray(v) for name, v in self.columns.items()}
        unknown = [name for name in self.columns if name not in TRIGGER_COLUMNS]
        if unknown:
            raise ValueError(
                f"columns {sorted(unknown)} are not in the shard schema "
                f"{TRIGGER_COLUMNS}; a column no reader knows about is written and then "
                "silently dropped by the next stage that copies the table"
            )
        lengths = {name: v.shape[0] for name, v in self.columns.items() if v.ndim}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"columns of unequal length: {lengths}")

    def __len__(self) -> int:
        """Number of triggers."""
        for values in self.columns.values():
            return int(values.shape[0])
        return 0

    def __getitem__(self, name: str) -> np.ndarray:
        """Column access."""
        if name not in self.columns:
            raise KeyError(
                f"{name!r} is not present; this table holds "
                f"{sorted(self.columns)}"
            )
        return self.columns[name]

    def sort_by(self, name: str) -> "TriggerTable":
        """
        Return a copy ordered by one column.

        Stable, so triggers equal in the sort column keep the order they were produced
        in. Clustering resolves ties toward the earlier trigger, and an unstable sort
        would make which one that is depend on the sort implementation.
        """
        order = np.argsort(self[name], kind="stable")
        return TriggerTable(
            columns={key: values[order] for key, values in self.columns.items()},
            attrs=dict(self.attrs),
        )

    def filter(self, mask: np.ndarray) -> "TriggerTable":
        """Return the subset selected by ``mask``."""
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise TypeError(
                f"mask must be boolean, got {mask.dtype}; an integer array would be "
                "read as fancy indexing and silently reorder or repeat triggers"
            )
        if mask.shape[0] != len(self):
            raise ValueError(
                f"mask selects over {mask.shape[0]} rows against {len(self)} triggers"
            )
        return TriggerTable(
            columns={key: values[mask] for key, values in self.columns.items()},
            attrs=dict(self.attrs),
        )

    def concat(self, other: "TriggerTable") -> "TriggerTable":
        """
        Append another table, checking schema and provenance compatibility.

        The provenance keys that decide what a trigger *means* must agree: two shards
        written under different specs, or for different observing runs or networks, are
        not one trigger set, and concatenating them produces a background whose events
        came from analyses that were never comparable.
        """
        if set(self.columns) != set(other.columns):
            raise ValueError(
                f"cannot concatenate tables holding different columns: "
                f"{sorted(self.columns)} against {sorted(other.columns)}"
            )
        for key in COMPATIBILITY_KEYS:
            mine, theirs = self.attrs.get(key), other.attrs.get(key)
            if mine != theirs:
                raise ValueError(
                    f"cannot concatenate shards whose {key} differs: {mine!r} against "
                    f"{theirs!r}; they describe different analyses"
                )
        merged = dict(self.attrs)
        for key, value in other.attrs.items():
            if merged.get(key) != value:
                merged.pop(key, None)
        return TriggerTable(
            columns={
                key: np.concatenate([self.columns[key], other.columns[key]])
                for key in self.columns
            },
            attrs=merged,
        )


@dataclass
class StatHistogram:
    """Exact ranking-statistic histogram on the fixed shared grid."""

    counts: np.ndarray
    underflow: int
    overflow: int
    clustered: bool

    def __post_init__(self) -> None:
        """Refuse a histogram that is not on the shared grid, or not in integers."""
        self.counts = np.asarray(self.counts)
        if self.counts.shape != (STAT_HIST_NBINS,):
            raise ValueError(
                f"histogram has {self.counts.shape} bins against the shared grid's "
                f"({STAT_HIST_NBINS},); histograms from different grids cannot be added"
            )
        if not np.issubdtype(self.counts.dtype, np.integer):
            raise TypeError(
                f"counts must be an integer dtype, got {self.counts.dtype}; a float "
                "histogram loses exactness once a campaign's counts exceed 2**53, and "
                "the whole point of storing counts is that adding them is exact"
            )
        self.underflow = int(self.underflow)
        self.overflow = int(self.overflow)
        self.clustered = bool(self.clustered)

    @property
    def total(self) -> int:
        """Every entry, including the two outside the grid."""
        return int(self.counts.sum()) + self.underflow + self.overflow

    def __add__(self, other: "StatHistogram") -> "StatHistogram":
        """
        Exact addition; requires the same grid and the same clustered flag.

        Adding a clustered histogram to an unclustered one produces a count that is
        neither, and the mixture is invisible afterwards -- the sum still looks like a
        histogram. The clustered flag travels with the counts so the two can never be
        added by accident.
        """
        if not isinstance(other, StatHistogram):
            return NotImplemented
        if self.clustered != other.clustered:
            raise ValueError(
                "refusing to add a clustered histogram to an unclustered one: the "
                "result would be counted as a background and is neither. Cluster first, "
                "or add the unclustered pair separately"
            )
        return StatHistogram(
            counts=self.counts + other.counts,
            underflow=self.underflow + other.underflow,
            overflow=self.overflow + other.overflow,
            clustered=self.clustered,
        )

    def n_above(self, stat: float) -> int:
        """
        Count of entries at or above ``stat``.

        Inclusive at ``stat``: a background event exactly as loud as the candidate is
        evidence that the noise reaches that value, so it counts toward the candidate's
        rate. Overflow always counts -- those entries are louder than any bin edge --
        and underflow never does unless the query is below the grid entirely.
        """
        edges = hist_edges()
        if stat > edges[-1]:
            # Everything on the grid is below the query; only the overflow can be above,
            # and it is unbounded above, so it counts.
            return self.overflow
        if stat <= edges[0]:
            return self.total
        # A bin holds [lo, hi). A query inside a bin cannot be resolved further, so the
        # whole bin counts -- an over-count of at most one bin, in the conservative
        # direction, which is the same direction as the (1 + n) FAR numerator.
        first = int(np.searchsorted(edges, stat, side="right")) - 1
        return int(self.counts[first:].sum()) + self.overflow

    def quantile_threshold(self, keep_rate: float) -> float:
        """
        Statistic value retaining a target fraction of entries.

        Used once per campaign to freeze the keep threshold from the complete zero-lag
        histogram. The value returned is a bin edge, so it is exactly representable and
        every slide job thresholds on the identical number.

        The rate is achieved from above: at least ``keep_rate`` of entries are at or
        above the returned value. Falling short would silently discard triggers the
        campaign was configured to keep.
        """
        if not 0.0 < keep_rate <= 1.0:
            raise ValueError(
                f"keep_rate must lie in (0, 1], got {keep_rate}"
            )
        total = self.total
        if total == 0:
            raise ValueError("an empty histogram has no quantile")
        edges = hist_edges()
        wanted = keep_rate * total
        # Survival at each left edge, plus the overflow which is above all of them.
        from_above = np.cumsum(self.counts[::-1])[::-1] + self.overflow
        usable = np.flatnonzero(from_above >= wanted)
        if usable.size == 0:
            return float(edges[-1])
        return float(edges[usable[-1]])


def hist_edges() -> np.ndarray:
    """
    The fixed shared bin edges; identical for every shard in a campaign.

    Fixed rather than derived from the data, because histograms from separate jobs are
    added: edges chosen per shard would put the same statistic in different bins in
    different shards, and the sum would be meaningless while still looking like a
    histogram. Returns ``STAT_HIST_NBINS + 1`` edges.
    """
    return np.linspace(STAT_HIST_LO, STAT_HIST_HI, STAT_HIST_NBINS + 1)


def histogram_stats(stat: np.ndarray, clustered: bool) -> StatHistogram:
    """
    Bin a statistic array onto the shared grid, keeping what falls outside it.

    A raw network output is unbounded, so values outside ``[STAT_HIST_LO, STAT_HIST_HI]``
    are ordinary rather than exceptional -- and the ones above are the loudest triggers
    in the campaign, exactly the ones a FAR is asked about. They are counted in the
    overflow instead of being clipped into the top bin, which would make them
    indistinguishable from merely loud ones, or dropped, which would remove them from the
    background entirely.

    NaN is refused. It is neither inside the grid nor outside it, silently vanishes from
    every comparison, and in a ranking statistic means the network produced nothing
    usable for that window -- which is a fault to report, not a value to bin.
    """
    stat = np.asarray(stat, dtype=np.float64).ravel()
    if np.isnan(stat).any():
        raise ValueError(
            f"{int(np.isnan(stat).sum())} of {stat.size} statistics are NaN; a NaN is "
            "neither inside the grid nor outside it and would disappear from the "
            "histogram without being counted anywhere"
        )
    edges = hist_edges()
    counts, _ = np.histogram(stat, bins=edges)
    # np.histogram's last bin is closed on the right, so a value exactly at the top edge
    # lands in it. Anything strictly above is overflow.
    return StatHistogram(
        counts=counts.astype(np.int64),
        underflow=int((stat < edges[0]).sum()),
        overflow=int((stat > edges[-1]).sum()),
        clustered=bool(clustered),
    )


def _cast(name: str, values, dtype) -> np.ndarray:
    """
    Cast one column to its stored dtype, refusing a cast that would lose information.

    Silent narrowing is the failure this exists to stop: a float GPS written into an
    integer index column reads back as a plausible sample number, and nothing downstream
    can tell it from one that was always integral.
    """
    values = np.asarray(values)
    if values.dtype == dtype:
        return values
    if not np.can_cast(values.dtype, dtype, casting="safe"):
        raise TypeError(
            f"column {name!r} is {values.dtype} and the shard stores it as "
            f"{np.dtype(dtype)}; the cast would lose information, and the truncated "
            "values would read back as ordinary ones"
        )
    return values.astype(dtype)


def _extend(dataset, values: np.ndarray) -> None:
    """Append to a resizable dataset, growing it by exactly the rows supplied."""
    start = int(dataset.shape[0])
    added = int(values.shape[0])
    if added == 0:
        return
    dataset.resize(start + added, axis=0)
    dataset[start:] = values


class TriggerWriter:
    """
    Buffered, resumable shard writer.

    Appends whole chunks under ``atomic_h5`` and marks each block complete, so a
    requeued job resumes at a block boundary and produces byte-identical output.

    The unit of commitment is the block, not the row. A block's triggers, its share of
    the per-window stream and its histogram are written inside one ``atomic_h5``
    transaction together with the block id, so a kill at any instant leaves a shard
    holding exactly the blocks that finished. Committing them separately would allow a
    shard whose triggers and whose counts describe different amounts of data, which is
    invisible afterwards because both look like ordinary products.

    Between commits the buffer spills in whole chunks so memory stays bounded on a block
    that produces millions of triggers. Those rows are on disk but belong to no completed
    block, so :meth:`__init__` truncates them away when the shard is reopened -- keeping
    them would count every trigger of the replayed block twice.

    The cost is one full copy of the shard per commit, which is what ``atomic_h5`` buys
    its atomicity with. That is why the block rather than the chunk is the commit unit:
    a campaign's blocks are minutes of data each, so the copy is amortised over a whole
    block instead of being paid once per chunk.
    """

    def __init__(
        self,
        path: str | Path,
        attrs: Dict[str, object],
        keep_stream: bool = False,
        chunk_rows: int = 1 << 16,
    ) -> None:
        """
        Open a shard for writing, creating it or resuming the one already there.

        Parameters
        ----------
        path : path
            Shard to write. An existing shard is resumed, never replaced: a requeued job
            reopens the same path and must add to what its predecessor committed.
        attrs : dict
            A complete provenance block, normally straight from
            :func:`sage.search.manifest.provenance`, plus ``clustered``, which states
            whether these triggers have already been reduced to one per event. That flag
            is required rather than defaulted because it decides whether the shard may be
            counted as a background at all, and a default would be believed.
        keep_stream : bool
            Store the statistic of every analysed window, not only the triggers above
            threshold. Zero-lag only: the stream holds one value per window, so a ladder
            of slides would write one copy of the whole run per slide.
        chunk_rows : int
            HDF5 chunk length, and the buffered row count at which the buffer spills.
            Fixed when the shard is created; a resumed writer takes the stored value, so
            a shard begun in one job and finished in another is chunked identically
            however the second job was configured.

        Raises
        ------
        ValueError
            The provenance block is incomplete, ``clustered`` is absent, the stream was
            requested for a slid shard, or the shard already on disk was written under a
            different configuration. Appending to that last one would build a single
            trigger set out of two analyses that were never comparable, and the result
            looks exactly like a valid shard.
        """
        from sage.search.manifest import PROVENANCE_KEYS

        self._path = Path(path)
        self._attrs = dict(attrs)
        self._closed = False

        chunk_rows = int(chunk_rows)
        if chunk_rows <= 0:
            raise ValueError(
                f"chunk_rows must be positive, got {chunk_rows}"
            )
        missing = [key for key in PROVENANCE_KEYS if key not in self._attrs]
        if missing:
            raise ValueError(
                f"refusing to open a shard without a complete provenance block; missing "
                f"{missing}. A shard that cannot be attributed to a configuration cannot "
                "be used in a result, and the fact is discovered a campaign later"
            )
        if "clustered" not in self._attrs:
            raise ValueError(
                "a shard must declare whether its triggers are clustered; the flag "
                "decides whether it may be counted as a background, and a default would "
                "be believed rather than checked"
            )
        self._clustered = bool(self._attrs["clustered"])
        self._attrs["clustered"] = self._clustered
        self._keep_stream = bool(keep_stream)
        slide_id = self._attrs.get("slide_id", 0)
        if self._keep_stream and int(slide_id or 0) != 0:
            raise ValueError(
                f"refusing to keep the per-window stream for slide {slide_id}: the "
                "stream is one value per analysed window, so a ladder of slides writes "
                "one copy of the whole run per slide. It is a zero-lag diagnostic"
            )

        self._chunk_rows = chunk_rows
        self._columns: Optional[Tuple[str, ...]] = None
        self._stream_columns: Optional[Tuple[str, ...]] = None
        self._stream_dtypes: Dict[str, np.dtype] = {}
        self._buffer: Dict[str, List[np.ndarray]] = {}
        self._buffered_rows = 0
        self._stream_buffer: Dict[str, List[np.ndarray]] = {}
        self._buffered_stream_rows = 0
        self._pending: Optional[StatHistogram] = None
        self._completed: List[int] = []

        if self._path.is_file():
            self._resume()
        else:
            self._create()

    def append(self, table: TriggerTable) -> None:
        """
        Buffer triggers for the current block.

        The first table fixes the shard's column set; a later one holding a different set
        is refused, since columns of unequal length do not describe one trigger set and
        the shortfall would only surface when a reader indexed past the end of one.

        The provenance keys that decide what a trigger means are checked against the
        shard's own, so a table produced under another spec, run, network or clustering
        state cannot be appended to it.
        """
        self._require_open()
        if not isinstance(table, TriggerTable):
            raise TypeError(
                f"append takes a TriggerTable, got {type(table).__name__}; the schema "
                "check lives in that type and a bare dict would bypass it"
            )
        for key in COMPATIBILITY_KEYS:
            if key in table.attrs and key in self._attrs:
                if table.attrs[key] != self._attrs[key]:
                    raise ValueError(
                        f"refusing to append a table whose {key} is "
                        f"{table.attrs[key]!r} to a shard whose {key} is "
                        f"{self._attrs[key]!r}; they describe different analyses"
                    )
        columns = tuple(name for name in TRIGGER_COLUMNS if name in table.columns)
        if self._columns is None:
            self._columns = columns
        elif set(columns) != set(self._columns):
            raise ValueError(
                f"this shard holds {sorted(self._columns)} and the table holds "
                f"{sorted(columns)}; a shard's columns are fixed by its first append"
            )
        if len(table) == 0:
            return
        for name in self._columns:
            self._buffer.setdefault(name, []).append(
                _cast(name, table[name], COLUMN_DTYPES[name])
            )
        self._buffered_rows += len(table)
        if self._buffered_rows >= self._chunk_rows:
            self._spill()

    def add_histogram(self, hist: StatHistogram) -> None:
        """
        Accumulate a block's histogram.

        Held until the block is completed so that the counts land in the same transaction
        as the triggers they describe. Added rather than replaced, because a block may be
        scored in several passes and the histogram is the sum over everything the block
        produced.
        """
        self._require_open()
        if not isinstance(hist, StatHistogram):
            raise TypeError(
                f"add_histogram takes a StatHistogram, got {type(hist).__name__}"
            )
        if hist.clustered != self._clustered:
            raise ValueError(
                f"refusing to add a histogram marked clustered={hist.clustered} to a "
                f"shard marked clustered={self._clustered}; the sum would be neither, "
                "and nothing afterwards distinguishes it from a valid one"
            )
        self._pending = hist if self._pending is None else self._pending + hist

    @property
    def keep_stream(self) -> bool:
        """
        Whether this shard records the per-window statistic.

        Public because the engine has to ask before calling :meth:`add_stream`, and a
        private name read through ``getattr(writer, "keep_stream", False)`` answers False
        for every writer -- so the stream is silently never written, on a shard whose own
        attributes say it holds one.
        """
        return self._keep_stream

    def add_stream(
        self, stat: np.ndarray, pe: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Store the full per-window statistic; zero-lag only.

        Parameters
        ----------
        stat : ndarray
            One value per analysed window, in lattice order.
        pe : dict of ndarray, optional
            Point estimates for the same windows, one array each, stored beside the
            statistic under their own names.

        Notes
        -----
        The stored dtype is whatever the first call supplies, and later calls must be
        safely castable to it. The stream is the largest product the search writes -- one
        value for every window of the run -- so promoting a float32 network output to
        float64 here would double it to gain nothing, and demoting a float64 one would
        lose the values the diagnostic exists to show.
        """
        self._require_open()
        if not self._keep_stream:
            raise ValueError(
                "this shard was opened with keep_stream=False, so it has nowhere to put "
                "a per-window stream; open it with keep_stream=True to record one"
            )
        stat = np.asarray(stat).ravel()
        payload = {"stat": stat}
        for name, values in (pe or {}).items():
            if "/" in name:
                raise ValueError(
                    f"{name!r} cannot name a stream column; a slash would nest it inside "
                    "another dataset rather than store it"
                )
            values = np.asarray(values).ravel()
            if values.shape[0] != stat.shape[0]:
                raise ValueError(
                    f"stream column {name!r} holds {values.shape[0]} values against "
                    f"{stat.shape[0]} windows; the two would be read side by side and "
                    "would describe different windows"
                )
            payload[name] = values
        names = tuple(sorted(payload))
        if self._stream_columns is None:
            self._stream_columns = names
            self._stream_dtypes = {
                name: payload[name].dtype for name in names
            }
        elif names != self._stream_columns:
            raise ValueError(
                f"this shard's stream holds {list(self._stream_columns)} and this call "
                f"supplies {list(names)}; the set is fixed by the first call"
            )
        if stat.shape[0] == 0:
            return
        for name in names:
            self._stream_buffer.setdefault(name, []).append(
                _cast(name, payload[name], self._stream_dtypes[name])
            )
        self._buffered_stream_rows += int(stat.shape[0])
        if self._buffered_stream_rows >= self._chunk_rows:
            self._spill_stream()

    def complete_block(self, block_id: int) -> None:
        """
        Flush and mark a block finished.

        Everything buffered since the last completed block, and the block id itself, are
        written in one crash-atomic transaction. After it returns the shard is consistent
        whatever happens to the job, and :meth:`completed_blocks` reports this block.

        Completing a block twice is refused rather than ignored. It would append the same
        triggers a second time, and a duplicated background event lowers every FAR taken
        from the shard while the file stays entirely well formed. A resumed job is
        expected to skip whatever :meth:`completed_blocks` already reports.
        """
        self._require_open()
        block_id = int(block_id)
        if block_id in self._completed:
            raise ValueError(
                f"block {block_id} is already recorded in {self._path}; completing it "
                "again would append its triggers a second time. A resumed job must skip "
                "the blocks completed_blocks() reports"
            )
        stacked = {
            name: np.concatenate(parts) for name, parts in self._buffer.items()
        }
        stream = {
            name: np.concatenate(parts) for name, parts in self._stream_buffer.items()
        }
        with self._transaction() as handle:
            if self._columns:
                group = self._trigger_group(handle)
                for name in self._columns:
                    if name in stacked:
                        _extend(group[name], stacked[name])
                handle.attrs["committed_rows"] = int(group[self._columns[0]].shape[0])
            if self._stream_columns:
                group = self._stream_group(handle)
                for name in self._stream_columns:
                    if name in stream:
                        _extend(group[name], stream[name])
                handle.attrs["committed_stream_rows"] = int(group["stat"].shape[0])
            if self._pending is not None:
                histogram = handle[_HISTOGRAM_GROUP]
                histogram["counts"][:] = (
                    np.asarray(histogram["counts"]) + self._pending.counts
                )
                histogram.attrs["underflow"] = (
                    int(histogram.attrs["underflow"]) + self._pending.underflow
                )
                histogram.attrs["overflow"] = (
                    int(histogram.attrs["overflow"]) + self._pending.overflow
                )
            _extend(handle[_BLOCK_DATASET], np.array([block_id], dtype=np.int64))

        self._buffer = {}
        self._buffered_rows = 0
        self._stream_buffer = {}
        self._buffered_stream_rows = 0
        self._pending = None
        self._completed.append(block_id)
        self._completed.sort()

    def completed_blocks(self) -> List[int]:
        """
        Block ids already finished, for resume.

        Ascending, and read from the shard when it was opened rather than from a sidecar,
        so the list a requeued job sees is the one the file itself can account for.
        """
        return list(self._completed)

    def close(self) -> None:
        """
        Finalise the shard.

        Refuses to discard buffered work: rows appended without their block being
        completed belong to a block that never finished, and dropping them silently would
        leave the driver believing a shard is whole. Complete the block, or let the
        resumed job replay it.

        Idempotent, so a driver may close in a ``finally`` without having to track
        whether it already did.
        """
        if self._closed:
            return
        pending = (
            self._buffered_rows
            + self._buffered_stream_rows
            + (0 if self._pending is None else 1)
        )
        if pending:
            raise ValueError(
                f"refusing to close {self._path} with {self._buffered_rows} buffered "
                f"triggers, {self._buffered_stream_rows} buffered stream values and "
                f"{'a' if self._pending is not None else 'no'} pending histogram; they "
                "belong to a block that was never completed and would be lost without a "
                "trace"
            )
        with self._transaction() as handle:
            handle.attrs["finalised"] = True
        self._repack()
        self._closed = True

    # ------------------------------------------------------------------ internals
    def _require_open(self) -> None:
        """Refuse any write after the shard has been finalised."""
        if self._closed:
            raise ValueError(
                f"{self._path} is closed; a shard is finalised once, and writing after "
                "that would extend a product other stages may already have read"
            )

    def _transaction(self):
        """One crash-atomic update of the shard."""
        from sage.utils.atomic_io import atomic_h5

        return atomic_h5(self._path)

    def _create(self) -> None:
        """
        Write the empty, stamped shard before any data exists.

        Stamping at creation rather than at close means a shard can never be committed
        without its provenance, including one abandoned half way through a campaign --
        which is exactly the shard whose origin someone will need to establish.
        """
        from sage.search.manifest import stamp

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as handle:
            stamp(handle, self._attrs)
            handle.attrs["keep_stream"] = self._keep_stream
            handle.attrs["chunk_rows"] = self._chunk_rows
            handle.attrs["committed_rows"] = 0
            handle.attrs["committed_stream_rows"] = 0
            handle.attrs["finalised"] = False
            handle.create_dataset(
                _BLOCK_DATASET, shape=(0,), maxshape=(None,), dtype=np.int64
            )
            histogram = handle.create_group(_HISTOGRAM_GROUP)
            histogram.create_dataset(
                "counts",
                data=np.zeros(STAT_HIST_NBINS, dtype=np.int64),
                compression="gzip",
            )
            histogram.attrs["underflow"] = 0
            histogram.attrs["overflow"] = 0
            histogram.attrs["clustered"] = self._clustered

    def _resume(self) -> None:
        """
        Reopen a shard, truncating rows whose block never finished.

        The configuration is checked before anything is written. A shard carried over
        from a different spec, a different clustering state or a different stream setting
        is refused: those are the three properties that decide what its triggers mean,
        and a shard holding two analyses is indistinguishable from one holding a single
        long analysis.

        Truncation is the counterpart of the mid-block spill. Rows past
        ``committed_rows`` are the tail of a block that was interrupted; the resumed job
        replays that block in full, so they have to go or every trigger in it is counted
        twice.
        """
        import h5py

        with h5py.File(self._path, "r") as handle:
            for name in ("chunk_rows", "committed_rows", "committed_stream_rows"):
                if name not in handle.attrs:
                    raise ValueError(
                        f"{self._path} carries no {name!r} attribute, so it is not a "
                        "trigger shard this writer produced and cannot be resumed"
                    )
            stored_hash = handle.attrs.get("spec_hash")
            if stored_hash is not None and "spec_hash" in self._attrs:
                stored_hash = _decoded(stored_hash)
                if stored_hash != self._attrs["spec_hash"]:
                    raise ValueError(
                        f"{self._path} was written under spec {stored_hash}, not the "
                        f"{self._attrs['spec_hash']} being run; resuming it would join "
                        "two configurations into one trigger set"
                    )
            stored_clustered = bool(handle[_HISTOGRAM_GROUP].attrs["clustered"])
            if stored_clustered != self._clustered:
                raise ValueError(
                    f"{self._path} holds clustered={stored_clustered} triggers and this "
                    f"writer was opened with clustered={self._clustered}; the shard "
                    "would be counted as one or the other and is neither"
                )
            if bool(handle.attrs["keep_stream"]) != self._keep_stream:
                raise ValueError(
                    f"{self._path} was written with keep_stream="
                    f"{bool(handle.attrs['keep_stream'])} and this writer was opened "
                    f"with keep_stream={self._keep_stream}; the stream would cover only "
                    "part of the run while looking complete"
                )
            self._chunk_rows = int(handle.attrs["chunk_rows"])
            self._completed = sorted(
                int(block) for block in np.asarray(handle[_BLOCK_DATASET])
            )
            committed = int(handle.attrs["committed_rows"])
            committed_stream = int(handle.attrs["committed_stream_rows"])
            excess = False
            if _TRIGGER_GROUP in handle:
                group = handle[_TRIGGER_GROUP]
                self._columns = tuple(
                    name for name in TRIGGER_COLUMNS if name in group
                )
                excess |= any(
                    int(group[name].shape[0]) != committed for name in self._columns
                )
            if _STREAM_GROUP in handle:
                group = handle[_STREAM_GROUP]
                self._stream_columns = tuple(sorted(group))
                self._stream_dtypes = {
                    name: group[name].dtype for name in self._stream_columns
                }
                excess |= any(
                    int(group[name].shape[0]) != committed_stream
                    for name in self._stream_columns
                )
        if not excess:
            return
        with self._transaction() as handle:
            for name in self._columns or ():
                handle[_TRIGGER_GROUP][name].resize(committed, axis=0)
            for name in self._stream_columns or ():
                handle[_STREAM_GROUP][name].resize(committed_stream, axis=0)

    def _repack(self) -> None:
        """
        Rewrite the shard once at close, so its bytes depend only on what it holds.

        Every block commit appends in place, and an interrupted block leaves space the
        resumed job's writes are then laid out around. Two shards holding identical
        triggers therefore differ in the container while agreeing in every value, which
        is enough to make a checksum of a released product useless as a check on it.
        Rewriting once, in a fixed order, removes that: a finished shard's bytes are a
        function of its content, so a rerun can be shown to have reproduced it rather
        than merely claimed to.

        The copy is made at the HDF5 level rather than through numpy, so the stream --
        the largest thing the search writes -- is never materialised in memory. The cost
        is one rewrite per shard, against the copy per block that ``atomic_h5`` already
        pays for atomicity.
        """
        import h5py

        with h5py.File(self._path, "r") as source:
            # The replace happens while `source` is still open. On this platform that
            # unlinks the old inode rather than disturbing the open handle, which is what
            # lets the shard be rewritten from itself without a second copy on disk.
            with self._transaction_fresh() as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                for name in (_BLOCK_DATASET, _HISTOGRAM_GROUP, _TRIGGER_GROUP,
                             _STREAM_GROUP):
                    if name in source:
                        source.copy(source[name], target, name=name)

    def _transaction_fresh(self):
        """A crash-atomic replacement of the shard, written from empty."""
        from sage.utils.atomic_io import atomic_h5

        return atomic_h5(self._path, mode="w")

    def _trigger_group(self, handle):
        """The trigger group, creating the shard's columns on first use."""
        group = handle.require_group(_TRIGGER_GROUP)
        for name in self._columns or ():
            if name not in group:
                group.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=COLUMN_DTYPES[name],
                    chunks=(self._chunk_rows,),
                    compression="gzip",
                )
        return group

    def _stream_group(self, handle):
        """The stream group, creating its columns on first use."""
        group = handle.require_group(_STREAM_GROUP)
        for name in self._stream_columns or ():
            if name not in group:
                group.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=self._stream_dtypes[name],
                    chunks=(self._chunk_rows,),
                    compression="gzip",
                )
        return group

    def _spill(self) -> None:
        """
        Write whole chunks out of the trigger buffer, keeping the remainder.

        Whole chunks only, so an HDF5 chunk is written once and never rewritten to be
        filled -- with compression on, a partly filled chunk would be decompressed and
        recompressed on the next append.
        """
        whole = (self._buffered_rows // self._chunk_rows) * self._chunk_rows
        if whole == 0:
            return
        stacked = {
            name: np.concatenate(parts) for name, parts in self._buffer.items()
        }
        with self._transaction() as handle:
            group = self._trigger_group(handle)
            for name in self._columns or ():
                _extend(group[name], stacked[name][:whole])
        self._buffer = {
            name: [values[whole:]] for name, values in stacked.items()
        }
        self._buffered_rows -= whole

    def _spill_stream(self) -> None:
        """Write whole chunks out of the stream buffer, keeping the remainder."""
        whole = (self._buffered_stream_rows // self._chunk_rows) * self._chunk_rows
        if whole == 0:
            return
        stacked = {
            name: np.concatenate(parts) for name, parts in self._stream_buffer.items()
        }
        with self._transaction() as handle:
            group = self._stream_group(handle)
            for name in self._stream_columns or ():
                _extend(group[name], stacked[name][:whole])
        self._stream_buffer = {
            name: [values[whole:]] for name, values in stacked.items()
        }
        self._buffered_stream_rows -= whole


def _decoded(value):
    """
    Render one stored attribute as the value that was stamped.

    Shared with :func:`sage.search.manifest.verify` rather than reimplemented, because
    :meth:`TriggerTable.concat` compares attributes for equality: h5py hands a stamped
    tuple back as a numpy array, and comparing two of those yields an array rather than a
    bool, so the compatibility check would raise instead of answering.
    """
    from sage.search.manifest import _decode_attr

    return _decode_attr(value)


def read_shard(path: str | Path) -> Tuple[TriggerTable, StatHistogram]:
    """
    Read one shard's triggers and histogram.

    Only rows belonging to completed blocks are returned. A shard whose last block was
    interrupted holds the tail of that block on disk; it will be replayed when the job is
    requeued, so counting it here as well would double every trigger in it -- and a
    reader has no other way to tell those rows from committed ones.

    Returns
    -------
    tuple
        ``(TriggerTable, StatHistogram)``. The table carries the shard's stamped
        attributes, so a merge can check that two shards describe the same analysis.
    """
    import h5py

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no trigger shard at {target}")
    with h5py.File(target, "r") as handle:
        if "committed_rows" not in handle.attrs or _HISTOGRAM_GROUP not in handle:
            raise ValueError(
                f"{target} is not a trigger shard: it carries no committed row count and "
                "no histogram, so nothing in it can be counted"
            )
        attrs = {key: _decoded(value) for key, value in handle.attrs.items()}
        committed = int(handle.attrs["committed_rows"])
        columns: Dict[str, np.ndarray] = {}
        if _TRIGGER_GROUP in handle:
            group = handle[_TRIGGER_GROUP]
            columns = {
                name: np.asarray(group[name][:committed])
                for name in TRIGGER_COLUMNS
                if name in group
            }
        histogram = handle[_HISTOGRAM_GROUP]
        hist = StatHistogram(
            counts=np.asarray(histogram["counts"]),
            underflow=int(histogram.attrs["underflow"]),
            overflow=int(histogram.attrs["overflow"]),
            clustered=bool(histogram.attrs["clustered"]),
        )
    return TriggerTable(columns=columns, attrs=attrs), hist


def merge_shards(
    paths: Sequence[str | Path], require_clustered: bool = False
) -> Tuple[TriggerTable, StatHistogram]:
    """
    Concatenate shards and add their histograms exactly.

    Both halves are done by the types that own them --
    :meth:`TriggerTable.concat` and :meth:`StatHistogram.__add__` -- so a merge cannot
    join shards whose provenance disagrees, and the counts add in integers rather than
    approximately.

    Parameters
    ----------
    require_clustered : bool
        Refuse any shard that has not been clustered. Set wherever the merged counts
        become a background: an unclustered trigger train contributes one event per
        window of a glitch instead of one per glitch, several times too many, and since
        that count is the FAR numerator every rate taken from it is wrong by the same
        factor while looking entirely ordinary.

    Notes
    -----
    A shard to which nothing was ever appended has no columns and contributes no
    triggers; it is skipped for the concatenation and still counted in the histogram sum.
    That case is ordinary rather than exceptional -- a slide that produced nothing above
    threshold -- and refusing it would make an empty background impossible to represent.
    """
    paths = list(paths)
    if not paths:
        raise ValueError(
            "no shards to merge; an empty merge has no provenance and would produce a "
            "background indistinguishable from one that measured nothing"
        )
    table: Optional[TriggerTable] = None
    total: Optional[StatHistogram] = None
    first_attrs: Dict[str, object] = {}
    for path in paths:
        shard, hist = read_shard(path)
        if require_clustered and not (
            hist.clustered and shard.attrs.get("clustered", True)
        ):
            raise ValueError(
                f"{path} holds unclustered triggers and this merge requires clustered "
                "ones: its count is one event per window of a glitch rather than one per "
                "glitch, and that count is the numerator of every rate taken from it"
            )
        total = hist if total is None else total + hist
        if not first_attrs:
            first_attrs = dict(shard.attrs)
        if not shard.columns:
            continue
        table = shard if table is None else table.concat(shard)
    if table is None:
        table = TriggerTable(columns={}, attrs=first_attrs)
    return table, total

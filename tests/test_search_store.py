#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_store.py
Description   : The queryable campaign store.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The store is the only place a campaign's facts meet, and every stage writes into it while
the campaign is still running. Two things therefore have to hold at every moment, not only
at the end: a re-run must replace what it wrote rather than adding to it, and a store
holding half the stages must still answer questions about the half that is there.

Runs on a store built in a temporary directory; needs no data, no GPU and no network.
"""

import re

import pytest

from sage.search.store import (
    TABLES,
    VIEW_NAME,
    EventRecord,
    SearchStore,
    UnknownArtefact,
    UnknownColumn,
    UnknownEvent,
    UnknownTable,
)

LOUD = "SGW230814_230901"
QUIET = "SGW230815_101112"
LOUD_GPS = 1244000000.0
QUIET_GPS = 1244003600.0


def _fresh(tmp_path, name="store.sqlite") -> SearchStore:
    """An empty store under the test's own directory."""
    return SearchStore(tmp_path / "campaign" / name)


def _populate(store: SearchStore) -> None:
    """Carry two candidates through the stages a search runs in order."""
    store.put(
        "events",
        {
            "name": [LOUD, QUIET],
            "gps": [LOUD_GPS, QUIET_GPS],
            "stat": [12.5, 8.1],
            "arm": ["HL", "HL"],
            "observing_run": ["O3a", "O3a"],
            "detectors": ["H1,L1", "H1,L1"],
        },
    )
    store.put(
        "significance",
        [
            {"name": LOUD, "far_per_yr": 0.002, "ifar_yr": 500.0, "p_value": 1e-5},
            {"name": QUIET, "far_per_yr": 5.0, "ifar_yr": 0.2, "p_value": 0.4},
        ],
    )
    store.put(
        "pastro",
        [
            {"name": LOUD, "pastro": 0.99, "pastro_lo": 0.97, "pastro_hi": 1.0},
            {"name": QUIET, "pastro": 0.05, "pastro_lo": 0.01, "pastro_hi": 0.20},
        ],
    )
    store.put(
        "dataquality",
        [
            {"name": LOUD, "task": "stationarity", "p_value": 0.62, "passed": 1},
            {"name": LOUD, "task": "excess_power", "p_value": 0.31, "passed": 1},
            {"name": QUIET, "task": "stationarity", "p_value": 0.001, "passed": 0},
            {"name": QUIET, "task": "excess_power", "p_value": 0.44, "passed": 1},
        ],
    )
    store.put(
        "consistency",
        [{"name": LOUD, "test": "coherent_versus_incoherent", "passed": 1, "value": 8.4}],
    )
    store.put(
        "parameters",
        [
            {
                "name": LOUD,
                "parameter": "chirp_mass",
                "median": 28.4,
                "lower": 25.1,
                "upper": 31.9,
                "level": 0.9,
                "waveform": "IMRPhenomXPHM",
            }
        ],
    )
    store.put(
        "catalogue_matches",
        [{"name": LOUD, "catalogue": "GWTC-3", "match_name": "GW230814", "dt_s": 0.02,
          "matched": 1}],
    )


@pytest.fixture
def store(tmp_path):
    """A store holding two candidates and the stages that ran on them."""
    with _fresh(tmp_path) as opened:
        _populate(opened)
        yield opened


class TestSchema:
    """Structure and discoverability."""

    def test_initialise_creates_every_table(self, tmp_path):
        """
        A fresh store has the full schema.

        Stages complete in whatever order a campaign runs them, so any table missing
        until its own stage first runs would make an earlier stage's write depend on a
        later one having already happened.
        """
        with _fresh(tmp_path) as opened:
            opened.initialise()
            present = set(
                opened.query("SELECT name FROM sqlite_master WHERE type = 'table'")[
                    "name"
                ]
            )
            assert set(TABLES) <= present, sorted(set(TABLES) - present)
            assert all(opened.count(table) == 0 for table in TABLES)

    def test_describe_lists_tables_and_counts(self, store):
        """
        The overview reports what is stored and how much.

        This is how a campaign is inspected part-way through: which stages have written
        and how much they wrote is the difference between a stage that is still running
        and one that finished having produced nothing.
        """
        text = store.describe()
        for table in TABLES:
            assert table in text, table
        assert re.search(r"\bevents\s+2 rows", text)
        assert re.search(r"\bdataquality\s+4 rows", text)
        assert re.search(r"\bsensitivity\s+0 rows", text)

    def test_columns_are_discoverable(self, store):
        """
        Queryable quantities can be listed without reading the source.

        A condition is written from memory at an interactive prompt; the names it may use
        have to be answerable from the store itself, including the ones that only exist
        on the joined view.
        """
        listed = store.columns()
        assert set(TABLES) <= set(listed)
        assert {"name", "gps", "tier"} <= set(store.columns("events")["events"])
        # Quantities from three separate stages, all selectable in one condition.
        assert {"ifar_yr", "pastro", "dq_vetoed"} <= set(listed[VIEW_NAME])

    def test_unknown_table_suggests(self, store):
        """A mistyped table name names the tables it was nearly."""
        with pytest.raises(UnknownTable, match="events"):
            store.table("evens")

    def test_records_survive_reopen(self, tmp_path):
        """
        A store is a file, and stages that run as separate jobs share it.

        Nothing may be held only in the writing process: a stage writes, exits, and the
        next stage opens the same file and reads what it wrote.
        """
        with _fresh(tmp_path) as opened:
            _populate(opened)
        with SearchStore(tmp_path / "campaign" / "store.sqlite", read_only=True) as again:
            assert again.count("events") == 2
            assert again.event(LOUD).fields["ifar_yr"] == pytest.approx(500.0)


class TestWriting:
    """Stages write idempotently."""

    def test_rerun_replaces_rows(self, store):
        """
        Writing the same stage twice does not duplicate rows.

        Stages are re-run whenever a campaign is resumed or a background is refitted.
        Appending instead of replacing would double every count and halve every rate
        derived from one, and nothing downstream would signal it.
        """
        before = store.count("significance")
        store.put(
            "significance",
            [
                {"name": LOUD, "far_per_yr": 0.001, "ifar_yr": 1000.0, "p_value": 1e-6},
                {"name": QUIET, "far_per_yr": 5.0, "ifar_yr": 0.2, "p_value": 0.4},
            ],
        )
        assert store.count("significance") == before
        assert store.event(LOUD).fields["ifar_yr"] == pytest.approx(1000.0)

    def test_merge_keeps_prior_columns(self, store):
        """
        Writing one column of a row leaves that row's other columns alone.

        Stages share a table: ``events`` carries the candidate from the trigger stage,
        the slide it came from, and the tier vetting assigns it. Each stage writes the
        columns it owns and no others, so a write that cleared the columns it did not
        carry would silently discard whatever ran before it -- and nothing downstream
        distinguishes a column never filled from one blanked by the next stage.
        """
        store.put("events", [{"name": LOUD, "slide_id": 7, "tc_gps": LOUD_GPS + 0.25}])
        assert store.event(LOUD).fields["slide_id"] == 7

        # A different stage now records the network, naming nothing else.
        store.put("events", [{"name": LOUD, "arm": "HLV"}])
        fields = store.event(LOUD).fields
        assert fields["arm"] == "HLV"
        assert fields["slide_id"] == 7
        assert fields["tc_gps"] == pytest.approx(LOUD_GPS + 0.25)
        assert fields["gps"] == pytest.approx(LOUD_GPS)
        assert fields["stat"] == pytest.approx(12.5)

    def test_merge_keeps_demoted_tier(self, store):
        """
        A tier survives a later write to the same row.

        Vetting only ever demotes, so restoring the undetermined default is not a lost
        value but a reversed judgement: a candidate demoted for failing a data-quality
        check would re-enter the candidate list as though nothing had judged it, and the
        release stage -- which refuses to emit only while a tier is *undetermined* --
        would pass it through.
        """
        from sage.search.candidates import TIER_UNDETERMINED

        store.put("events", [{"name": LOUD, "tier": 2}])
        store.put("events", [{"name": LOUD, "stat": 12.6}])

        fields = store.event(LOUD).fields
        assert fields["tier"] == 2
        assert fields["tier"] != TIER_UNDETERMINED
        assert fields["stat"] == pytest.approx(12.6)

    def test_partial_row_spares_neighbours(self, store):
        """
        Rows in one write assign only their own columns.

        Stages assemble their rows from whatever ran, so a write commonly holds rows of
        differing completeness. Taking the union of the columns and binding null for the
        rows that lack them would let one incomplete row erase a value on another.
        """
        store.put(
            "significance",
            [{"name": LOUD, "ifar_yr": 900.0}, {"name": QUIET, "far_per_yr": 6.0}],
        )
        loud, quiet = store.event(LOUD).fields, store.event(QUIET).fields
        assert loud["ifar_yr"] == pytest.approx(900.0)
        assert loud["far_per_yr"] == pytest.approx(0.002)  # untouched by this write
        assert quiet["far_per_yr"] == pytest.approx(6.0)
        assert quiet["ifar_yr"] == pytest.approx(0.2)

    def test_widened_key_clears_group(self, store):
        """
        Passing ``key`` deletes the rows it matches rather than merging into them.

        A stage whose output changes shape on a re-run -- a refitted background emitting
        fewer bins, a vetting pass dropping a test -- needs the surplus gone, which
        merging cannot do. The widened write is that tool, and it clears by design.
        """
        assert len(store.event(LOUD).dataquality) == 2
        store.put(
            "dataquality",
            [{"name": LOUD, "task": "stationarity", "p_value": 0.55, "passed": 1}],
            key=["name"],
        )
        remaining = store.event(LOUD).dataquality
        assert set(remaining) == {"stationarity"}
        assert remaining["stationarity"]["p_value"] == pytest.approx(0.55)
        # Scoped to the rows it matched: the other candidate is untouched.
        assert len(store.event(QUIET).dataquality) == 2

    def test_missing_reads_as_none(self, tmp_path):
        """
        Whether a field is unrecorded does not depend on the other candidates.

        The records come out of a data frame, which has no null for a numeric column and
        substitutes NaN. With one candidate the column stays object-typed and the null
        survives; with two, one of which has the value, it becomes float and the other's
        null turns into NaN. Every ``is None`` filter downstream -- including the ones
        that build the summary and the markdown table -- is then dead, and the report
        fills with ``nan`` rows for stages that simply have not run.
        """
        with _fresh(tmp_path) as opened:
            opened.put(
                "events",
                [{"name": LOUD, "gps": LOUD_GPS}, {"name": QUIET, "gps": QUIET_GPS}],
            )
            # One candidate has a p_astro; the other does not.
            opened.put("pastro", [{"name": LOUD, "pastro": 0.99}])

            assert opened.event(LOUD).fields["pastro"] == pytest.approx(0.99)
            assert opened.event(QUIET).fields["pastro"] is None
            assert opened.event(QUIET).fields["far_per_yr"] is None
            # to_markdown keeps only what is not None, so the dead filter shows up as
            # rows of `nan` for stages that have not run.
            table = opened.event(QUIET).to_markdown()
            assert "nan" not in table.lower()
            assert "pastro" not in table
            assert opened.event(LOUD).to_markdown().count("pastro") == 1

    def test_columnar_broadcasts_scalars(self, store):
        """
        A scalar beside columns is broadcast, not treated as a row of its own.

        ``{"name": [...], "arm": "HL"}`` is the natural way to write a batch that shares
        a field, and the mapping form documents columns of equal length. Falling through
        to the single-row branch instead wrote one row whose natural key was a
        JSON-encoded list of names -- a corrupt primary key in a declared column, with no
        complaint from the store that refuses a mistyped column name with suggestions.
        """
        store.put(
            "events",
            {
                "name": ["SGW230901_000001", "SGW230901_000002"],
                "gps": [1250000000.0, 1250000060.0],
                "arm": "HLV",
                "observing_run": "O3b",
            },
        )
        for name in ("SGW230901_000001", "SGW230901_000002"):
            fields = store.event(name).fields
            assert fields["arm"] == "HLV"
            assert fields["observing_run"] == "O3b"
        assert store.event("SGW230901_000001").gps == pytest.approx(1250000000.0)

    def test_frame_missing_key_refused(self, store):
        """
        A key absent from a data frame is caught by the guard written for it.

        A frame is one of the four documented input forms, and it carries a missing
        string as NaN rather than None -- so the guard that exists to explain what a
        natural key is for slipped, and sqlite raised a bare IntegrityError naming a
        constraint instead.
        """
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(
            {"name": [LOUD, None], "far_per_yr": [0.001, 5.0]}
        )
        with pytest.raises(ValueError, match="natural key"):
            store.put("significance", frame)

    def test_partial_stages_queryable(self, tmp_path):
        """
        A candidate lacking later-stage rows is still retrievable.

        The store is queried while the campaign is running, so most of it is empty most
        of the time. A missing stage has to read as an unrecorded quantity, not as a
        broken query, or the store is useless until the last stage finishes.
        """
        with _fresh(tmp_path) as opened:
            opened.put("events", [{"name": LOUD, "gps": LOUD_GPS, "stat": 12.5}])
            record = opened.event(LOUD)
            assert record.gps == pytest.approx(LOUD_GPS)
            assert record.fields["pastro"] is None
            assert record.dataquality == {}
            # A condition over a stage that has not run selects nothing, and one over a
            # stage that has still works.
            assert len(opened.select(where="ifar_yr > 1")) == 0
            assert len(opened.select(where="stat > 10")) == 1

    def test_provenance_per_stage(self, store):
        """
        Each stage records how its rows were produced.

        A number in a paper has to be traceable to the code and configuration that
        produced it. Stamping the stage and spec hash on the row itself keeps that true
        for a row whose stage is later re-run under a different configuration.
        """
        store.put_provenance("events", {"spec_hash": "aaaa", "sage_version": "0.0.1"})
        store.put_provenance("pastro", {"spec_hash": "bbbb", "n_rate_grid": 512})

        stamped = store.query("SELECT DISTINCT stage, spec_hash FROM events")
        assert list(stamped["stage"]) == ["events"]
        assert list(stamped["spec_hash"]) == ["aaaa"]

        # Provenance written before a stage's rows reaches them the same way.
        store.put("pastro", [{"name": LOUD, "pastro": 0.995}])
        traced = store.query(
            "SELECT p.name, p.stage, v.spec_hash FROM pastro p "
            "JOIN provenance v USING (stage) WHERE p.name = ?",
            [LOUD],
        )
        assert traced["spec_hash"].tolist() == ["bbbb"]
        assert store.table("provenance")["stage"].tolist() == ["events", "pastro"]

    def test_tier_starts_undetermined(self, store):
        """
        A candidate arrives at the store with no tier, and the store assigns none.

        Tiers come from significance and vetting, and vetting can only demote. A store
        that defaulted a written row to a real tier would promote every candidate it
        recorded, silently, before anything had judged it.
        """
        from sage.search.candidates import TIER_UNDETERMINED

        tiers = store.select(columns=["tier", "tier_trials"])
        assert set(tiers["tier"]) == {TIER_UNDETERMINED}
        assert set(tiers["tier_trials"]) == {TIER_UNDETERMINED}

    def test_missing_key_refused(self, store):
        """
        A row missing its natural key cannot be replaced, so it is refused on the way in.

        Such a row would be inserted afresh by every re-run, which is the exact failure
        the idempotent write exists to prevent.
        """
        with pytest.raises(ValueError, match="name"):
            store.put("significance", [{"far_per_yr": 1.0}])

    def test_read_only_refuses_writes(self, tmp_path, store):
        """
        A store opened for reading refuses to record anything.

        Inspecting a campaign is routine and often happens while it is still running;
        that must not be able to modify the campaign it is inspecting.
        """
        with SearchStore(store.path, read_only=True) as reader:
            assert reader.count("events") == 2
            with pytest.raises(PermissionError):
                reader.put("events", [{"name": "SGW000000_000000", "gps": 0.0}])


class TestQuerying:
    """Retrieval by name and by condition."""

    def test_event_returns_all_facts(self, store):
        """
        One name resolves to every stage's facts about that candidate.

        Gathering a candidate by hand means opening one file per stage and joining them
        on a name; that is the work this store exists to remove.
        """
        record = store.event(LOUD)
        assert isinstance(record, EventRecord)
        assert record.gps == pytest.approx(LOUD_GPS)
        assert record.fields["ifar_yr"] == pytest.approx(500.0)
        assert record.fields["pastro"] == pytest.approx(0.99)
        assert set(record.dataquality) == {"stationarity", "excess_power"}
        assert record.consistency["coherent_versus_incoherent"]["passed"] == 1
        assert record.parameters["chirp_mass"]["median"] == pytest.approx(28.4)
        assert record.catalogue["GWTC-3"]["match_name"] == "GW230814"
        assert "ifar_yr" in record.to_dict()
        assert LOUD in record.summary()
        assert "chirp_mass" in record.to_markdown()

    def test_select_spans_stages(self, store):
        """
        A condition may combine quantities from different stages.

        Significance, probability and the data-quality verdict are produced separately;
        selecting on all three at once should need no manual joining.
        """
        selected = store.select(
            where="pastro > 0.9 AND dq_vetoed = 0 AND ifar_yr > 100",
            order_by="ifar_yr DESC",
        )
        assert selected["name"].tolist() == [LOUD]
        # The quiet candidate fails on each of the three counts independently.
        assert store.select(where="pastro > 0.9")["name"].tolist() == [LOUD]
        assert store.select(where="dq_vetoed = 1")["name"].tolist() == [QUIET]
        assert store.select(where="ifar_yr > 100")["name"].tolist() == [LOUD]

    def test_at_gps_finds_nearby(self, store):
        """
        An externally reported time resolves to nearby candidates.

        Cross-checking a published event is a time lookup, never a name lookup: the same
        event is named differently by different groups, so matching on the label both
        misses real associations and invents false ones.
        """
        near = store.at_gps(LOUD_GPS + 0.3, tolerance_s=1.0)
        assert [record.name for record in near] == [LOUD]
        assert store.at_gps(LOUD_GPS + 0.3, tolerance_s=0.1) == []
        assert len(store.at_gps(LOUD_GPS, tolerance_s=1e5)) == 2

    def test_raw_sql_supported(self, store):
        """
        Questions beyond the helpers can be asked directly.

        The helpers cover the common questions and cannot cover the rest; a campaign that
        can only be asked what was anticipated is a campaign whose surprises stay hidden.
        """
        frame = store.query(
            """
            SELECT e.name, s.ifar_yr, p.pastro, c.catalogue, c.dt_s
            FROM events e
            JOIN significance s USING (name)
            JOIN pastro p       USING (name)
            LEFT JOIN catalogue_matches c USING (name)
            WHERE p.pastro > 0.5
            ORDER BY s.ifar_yr DESC
            """
        )
        assert frame["name"].tolist() == [LOUD]
        assert frame["catalogue"].tolist() == ["GWTC-3"]
        counted = store.query(
            "SELECT COUNT(*) AS n FROM events WHERE gps > ?", [LOUD_GPS]
        )
        assert counted["n"].tolist() == [1]

    def test_unknown_column_suggests(self, store):
        """
        A mistyped quantity is reported clearly rather than returning nothing.

        An unrecognised name in a condition is indistinguishable, in its result, from a
        correct condition over a stage that has not run: both give an empty selection.
        Naming the near neighbours is what separates the two.
        """
        with pytest.raises(UnknownColumn, match="pastro"):
            store.select(where="pastroo > 0.9")
        with pytest.raises(UnknownColumn, match="ifar_yr"):
            store.select(columns=["name", "ifar_year"])
        with pytest.raises(UnknownColumn, match="ifar_yr"):
            store.query(f"SELECT ifar_year FROM {VIEW_NAME}")
        with pytest.raises(UnknownEvent, match=LOUD):
            store.event("SGW230814_230900")
        # A function call and a literal are not column names.
        assert len(store.select(where="ABS(gps - 1244000000.0) < 1.0")) == 1
        assert len(store.select(where="detectors = 'H1,L1'")) == 2


class TestArtefacts:
    """Bulk data is referenced, not copied."""

    def test_artefact_paths_resolve(self, store, tmp_path):
        """
        Recorded spectrogram and posterior paths resolve from the record.

        Spectrograms and posterior samples are far too large to sit in a row, so the
        store holds their locations; a candidate's record has to resolve to them anyway,
        or the split costs more than it saves.
        """
        bulk = tmp_path / "bulk"
        bulk.mkdir()
        spectrogram = bulk / f"{LOUD}_H1_qscan.hdf"
        posterior = bulk / f"{LOUD}_posterior.hdf5"
        for path in (spectrogram, posterior):
            path.write_bytes(b"\x00" * 32)
        store.put_artefact(LOUD, "spectrogram", spectrogram, detector="H1")
        store.put_artefact(LOUD, "posterior", posterior, waveform="IMRPhenomXPHM")

        record = store.event(LOUD)
        assert set(record.artefacts) == {"spectrogram", "posterior"}
        assert record.artefact("spectrogram") == spectrogram
        assert record.artefact("posterior").read_bytes() == b"\x00" * 32
        assert store.query("SELECT bytes FROM artefacts WHERE kind = 'posterior'")[
            "bytes"
        ].tolist() == [32]

    def test_missing_artefact_reported(self, store, tmp_path):
        """
        A recorded path that no longer exists is reported, not silently empty.

        The store records locations, not contents, so a purged scratch directory is an
        ordinary occurrence. Returning the path regardless would surface as a failure to
        read it somewhere far from the store, without saying what was expected where.
        """
        bulk = tmp_path / "bulk"
        bulk.mkdir()
        spectrogram = bulk / f"{LOUD}_H1_qscan.hdf"
        spectrogram.write_bytes(b"\x00")
        store.put_artefact(LOUD, "spectrogram", spectrogram)
        record = store.event(LOUD)
        spectrogram.unlink()

        with pytest.raises(FileNotFoundError, match=str(spectrogram)):
            record.artefact("spectrogram")
        with pytest.raises(UnknownArtefact, match="spectrogram"):
            record.artefact("spectrograms")


class TestExport:
    """Output for papers and sharing."""

    def test_export_formats_round_trip(self, store, tmp_path):
        """
        A selection exports and reads back unchanged.

        A table in a paper is written once and checked many times; if the exported file
        cannot be read back and compared against the store, the check is by eye.
        """
        import h5py
        import pandas as pd

        selection = store.select(where="pastro > 0.9")
        out = tmp_path / "export"

        csv = store.export(out / "candidates.csv", fmt="csv", where="pastro > 0.9")
        back = pd.read_csv(csv)
        assert back["name"].tolist() == selection["name"].tolist()
        assert back["ifar_yr"].tolist() == pytest.approx(selection["ifar_yr"].tolist())

        h5 = store.export(out / "candidates.h5", fmt="hdf5", where="pastro > 0.9")
        with h5py.File(h5, "r") as handle:
            assert [name.decode() for name in handle["name"][:]] == [LOUD]
            assert handle["ifar_yr"][:] == pytest.approx([500.0])

        for fmt, suffix in (("markdown", ".md"), ("latex", ".tex")):
            path = store.export(out / f"candidates{suffix}", fmt=fmt, where="pastro > 0.9")
            text = path.read_text(encoding="utf-8")
            assert LOUD in text and QUIET not in text

        with pytest.raises(ValueError, match="hdf5"):
            store.export(out / "candidates.parquet", fmt="parquet")

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="comparison_matrix is a layer-9 aggregate",
    )
    def test_matrix_marks_uncovered(self, store):
        """
        Not found and not searched are distinct states.

        A catalogue that did not cover a candidate's parameters or time says nothing
        about it, and the matrix must not present that as a non-detection.
        """
        store.put(
            "catalogue_coverage",
            [
                {"catalogue": "GWTC-3", "name": LOUD, "covered": 1},
                {"catalogue": "GWTC-3", "name": QUIET, "covered": 1},
                {"catalogue": "IAS-O3a", "name": LOUD, "covered": 1},
                {
                    "catalogue": "IAS-O3a",
                    "name": QUIET,
                    "covered": 0,
                    "reason": "outside the searched chirp-mass range",
                },
            ],
        )
        matrix = store.comparison_matrix()
        assert matrix.loc[LOUD, "GWTC-3"] == "found"
        assert matrix.loc[QUIET, "GWTC-3"] == "not found"
        assert matrix.loc[QUIET, "IAS-O3a"] == "not searched"

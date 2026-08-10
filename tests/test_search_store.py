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
"""

import pytest


class TestSchema:
    """Structure and discoverability."""

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_initialise_creates_every_table(self):
        """A fresh store has the full schema."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_describe_lists_tables_and_counts(self):
        """The overview reports what is stored and how much."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_columns_are_discoverable(self):
        """Queryable quantities can be listed without reading the source."""
        raise NotImplementedError


class TestWriting:
    """Stages write idempotently."""

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_rerunning_a_stage_replaces_its_rows(self):
        """Writing the same stage twice does not duplicate rows."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_partial_stages_do_not_block_queries(self):
        """A candidate lacking later-stage rows is still retrievable."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_provenance_recorded_per_stage(self):
        """Each stage records how its rows were produced."""
        raise NotImplementedError


class TestQuerying:
    """Retrieval by name and by condition."""

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_event_returns_everything_recorded(self):
        """One name resolves to every stage's facts about that candidate."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_select_spans_stages_in_one_condition(self):
        """
        A condition may combine quantities from different stages.

        Significance, probability and the data-quality verdict are produced separately;
        selecting on all three at once should need no manual joining.
        """
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_at_gps_finds_candidates_near_a_time(self):
        """An externally reported time resolves to nearby candidates."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_arbitrary_sql_is_supported(self):
        """Questions beyond the helpers can be asked directly."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_unknown_column_raises_with_suggestions(self):
        """A mistyped quantity is reported clearly rather than returning nothing."""
        raise NotImplementedError


class TestArtefacts:
    """Bulk data is referenced, not copied."""

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_artefact_paths_resolve(self):
        """Recorded spectrogram and posterior paths resolve from the record."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_missing_artefact_reports_clearly(self):
        """A recorded path that no longer exists is reported, not silently empty."""
        raise NotImplementedError


class TestExport:
    """Output for papers and sharing."""

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_export_formats_round_trip(self):
        """A selection exports and reads back unchanged."""
        raise NotImplementedError

    @pytest.mark.xfail(
        strict=True,
        raises=NotImplementedError,
        reason="sage.search.store is not implemented yet",
    )
    def test_comparison_matrix_distinguishes_uncovered(self):
        """
        Not found and not searched are distinct states.

        A catalogue that did not cover a candidate's parameters or time says nothing
        about it, and the matrix must not present that as a non-detection.
        """
        raise NotImplementedError

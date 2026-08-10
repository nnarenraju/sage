#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_references.py
Description   : The reference registry is well formed and renders a bibliography.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A convenience registry: it collects the documents behind the methods so a bibliography
can be generated and a reader can find the sources locally. It is not a constraint on the
code, so nothing here asserts that a particular method cites a particular equation.

Whether the PDFs are present is a property of the checkout rather than of the code, so
that check is skipped when they are absent instead of failing a fresh clone.

Runs anywhere; needs no data, no GPU and no network.
"""

import pytest

from sage.search import references as R


@pytest.fixture
def references_present():
    """Skip when the reference PDFs are not in this checkout."""
    if not R.REFERENCE_DIR.is_dir():
        pytest.skip(
            f"{R.REFERENCE_DIR} is absent; run docs/references/fetch.py to populate it"
        )


class TestRegistry:
    """The registry is internally consistent."""

    def test_known_key(self):
        assert R.get("fgmc").arxiv_id == "1302.5341"

    def test_unknown_key_suggests_alternatives(self):
        with pytest.raises(KeyError, match="fgmc"):
            R.get("fgmcc")

    def test_every_entry_has_a_title_and_filename(self):
        for key, ref in R.REFERENCES.items():
            assert ref.title.strip(), key
            assert ref.filename == f"arxiv_{ref.arxiv_id}.pdf", key

    def test_documents_are_not_duplicated(self):
        ids = [ref.arxiv_id for ref in R.REFERENCES.values()]
        assert len(ids) == len(set(ids))

    def test_registry_points_inside_the_repository(self):
        assert R.REFERENCE_DIR.name == "references"
        assert R.REFERENCE_DIR.parent.name == "docs"


class TestCitation:
    """Rendering a pointer to a local document."""

    def test_cite_names_the_local_file(self):
        assert "arxiv_1302.5341.pdf" in R.cite("fgmc")

    def test_cite_can_mention_an_equation(self):
        text = R.cite("fgmc", "21")
        assert "arxiv_1302.5341.pdf" in text and "21" in text


class TestPresence:
    """Whether the documents are on disk."""

    def test_verify_all_reports_every_key(self, references_present):
        assert set(R.verify_all()) == set(R.REFERENCES)

    def test_bibliography_lists_every_entry(self, references_present):
        text = R.bibliography()
        for ref in R.REFERENCES.values():
            assert ref.arxiv_id in text

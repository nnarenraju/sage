#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_gwosc.py
Description   : GWOSC event-list parsing.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The parser turns one release's JSON into the event list every comparison is scored
against, so a field read under the wrong convention does not raise -- it changes which
events the search is judged on.
"""

import pytest

from sage.search.catalogue.gwosc import parse_allevents


def _payload(**fields):
    """One GWOSC record, with the mass fields under test."""
    record = {"GPS": 1238782700.2, "far": 1e-5, "catalog.shortName": "GWTC-2.1-confident"}
    record.update(fields)
    return {"events": {"GW190408_181802-v2": record}}


class TestMassBounds:
    """``*_lower`` is a signed offset from the median, not an absolute bound."""

    def test_offset_becomes_a_mass(self):
        """
        GWOSC publishes ``mass_2_source = 18.5`` with ``mass_2_source_lower = -4.0``,
        meaning the interval reaches 14.5. Read directly, the bound compares as -4.0
        solar masses; measured before this fix, a BBH cut at 3.0 then kept 109 of 391
        catalogue entries and excluded every confident event in the O3a development
        window, scoring seven recoverable events as unsearchable.
        """
        catalogue = parse_allevents(
            _payload(mass_1_source=24.8, mass_2_source=18.5, mass_2_source_lower=-4.0)
        )
        event = catalogue.events[0]
        assert event.mass2 == pytest.approx(18.5)
        assert event.extra["mass2_lower_bound"] == pytest.approx(14.5)
        assert len(catalogue.filter_bbh(min_secondary_mass=3.0)) == 1

    def test_neutron_star_still_excluded(self):
        """The cut must keep rejecting what it exists to reject."""
        catalogue = parse_allevents(
            _payload(mass_1_source=2.1, mass_2_source=1.3, mass_2_source_lower=-0.2)
        )
        assert catalogue.events[0].extra["mass2_lower_bound"] == pytest.approx(1.1)
        assert len(catalogue.filter_bbh(min_secondary_mass=3.0)) == 0

    def test_absent_bound_falls_back(self):
        """
        A release that publishes no error bar leaves the point estimate to judge on,
        rather than a bound invented from a missing field.
        """
        catalogue = parse_allevents(_payload(mass_2_source=18.5))
        assert catalogue.events[0].extra["mass2_lower_bound"] == pytest.approx(18.5)

    def test_unsigned_offset_refused(self):
        """
        A positive offset means the convention changed under us. Adding it would raise
        the bound above the median and silently harden the cut, so the point estimate is
        used instead and the event is judged on a real mass.
        """
        catalogue = parse_allevents(
            _payload(mass_2_source=18.5, mass_2_source_lower=4.0)
        )
        assert catalogue.events[0].extra["mass2_lower_bound"] == pytest.approx(18.5)

    def test_no_masses_kept(self):
        """
        A marginal candidate carries no masses at all. It must survive the cut: absence
        of a measurement is not evidence of a light companion.
        """
        catalogue = parse_allevents(_payload())
        assert catalogue.events[0].extra["mass2_lower_bound"] is None
        assert len(catalogue.filter_bbh(min_secondary_mass=3.0)) == 1


class TestRecordFields:
    """Everything but the time is optional, and a missing field is not an error."""

    def test_gps_required(self):
        """An event that cannot be placed in time cannot be compared against anything."""
        with pytest.raises(ValueError, match="no GPS time"):
            parse_allevents({"events": {"GW000000": {"far": 1.0}}})

    def test_missing_far_is_none(self):
        """A release that reported no FAR is not a malformed record."""
        catalogue = parse_allevents({"events": {"GW000000": {"GPS": 1.0}}})
        assert catalogue.events[0].far_per_yr is None

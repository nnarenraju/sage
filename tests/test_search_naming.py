#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_naming.py
Description   : Candidate names: the UTC stamp, the round trip and the prefix policy.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The stamp is a UTC calendar time derived from a GPS time, so it depends on leap seconds
and cannot be produced by arithmetic on the GPS epoch. The check that matters is that
real published events reproduce their real published names: an implementation off by the
current 18-second offset still looks like a plausible timestamp.

Runs anywhere; needs no data, no GPU and no network. Needs astropy for the time scales.

Sage candidates carry the ``SGW`` prefix at every tier, so a candidate is never mistaken
for a published LVK event in a table or a filename, and a name never changes when a tier
does. Cross-matched events report their catalogue name in a separate column.
"""

import pytest

from sage.search.naming import (
    DEFAULT_PREFIX,
    check_prefix_policy,
    gps_from_name,
    name_from_gps,
    normalise_external_name,
)

# GPS times and published stamps of real events, taken from the GWOSC event list.
PUBLISHED = [
    (1239082262.1, "190412_053044"),
    (1238782700.2, "190408_181802"),
    (1240215503.0, "190425_081805"),
    (1238303737.2, "190403_051519"),
]


class TestStamp:
    """The calendar stamp is real UTC, leap seconds included."""

    @pytest.mark.parametrize("gps,stamp", PUBLISHED)
    def test_reproduces_published_stamps(self, gps, stamp):
        """
        Four real events reproduce their published names exactly.

        This is the test that catches a GPS-to-UTC conversion that ignores leap seconds:
        the result is still a well-formed timestamp, just wrong by 18 seconds.
        """
        assert name_from_gps(gps) == f"{DEFAULT_PREFIX}{stamp}"

    def test_default_prefix_is_sage_specific(self):
        """A Sage candidate is never presented as a published LVK detection."""
        assert DEFAULT_PREFIX == "SGW"
        assert name_from_gps(PUBLISHED[0][0]).startswith("SGW")

    def test_sub_second_is_truncated_not_rounded(self):
        """
        The stamp names the second the event falls in.

        Rounding would push an event at .9 s into the next second and give it a name that
        disagrees with its own GPS time by more than the naming resolution.
        """
        base = 1239082262.0
        assert name_from_gps(base + 0.1) == name_from_gps(base)
        assert name_from_gps(base + 0.9) == name_from_gps(base)
        assert name_from_gps(base + 1.0) != name_from_gps(base)

    def test_custom_prefix(self):
        assert name_from_gps(PUBLISHED[0][0], prefix="TEST").startswith("TEST190412")

    def test_negative_gps_is_refused(self):
        """GPS time starts in 1980; a negative value is a unit or offset error."""
        with pytest.raises(ValueError):
            name_from_gps(-1.0)


class TestRoundTrip:
    """A name parses back to the second it names."""

    @pytest.mark.parametrize("gps,stamp", PUBLISHED)
    def test_round_trip_to_the_second(self, gps, stamp):
        recovered = gps_from_name(name_from_gps(gps))
        assert recovered == pytest.approx(int(gps), abs=1.0)

    def test_parses_an_external_gw_name(self):
        """A published GW name parses too, so catalogue rows can be placed in time."""
        assert gps_from_name("GW190412_053044") == pytest.approx(1239082262.0, abs=1.0)

    def test_name_without_a_time_is_refused(self):
        """The short form is ambiguous within a day and is not accepted."""
        with pytest.raises(ValueError):
            gps_from_name("GW190412")

    def test_malformed_name_is_refused(self):
        for bad in ("SGW", "SGWnotadate_000000", "190412_053044_extra", ""):
            with pytest.raises(ValueError):
                gps_from_name(bad)


class TestPrefixPolicy:
    """The bare GW prefix is reserved."""

    def test_bare_gw_prefix_is_refused(self):
        """
        Sage does not assign GW names.

        The prefix denotes a published detection, and a search producing its own
        candidate list must not mint one.
        """
        with pytest.raises(ValueError, match="GW"):
            check_prefix_policy("GW", p_astro=0.99)

    def test_sage_prefix_always_allowed(self):
        """The Sage prefix carries no claim, so no probability bar applies to it."""
        for p in (0.0, 0.5, 0.99):
            check_prefix_policy(DEFAULT_PREFIX, p_astro=p)

    def test_override_needs_reason(self):
        """An override is possible but must be justified in writing."""
        check_prefix_policy("GW", p_astro=0.99, force_reason="reproducing GWTC-2.1")

    def test_override_reason_non_empty(self):
        with pytest.raises(ValueError):
            check_prefix_policy("GW", p_astro=0.99, force_reason="")


class TestExternalNames:
    """Catalogue names are labels; matching happens on time."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("GW190412_053044", "190412_053044"),
            ("GW190412", "190412"),
            ("gw190412-053044", "190412_053044"),
            ("  GW190412_053044  ", "190412_053044"),
            ("SGW190412_053044", "190412_053044"),
            ("190412_053044", "190412_053044"),
        ],
    )
    def test_normalisation(self, raw, expected):
        """Different catalogues spell the same event differently."""
        assert normalise_external_name(raw) == expected

    def test_normalisation_is_idempotent(self):
        once = normalise_external_name("GW190412_053044")
        assert normalise_external_name(once) == once

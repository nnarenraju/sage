#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_eventlist.py
Description   : The one input every external comparison goes through.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

Comparing against an external catalogue used to mean reading event times off a paper,
converting them to GPS by hand, and checking them against the candidate list by eye. This
is what replaces that, so the tests are about the two places that went wrong: the time
conversion, and columns lining up with the wrong events.
"""

import numpy as np
import pytest

from sage.search.catalogue.eventlist import from_times, read_event_times, to_gps

#: GW190412's GPS time, which several of these check against.
GW190412_GPS = 1239082262.0


class TestTimeConversion:
    """A time, however it happened to be written down."""

    def test_gps_number(self):
        assert to_gps(GW190412_GPS + 0.2) == GW190412_GPS + 0.2

    def test_gps_string(self):
        assert to_gps(str(GW190412_GPS)) == GW190412_GPS

    def test_event_name(self):
        """A GWTC name carries its own UTC stamp, so no conversion is needed by hand."""
        assert to_gps("GW190412_053044") == GW190412_GPS

    def test_sage_name(self):
        assert to_gps("SGW190412_053044") == GW190412_GPS

    def test_utc_string(self):
        assert to_gps("2019-04-12 05:30:44") == GW190412_GPS

    def test_iso_string(self):
        assert to_gps("2019-04-12T05:30:44") == GW190412_GPS

    def test_all_forms_agree(self):
        """
        The four spellings of one event must give one time. They are what different
        sources publish, and a disagreement here is a systematic offset in every
        comparison that mixes them.
        """
        forms = ("GW190412_053044", "2019-04-12 05:30:44", "2019-04-12T05:30:44",
                 str(GW190412_GPS))
        assert len({to_gps(f) for f in forms}) == 1

    def test_unparseable_is_refused_with_guidance(self):
        with pytest.raises(ValueError, match="not a time this understands"):
            to_gps("last Tuesday")

    def test_empty_is_refused(self):
        with pytest.raises(ValueError):
            to_gps("   ")


class TestFromTimes:
    """Building a list directly."""

    def test_ifar_is_derived(self):
        catalogue = from_times("x", [1.0, 2.0], far_per_yr=[0.01, 0.5])
        assert [e.ifar_yr for e in catalogue.events] == [100.0, 2.0]

    def test_missing_significance_stays_missing(self):
        """A source that published no FAR has none; nan would read as a value."""
        catalogue = from_times("x", [1.0], far_per_yr=[None])
        assert catalogue.events[0].far_per_yr is None
        assert catalogue.events[0].ifar_yr is None

    def test_mismatched_column_is_refused(self):
        """
        Columns are paired elementwise, so a length mismatch attaches a significance to
        the wrong event -- silently, and in a way nothing downstream can detect.
        """
        with pytest.raises(ValueError, match="paired elementwise"):
            from_times("x", [1.0, 2.0], far_per_yr=[0.1])

    def test_no_times_is_refused(self):
        with pytest.raises(ValueError, match="no event times"):
            from_times("x", [])

    def test_names_are_generated_when_absent(self):
        catalogue = from_times("x", ["GW190412_053044"])
        assert catalogue.events[0].name


class TestReadEventTimes:
    """What a person actually produces when copying a table out of a paper."""

    def test_header_in_any_order(self, tmp_path):
        path = tmp_path / "cat.csv"
        path.write_text("far, name, utc\n0.01, GW190412, 2019-04-12 05:30:44\n")
        catalogue = read_event_times(path, key="ias")
        assert len(catalogue) == 1
        assert catalogue.events[0].name == "GW190412"
        assert catalogue.events[0].gps == GW190412_GPS
        assert catalogue.events[0].far_per_yr == 0.01

    def test_comments_and_blank_lines(self, tmp_path):
        path = tmp_path / "cat.txt"
        path.write_text("# from table 2\n\nGW190412_053044\n\n# note\nGW190413_052954\n")
        assert len(read_event_times(path)) == 2

    def test_no_header_times_only(self, tmp_path):
        path = tmp_path / "t.txt"
        path.write_text(f"{GW190412_GPS}\n{GW190412_GPS + 100}\n")
        catalogue = read_event_times(path)
        assert [e.gps for e in catalogue.events] == [GW190412_GPS, GW190412_GPS + 100]

    def test_no_header_time_and_far(self, tmp_path):
        path = tmp_path / "t.txt"
        path.write_text(f"{GW190412_GPS} 0.01\n{GW190412_GPS + 100} 0.5\n")
        catalogue = read_event_times(path)
        assert [e.far_per_yr for e in catalogue.events] == [0.01, 0.5]

    def test_header_without_a_time_is_refused(self, tmp_path):
        """
        Reading it positionally would succeed on the wrong column and give plausible
        times that are wrong by hours.
        """
        path = tmp_path / "bad.csv"
        path.write_text("name,far\nX,0.1\n")
        with pytest.raises(ValueError, match="no time column"):
            read_event_times(path)

    def test_ragged_row_is_refused(self, tmp_path):
        path = tmp_path / "ragged.csv"
        path.write_text("gps,far\n1.0,0.1\n2.0\n")
        with pytest.raises(ValueError, match="fields against"):
            read_event_times(path)

    def test_empty_file_is_refused(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("# nothing here\n\n")
        with pytest.raises(ValueError, match="no event lines"):
            read_event_times(path)

    def test_feeds_the_comparison(self, tmp_path):
        """
        The whole point: a pasted table crossmatches against candidates without anyone
        converting a time.
        """
        from sage.search.crossmatch import classify

        path = tmp_path / "cat.csv"
        path.write_text("name,utc\nGW190412,2019-04-12 05:30:44\n")
        catalogue = read_event_times(path, key="paper")
        candidates = {"gps": np.array([GW190412_GPS + 0.1, GW190412_GPS + 5000.0])}
        out = classify(candidates, {"paper": catalogue}, tolerance_s=1.0)
        assert out["known"].tolist() == [True, False]
        assert out["catalogue_match"][0] == "GW190412"

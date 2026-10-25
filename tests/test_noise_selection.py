"""Tests for sage.data.noise.selection: portion and gps_range."""

import pytest

from sage.data.noise import selection


def record(start: float, nsamples: int, rate: float = 1.0, index: int = 0) -> dict:
    """A sidecar segment record."""
    return {
        "gps_start": start,
        "gps_end": start + nsamples / rate,
        "nsamples": nsamples,
        "sample_rate": rate,
        "sample_start_idx": index,
    }


RECORDS = {
    "H1": [record(0, 10), record(20, 10), record(40, 10)],
    "L1": [record(0, 10), record(25, 10), record(40, 10)],
}


def spans(selected: dict) -> dict:
    return {det: [(r["gps_start"], r["gps_end"]) for r in recs] for det, recs in selected.items()}


def test_portion_cuts():
    assert spans(selection.select_portion(RECORDS, [0.0, 0.5])) == {
        "H1": [(0, 10)], "L1": [(0, 10)]
    }
    assert spans(selection.select_portion(RECORDS, [0.5, 1.0])) == {
        "H1": [(20, 30), (40, 50)], "L1": [(25, 35), (40, 50)]
    }


def test_portion_drops_straddler():
    early = spans(selection.select_portion(RECORDS, [0.0, 0.6]))
    late = spans(selection.select_portion(RECORDS, [0.6, 1.0]))
    assert (20, 30) not in early["H1"] + late["H1"]
    assert early["L1"] == [(0, 10)] and late["L1"] == [(25, 35), (40, 50)]


def test_gps_range_trims():
    records = {"H1": [record(100, 40, rate=4.0, index=1000), record(120, 8, rate=4.0), record(300, 8, rate=4.0)]}
    kept = selection.select_gps_range(records, [102.25, 200.0])["H1"]
    assert (kept[0]["sample_start_idx"], kept[0]["nsamples"], kept[0]["gps_start"]) == (1009, 31, 102.25)
    assert kept[1] is records["H1"][1]
    assert len(kept) == 2


def test_disjoint():
    late = selection.select_portion(RECORDS, [0.5, 1.0])
    selection.check_disjoint(selection.select_portion(RECORDS, [0.0, 0.5]), late, "o3a")
    with pytest.raises(ValueError, match="share o3a H1 data"):
        selection.check_disjoint(selection.select_portion(RECORDS, [0.0, 0.7]), late, "o3a")


def test_source_errors():
    with pytest.raises(ValueError, match="portion or gps_range, not both"):
        selection.select_records(RECORDS, {"portion": [0, 1], "gps_range": [0, 1]})
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        selection.select_records(RECORDS, {"portion": [0.9, 0.1]})
    with pytest.raises(ValueError, match="no H1 data"):
        selection.select_records(RECORDS, {"gps_range": [1000.0, 2000.0]})

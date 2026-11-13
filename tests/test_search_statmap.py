"""Statmap: triggers of one stream, NA removal orders, hierarchical removal."""

import numpy as np
from pycbc.conversions import sec_to_year

from sage.search.statmap import hierarchical_rates, stream_triggers

COLUMNS = ["ranking_statistic", "non_astrophysical_logit", "delta_tc_raw_mean",
           "mchirp_raw_mean", "delta_tc_raw_width", "mchirp_raw_width"]
NETWORK = {"detectors": ["H1", "L1"], "targets": ["delta_tc", "mchirp"],
           "offsets": np.zeros(2), "scales": np.ones(2), "width_bounds": (1e-3, 10.0),
           "padding_s": 0.0}
GPS = np.array([0.0, 5.0, 100.0, 200.0, 300.0, 400.0])


def rows() -> np.ndarray:
    """Six windows; window 1 is NA-like and beside window 0."""
    values = np.zeros((6, 6), dtype=np.float32)
    values[:, 0] = [3, 4, 1, 6, 2, 5]
    values[:, 1] = [0, 10, 0, 0, 0, 0]
    return values


def windows(remove_na: str, shift: int = 0, l1_bad: int = None) -> tuple:
    """Kept windows and live time of one stream; H1 window 4 is bad time."""
    clear = {"H1": np.ones(6, dtype=bool), "L1": np.ones(6, dtype=bool)}
    clear["H1"][4] = False
    if l1_bad is not None:
        clear["L1"][l1_bad] = False
    triggers, live_s = stream_triggers(NETWORK, rows(), COLUMNS, 6, GPS, clear,
                                       shift, 1.0, "greedy", remove_na)
    return triggers["window"].tolist(), live_s, triggers


def test_stream_triggers():
    """Bad time dropped; NA removed before or after clustering; pairs shifted."""
    assert windows("off")[:2] == ([1, 2, 3, 5], 5.0)
    assert windows("before-clustering")[0] == [0, 2, 3, 5]  # window 0 now leads
    assert windows("after-clustering")[0] == [2, 3, 5]  # window 1 won, then went
    kept, live_s, triggers = windows("off", shift=1, l1_bad=3)
    assert live_s == 4.0  # pairs 2 (L1 window 3) and 4 (H1 bad) out
    assert triggers["partner"].tolist() == [(i + 1) % 6 for i in kept]
    assert np.array_equal(triggers["L1_tc"], GPS[triggers["partner"]])


def test_hierarchical_rates():
    """The loudest trigger beats all background, keeps its rate, takes its data."""
    year = 1 / sec_to_year(1.0)
    foreground = {"ranking_statistic": np.array([10.0, 7.0, 4.0]),
                  "H1_tc": np.array([0.0, 100.0, 200.0]),
                  "L1_tc": np.array([0.0, 100.0, 200.0])}
    background = {"ranking_statistic": np.array([8.0, 5.0, 2.0]),
                  "H1_tc": np.array([0.5, 500.0, 700.0]),
                  "L1_tc": np.array([900.0, 600.0, 800.0])}
    rates, far, removed_round = hierarchical_rates(foreground, background, year,
                                                   ["H1", "L1"])
    assert np.allclose(rates, [[1, 2, 3], [np.nan, 1, 2], [np.nan, np.nan, 2]],
                       equal_nan=True)
    assert np.allclose(far, [1, 1, 2])
    assert removed_round.tolist() == [1, 0, 0]

"""Tests for the P-P (PIT) sigma-calibration plot."""

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless

from sage.plotting import plot_pp_calibration


def _data(scale, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, (n, 2))
    sigma = np.abs(rng.normal(1, 0.2, (n, 2))) + 0.5
    y = mu + scale * sigma * rng.standard_normal((n, 2))
    return mu, sigma, y


def test_calibrated_is_near_diagonal():
    m = plot_pp_calibration(*_data(1.0), param_names=["tc", "mchirp"], save=True)
    for v in m.values():
        assert v["ks"] < 0.02
        assert abs(v["cov1sigma"] - 0.68) < 0.03
        assert abs(v["cov2sigma"] - 0.955) < 0.02


def test_overconfident_has_low_coverage_and_large_ks():
    m = plot_pp_calibration(*_data(2.0), param_names=["tc", "mchirp"], save=True)
    for v in m.values():
        assert v["ks"] > 0.1
        assert v["cov1sigma"] < 0.5      # sigma too small -> under-covers


def test_underconfident_has_high_coverage():
    m = plot_pp_calibration(*_data(0.5), param_names=["tc", "mchirp"], save=True)
    for v in m.values():
        assert v["ks"] > 0.1
        assert v["cov1sigma"] > 0.9      # sigma too large -> over-covers


def test_accepts_1d_input():
    rng = np.random.default_rng(1)
    mu = rng.normal(0, 1, 5000)
    sigma = np.ones(5000)
    y = mu + sigma * rng.standard_normal(5000)
    m = plot_pp_calibration(mu, sigma, y, save=True)
    assert len(m) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL P-P CALIBRATION TESTS PASSED <<<")

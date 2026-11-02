"""Tests for sage.data.asd.lines: the F-test, line detection and excision."""

import numpy as np
import torch
from multitaper import MTSpec

from sage.core.geometry import window_from_data
from sage.data.asd.lines import excision_weight, ftest, line_bins, line_detections
from sage.data.waveform.snr import OptimalSNREstimator
from sage.dsp.whiten import FiducialWhitening


def test_ftest_matches_package():
    """The array F-test equals multitaper's own loop."""
    x = np.random.default_rng(0).standard_normal(1024)
    spec = MTSpec(x, nw=4.0, kspec=7, dt=1 / 1024, nfft=1024, iadapt=1)
    assert np.allclose(ftest(spec), spec.ftest()[0][:, 0], rtol=1e-10, atol=0)


def test_finds_a_sinusoid():
    """A sinusoid in white noise is a line bin; no line bin lies away from it."""
    npts, k = 1024, 100
    rng = np.random.default_rng(1)

    def sample(n):
        phase = rng.uniform(0, 2 * np.pi)
        line = 0.5 * np.sin(2 * np.pi * k * np.arange(n) / n + phase)
        return rng.standard_normal(n) + line

    detections = line_detections(sample, npts, 20, 4.0, 7, 1024.0)
    lines = line_bins(detections, 20, 1 / (npts // 2 + 1))
    assert k in lines and (np.abs(lines - k) <= 4).all()


def test_excision_weight():
    """Zero within half of a line, a rising Hann edge over the next half, one beyond."""
    weight = excision_weight(np.array([50]), 100, 4)
    assert (weight[46:55] == 0).all()
    assert (weight[:41] == 1).all() and (weight[60:] == 1).all()
    assert (np.diff(weight[54:60]) > 0).all()
    assert np.allclose(weight[41:46], weight[55:60][::-1])
    assert (excision_weight(np.array([], dtype=int), 100, 4) == 1).all()


def test_excised_bins_get_no_weight():
    """An infinite ASD bin: zero whitening weight, and no SNR from it."""
    window = window_from_data(
        {"sample_rate_in_hz": 256, "sample_length_in_s": 4, "padding_length_in_s": 2}
    )
    asd = torch.ones(1, window.n_frequency_bins)
    excised = asd.clone()
    excised[0, 300] = float("inf")
    assert FiducialWhitening(excised, window, "cpu").whitening[0, 300] == 0
    h = torch.ones(1, 1, window.n_frequency_bins, dtype=torch.complex64)
    snr = lambda a: OptimalSNREstimator(a, window, 20.0, "cpu")(h)[0]
    full, cut = snr(asd), snr(excised)
    assert torch.isfinite(cut).all() and (cut < full).all()

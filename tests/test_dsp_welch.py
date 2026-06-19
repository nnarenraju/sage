"""Tests for sage.dsp.welch — ScipyWelch and the pure-torch TorchWelch estimator.

TorchWelch reimplements Welch's method (overlapping windowed segments, density
scaling, mean/median/median-mean averaging with median-bias correction). The
strongest check is that, with matched parameters, it agrees with
``scipy.signal.welch`` to numerical precision; the rest pin the segmentation,
window, averaging and validation branches. CPU-only, no GPU/pycbc/lal.
"""

import numpy as np
import pytest
import scipy.signal as ss
import torch

from sage.dsp.welch import ScipyWelch, TorchWelch


# ── helpers ────────────────────────────────────────────────────────────────
def _white(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64)


def _n_for_segments(seg_len, seg_stride, n_seg):
    return seg_len + (n_seg - 1) * seg_stride


# ── ScipyWelch ─────────────────────────────────────────────────────────────
class TestScipyWelch:
    def test_shapes_and_freqs(self):
        fs = 512.0
        w = ScipyWelch(sample_rate=fs, nperseg_in_seconds=0.5)  # nperseg = 256
        x = np.random.default_rng(0).standard_normal(4096)
        freqs, pxx = w(x)
        assert w.nperseg == 256
        assert freqs.shape == pxx.shape == (256 // 2 + 1,)
        assert freqs[0] == 0.0 and np.isclose(freqs[-1], fs / 2)
        assert (pxx >= 0).all()

    def test_rejects_nonpositive_nperseg(self):
        with pytest.raises(ValueError):
            ScipyWelch(sample_rate=512.0, nperseg_in_seconds=0.0)

    def test_rejects_non_1d(self):
        w = ScipyWelch(sample_rate=512.0, nperseg_in_seconds=0.5)
        with pytest.raises(ValueError):
            w(np.zeros((2, 4096)))

    def test_mean_vs_median_average_both_run(self):
        x = np.random.default_rng(1).standard_normal(4096)
        for avg in ("mean", "median"):
            _, pxx = ScipyWelch(512.0, 0.5, average=avg)(x)
            assert np.isfinite(pxx).all() and (pxx >= 0).all()


# ── TorchWelch: construction / validation ──────────────────────────────────
class TestTorchWelchConstruction:
    def test_freqs_and_delta_f(self):
        tw = TorchWelch(delta_t=1.0 / 512, seg_len=256, seg_stride=128)
        assert tw.freqs.shape == (256 // 2 + 1,)
        assert np.isclose(tw.delta_f, 512.0 / 256)
        assert torch.isclose(tw.freqs[-1], torch.tensor(256.0, dtype=torch.float64))

    def test_hann_window_default(self):
        tw = TorchWelch(seg_len=128)
        assert torch.allclose(tw.window, torch.hann_window(128))

    def test_unknown_window_raises(self):
        with pytest.raises(ValueError):
            TorchWelch(seg_len=128, window="blackman")

    def test_custom_window_tensor(self):
        win = torch.ones(64)
        tw = TorchWelch(seg_len=64, window=win)
        assert torch.equal(tw.window, win)

    def test_window_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            TorchWelch(seg_len=64, window=torch.ones(32))


# ── TorchWelch: call / segmentation / averaging ────────────────────────────
class TestTorchWelchCall:
    def test_output_shape(self):
        seg_len, seg_stride = 256, 128
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride)
        psd = tw(_white(_n_for_segments(seg_len, seg_stride, 8)))
        assert psd.shape == (seg_len // 2 + 1,)
        assert torch.isfinite(psd).all() and (psd >= 0).all()

    def test_rejects_non_1d(self):
        tw = TorchWelch(seg_len=64, seg_stride=32)
        with pytest.raises(ValueError):
            tw(torch.zeros(2, 256))

    def test_minimum_segments_enforced(self):
        seg_len, seg_stride = 128, 64
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride, minimum_segments=20)
        with pytest.raises(ValueError):
            tw(_white(_n_for_segments(seg_len, seg_stride, 5)))

    def test_exact_fit_required_raises_on_misfit(self):
        seg_len, seg_stride = 256, 128
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride, require_exact_data_fit=True)
        # 1300 does not equal (n_seg-1)*stride + seg_len for any integer n_seg
        with pytest.raises(ValueError):
            tw(_white(1300))

    def test_trims_when_not_exact_fit(self):
        seg_len, seg_stride = 256, 128
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride, require_exact_data_fit=False)
        psd = tw(_white(1300))                     # trimmed internally to 1280
        assert psd.shape == (seg_len // 2 + 1,)

    def test_all_avg_methods_run(self):
        seg_len, seg_stride = 128, 64
        x = _white(_n_for_segments(seg_len, seg_stride, 11))
        for avg in ("mean", "median", "median-mean"):
            psd = TorchWelch(1.0 / 512, seg_len, seg_stride, avg_method=avg)(x)
            assert psd.shape == (seg_len // 2 + 1,)
            assert torch.isfinite(psd).all() and (psd >= 0).all()

    def test_unknown_avg_method_raises(self):
        seg_len, seg_stride = 128, 64
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride, avg_method="bogus")
        with pytest.raises(ValueError):
            tw(_white(_n_for_segments(seg_len, seg_stride, 4)))


# ── TorchWelch._median_bias ────────────────────────────────────────────────
class TestMedianBias:
    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            TorchWelch._median_bias(0)

    def test_n_one_is_unity(self):
        assert TorchWelch._median_bias(1) == 1.0

    def test_large_n_tends_to_log2(self):
        assert np.isclose(TorchWelch._median_bias(1000), np.log(2.0))

    def test_matches_scipy_formula(self):
        # scipy.signal._spectral_py._median_bias closed form
        def scipy_bias(n):
            ii_2 = 2 * np.arange(1.0, (n - 1) // 2 + 1)
            return 1 + np.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)
        for n in (3, 5, 9, 50):
            assert np.isclose(TorchWelch._median_bias(n), scipy_bias(n))


# ── TorchWelch vs scipy.signal.welch (the correctness anchor) ──────────────
class TestTorchMatchesScipy:
    def _compare(self, avg, n_seg):
        fs, seg_len, seg_stride = 512.0, 256, 128
        x = _white(_n_for_segments(seg_len, seg_stride, n_seg), seed=3)
        psd_t = TorchWelch(1.0 / fs, seg_len, seg_stride,
                           window="hann", avg_method=avg)(x).numpy()
        f, psd_s = ss.welch(
            x.numpy(), fs=fs, nperseg=seg_len, noverlap=seg_len - seg_stride,
            window="hann", detrend="constant", scaling="density", average=avg,
        )
        assert psd_t.shape == psd_s.shape
        # float32 hann window in torch -> allow a small relative tolerance
        np.testing.assert_allclose(psd_t, psd_s, rtol=2e-3, atol=1e-12)

    def test_matches_scipy_mean(self):
        self._compare("mean", n_seg=10)

    def test_matches_scipy_median_odd_segments(self):
        # odd #segments: torch.median picks the same element scipy's np.median does
        self._compare("median", n_seg=9)

    def test_white_noise_level(self):
        # one-sided PSD of unit-variance white noise is ~2/fs (density scaling)
        fs, seg_len, seg_stride = 512.0, 256, 128
        x = _white(_n_for_segments(seg_len, seg_stride, 60), seed=4)
        psd = TorchWelch(1.0 / fs, seg_len, seg_stride, avg_method="mean")(x)
        assert 0.5 * (2.0 / fs) < psd.median().item() < 2.0 * (2.0 / fs)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

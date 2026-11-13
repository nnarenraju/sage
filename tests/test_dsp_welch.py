"""Tests for sage.dsp.welch: ScipyWelch and TorchWelch, checked against
scipy.signal.welch."""

import numpy as np
import pytest
import scipy.signal as ss
import torch

from sage.dsp.welch import ScipyWelch, TorchWelch


def _white(n, seed=0):
    """Unit-variance float64 white noise of length n."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64)


def _n_for_segments(seg_len, seg_stride, n_seg):
    """Data length that holds exactly n_seg segments."""
    return seg_len + (n_seg - 1) * seg_stride


class TestScipyWelch:
    """Input checks of ScipyWelch."""

    def test_rejects_nonpositive_nperseg(self):
        """A non-positive segment length is refused."""
        with pytest.raises(ValueError):
            ScipyWelch(sample_rate=512.0, nperseg_in_seconds=0.0)

    def test_rejects_non_1d(self):
        """A 2-D input is refused."""
        w = ScipyWelch(sample_rate=512.0, nperseg_in_seconds=0.5)
        with pytest.raises(ValueError):
            w(np.zeros((2, 4096)))


class TestTorchWelchConstruction:
    """Constructor checks of TorchWelch."""


    def test_unknown_window_raises(self):
        """An unknown window name is refused."""
        with pytest.raises(ValueError):
            TorchWelch(seg_len=128, window="blackman")


    def test_window_length_mismatch_raises(self):
        """A window of the wrong length is refused."""
        with pytest.raises(ValueError):
            TorchWelch(seg_len=64, window=torch.ones(32))


class TestTorchWelchCall:
    """Call-time checks of TorchWelch."""

    def test_rejects_non_1d(self):
        """A 2-D input is refused."""
        tw = TorchWelch(seg_len=64, seg_stride=32)
        with pytest.raises(ValueError):
            tw(torch.zeros(2, 256))

    def test_minimum_segments_enforced(self):
        """Fewer segments than minimum_segments are refused."""
        seg_len, seg_stride = 128, 64
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride, minimum_segments=20)
        with pytest.raises(ValueError):
            tw(_white(_n_for_segments(seg_len, seg_stride, 5)))

    def test_exact_fit_required_raises_on_misfit(self):
        """With require_exact_data_fit, a partial segment is refused."""
        seg_len, seg_stride = 256, 128
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride, require_exact_data_fit=True)
        # 1300 is not (n_seg - 1) * stride + seg_len for any n_seg.
        with pytest.raises(ValueError):
            tw(_white(1300))


    def test_unknown_avg_method_raises(self):
        """An unknown averaging method is refused."""
        seg_len, seg_stride = 128, 64
        tw = TorchWelch(1.0 / 512, seg_len, seg_stride, avg_method="bogus")
        with pytest.raises(ValueError):
            tw(_white(_n_for_segments(seg_len, seg_stride, 4)))


class TestMedianBias:
    """Tests of TorchWelch._median_bias."""

    def test_rejects_nonpositive(self):
        """A non-positive segment count is refused."""
        with pytest.raises(ValueError):
            TorchWelch._median_bias(0)

    def test_n_one_is_unity(self):
        """One segment has no median bias."""
        assert TorchWelch._median_bias(1) == 1.0


class TestTorchMatchesScipy:
    """TorchWelch against scipy.signal.welch."""

    def _compare(self, avg, n_seg):
        """Compare with scipy for one averaging method."""
        fs, seg_len, seg_stride = 512.0, 256, 128
        x = _white(_n_for_segments(seg_len, seg_stride, n_seg), seed=3)
        psd_t = TorchWelch(1.0 / fs, seg_len, seg_stride,
                           window="hann", avg_method=avg)(x).numpy()
        f, psd_s = ss.welch(
            x.numpy(), fs=fs, nperseg=seg_len, noverlap=seg_len - seg_stride,
            window="hann", detrend="constant", scaling="density", average=avg,
        )
        assert psd_t.shape == psd_s.shape
        # torch builds the Hann window in float32.
        np.testing.assert_allclose(psd_t, psd_s, rtol=2e-3, atol=1e-12)

    def test_matches_scipy_mean(self):
        """Mean averaging matches scipy."""
        self._compare("mean", n_seg=10)

    def test_matches_scipy_median_odd_segments(self):
        """Median averaging over an odd segment count matches scipy."""
        # With an odd count, torch.median and np.median agree.
        self._compare("median", n_seg=9)

    def test_white_noise_level(self):
        """Unit white noise has a one-sided PSD near 2/fs."""
        fs, seg_len, seg_stride = 512.0, 256, 128
        x = _white(_n_for_segments(seg_len, seg_stride, 60), seed=4)
        psd = TorchWelch(1.0 / fs, seg_len, seg_stride, avg_method="mean")(x)
        assert 0.5 * (2.0 / fs) < psd.median().item() < 2.0 * (2.0 / fs)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

"""Unit tests for sage.dsp.fft."""

import math
import pytest
import torch

from sage.dsp.fft import BatchToFrequencyDomain


_SR = 4096.0
_DELTA_T = 1.0 / _SR


@pytest.fixture
def bfd():
    """A BatchToFrequencyDomain at 4096 Hz."""
    return BatchToFrequencyDomain(delta_t=_DELTA_T)


class TestBatchToFrequencyDomain:
    """Tests of BatchToFrequencyDomain."""


    def test_dimension_error_2d_input(self, bfd):
        """A 2-D input is refused."""
        x = torch.randn(4, 512)
        with pytest.raises(ValueError, match="Expected"):
            bfd(x)

    def test_dimension_error_4d_input(self, bfd):
        """A 4-D input is refused."""
        x = torch.randn(2, 2, 2, 512)
        with pytest.raises(ValueError, match="Expected"):
            bfd(x)

    def test_dc_of_constant_signal(self, bfd):
        """A constant A has a DC bin of A."""
        B, D, T = 1, 1, 256
        A = 3.0
        x = torch.full((B, D, T), A)
        out = bfd(x)
        dc = out[0, 0, 0]
        # With norm='forward', DC = sum / T = A
        assert abs(dc.real - A) < 1e-5
        assert abs(dc.imag) < 1e-5

    def test_known_sinusoid_peak_at_correct_bin(self, bfd):
        """A sinusoid at bin k0 peaks at bin k0."""
        T = 1024
        k0 = 10  # target frequency bin
        t = torch.arange(T, dtype=torch.float32) / _SR
        x = torch.sin(2 * math.pi * k0 / (T / _SR) * t)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        out = bfd(x)
        magnitudes = out[0, 0].abs()
        peak_bin = magnitudes.argmax().item()
        assert peak_bin == k0



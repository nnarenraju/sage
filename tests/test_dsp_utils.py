"""Unit tests for sage.dsp.utils — trim_edges."""

import numpy as np
import pytest

pytest.importorskip("h5py", reason="sage.core.logger requires h5py")

from sage.dsp.utils import trim_edges


class TestTrimEdges:
    def test_correct_length(self):
        # trim=0.5s at fs=10 → remove 5 samples each side from 100
        data = np.ones(100)
        out = trim_edges(data, fs=10.0, trim=0.5)
        assert len(out) == 90

    def test_correct_values(self):
        data = np.arange(50, dtype=float)
        out = trim_edges(data, fs=10.0, trim=0.5)  # n=5
        np.testing.assert_array_equal(out, data[5:45])

    def test_zero_trim_raises(self):
        # trim=0.0 → n=0 → invalid
        with pytest.raises(ValueError):
            trim_edges(np.ones(100), fs=10.0, trim=0.0)

    def test_trim_too_large_raises(self):
        # 2*n >= len(data) → invalid
        with pytest.raises(ValueError):
            trim_edges(np.ones(10), fs=10.0, trim=0.5)  # n=5, 2*5 >= 10

    def test_output_type_preserved(self):
        data = np.zeros(200, dtype=np.float32)
        out = trim_edges(data, fs=100.0, trim=0.1)
        assert out.dtype == np.float32

    def test_symmetry(self):
        # Trim removes the same number of samples from each end
        data = np.arange(100, dtype=float)
        out = trim_edges(data, fs=10.0, trim=1.0)  # n=10
        assert out[0] == 10.0
        assert out[-1] == 89.0

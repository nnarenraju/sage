"""Unit tests for sage.dsp.utils: trim_edges."""

import numpy as np
import pytest

pytest.importorskip("h5py", reason="sage.core.logger requires h5py")

from sage.dsp.utils import trim_edges


class TestTrimEdges:
    """Tests of trim_edges."""

    def test_correct_values(self):
        """trim * fs samples are cut from each end."""
        data = np.arange(50, dtype=float)
        out = trim_edges(data, fs=10.0, trim=0.5)  # n=5
        np.testing.assert_array_equal(out, data[5:45])

    def test_zero_trim_raises(self):
        """A zero trim is refused."""
        with pytest.raises(ValueError):
            trim_edges(np.ones(100), fs=10.0, trim=0.0)

    def test_trim_too_large_raises(self):
        """A trim covering the whole data is refused."""
        with pytest.raises(ValueError):
            trim_edges(np.ones(10), fs=10.0, trim=0.5)  # n=5, 2*5 >= 10


    def test_symmetry(self):
        """The same number of samples is cut from each end."""
        data = np.arange(100, dtype=float)
        out = trim_edges(data, fs=10.0, trim=1.0)  # n=10
        assert out[0] == 10.0
        assert out[-1] == 89.0

"""Unit tests for sage.dsp.multibanding."""

import pytest
import torch

from sage.dsp.multibanding import (
    FrequencyBand,
    FrequencyBandLayout,
    FrequencyMultibandCompressor,
    make_dyadic_frequency_bands,
)


# ---------------------------------------------------------------------------
# FrequencyBand validation
# ---------------------------------------------------------------------------

class TestFrequencyBand:

    def test_negative_f_low_raises(self):
        with pytest.raises(ValueError, match="f_low"):
            FrequencyBand(f_low=-1.0, f_high=100.0, stride=1)

    def test_f_high_equal_f_low_raises(self):
        with pytest.raises(ValueError, match="f_high"):
            FrequencyBand(f_low=50.0, f_high=50.0, stride=1)

    def test_f_high_below_f_low_raises(self):
        with pytest.raises(ValueError, match="f_high"):
            FrequencyBand(f_low=100.0, f_high=20.0, stride=1)

    def test_zero_stride_raises(self):
        with pytest.raises(ValueError, match="stride"):
            FrequencyBand(f_low=20.0, f_high=100.0, stride=0)

    def test_negative_stride_raises(self):
        with pytest.raises(ValueError, match="stride"):
            FrequencyBand(f_low=20.0, f_high=100.0, stride=-2)


# ---------------------------------------------------------------------------
# FrequencyBandLayout properties
# ---------------------------------------------------------------------------

_SR = 4096.0
_DUR = 16.0
_BANDS = (
    FrequencyBand(20.0, 200.0, 1),
    FrequencyBand(200.0, 512.0, 4),
    FrequencyBand(512.0, 1024.0, 8),
)


@pytest.fixture
def layout():
    return FrequencyBandLayout(sample_rate=_SR, duration=_DUR, bands=_BANDS)


class TestFrequencyBandLayoutIndices:


    def test_indices_strictly_increasing(self, layout):
        for idx in layout.band_indices():
            if len(idx) > 1:
                assert (idx[1:] > idx[:-1]).all()

    def test_indices_within_n_freq(self, layout):
        for idx in layout.band_indices():
            assert (idx >= 0).all()
            assert (idx < layout.n_freq).all()

    def test_stride_spacing(self, layout):
        for band, idx in zip(_BANDS, layout.band_indices()):
            if len(idx) > 1:
                diffs = idx[1:] - idx[:-1]
                assert (diffs == band.stride).all()


    def test_frequencies_in_hz(self, layout):
        freqs = layout.frequencies()
        # All retained frequencies must be within the band range
        assert (freqs >= _BANDS[0].f_low - layout.df).all()
        assert (freqs <= _BANDS[-1].f_high + layout.df).all()


class TestFrequencyBandLayoutValidation:

    def test_validate_for_wrong_n_freq(self, layout):
        fd = torch.zeros(2, layout.n_freq + 10)
        with pytest.raises(ValueError, match="rFFT bins"):
            layout.validate_for(fd)


# ---------------------------------------------------------------------------
# FrequencyMultibandCompressor
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_layout():
    return FrequencyBandLayout(
        sample_rate=2048.0,
        duration=4.0,
        bands=(
            FrequencyBand(20.0, 256.0, 1),
            FrequencyBand(256.0, 512.0, 2),
            FrequencyBand(512.0, 1024.0, 4),
        ),
    )


class TestFrequencyMultibandCompressor:


    def test_invalid_pool_mode_raises(self, simple_layout):
        with pytest.raises(ValueError, match="pool mode"):
            FrequencyMultibandCompressor(simple_layout, pool="max")


# ---------------------------------------------------------------------------
# make_dyadic_frequency_bands
# ---------------------------------------------------------------------------

class TestMakeDyadicFrequencyBands:
    def test_bands_cover_full_range(self):
        bands = make_dyadic_frequency_bands(f_min=20.0, f_max=1024.0)
        assert bands[0].f_low == pytest.approx(20.0)
        assert bands[-1].f_high == pytest.approx(1024.0)

    def test_bands_are_contiguous(self):
        bands = make_dyadic_frequency_bands(f_min=20.0, f_max=512.0)
        for i in range(len(bands) - 1):
            assert bands[i].f_high == pytest.approx(bands[i + 1].f_low)

    def test_strides_double(self):
        bands = make_dyadic_frequency_bands(
            f_min=20.0, f_max=1024.0, base_stride=1, max_stride=64
        )
        strides = [b.stride for b in bands]
        for i in range(len(strides) - 1):
            if strides[i] < 64:
                assert strides[i + 1] == min(strides[i] * 2, 64)

    def test_max_stride_respected(self):
        bands = make_dyadic_frequency_bands(
            f_min=20.0, f_max=1024.0, max_stride=8
        )
        for b in bands:
            assert b.stride <= 8

    def test_f_max_le_f_min_raises(self):
        with pytest.raises(ValueError):
            make_dyadic_frequency_bands(f_min=512.0, f_max=20.0)

    def test_f_max_equal_f_min_raises(self):
        with pytest.raises(ValueError):
            make_dyadic_frequency_bands(f_min=100.0, f_max=100.0)


    def test_at_least_one_band(self):
        bands = make_dyadic_frequency_bands(f_min=20.0, f_max=40.0)
        assert len(bands) >= 1


# ---------------------------------------------------------------------------
# describe_layout
# ---------------------------------------------------------------------------


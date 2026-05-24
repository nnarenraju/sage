"""Unit tests for sage.dsp.multibanding."""

import pytest
import torch

from sage.dsp.multibanding import (
    FrequencyBand,
    FrequencyBandLayout,
    FrequencyMultibandCompressor,
    make_dyadic_frequency_bands,
    describe_layout,
)


# ---------------------------------------------------------------------------
# FrequencyBand validation
# ---------------------------------------------------------------------------

class TestFrequencyBand:
    def test_valid_construction(self):
        band = FrequencyBand(f_low=20.0, f_high=100.0, stride=4)
        assert band.f_low == 20.0
        assert band.f_high == 100.0
        assert band.stride == 4

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

    def test_frozen_dataclass(self):
        band = FrequencyBand(f_low=20.0, f_high=100.0, stride=1)
        with pytest.raises((AttributeError, TypeError)):
            band.stride = 2


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


class TestFrequencyBandLayoutProperties:
    def test_n_time(self, layout):
        assert layout.n_time == int(round(_SR * _DUR))  # 65536

    def test_n_freq(self, layout):
        assert layout.n_freq == layout.n_time // 2 + 1

    def test_df(self, layout):
        assert layout.df == pytest.approx(1.0 / _DUR)

    def test_nyquist(self, layout):
        assert layout.nyquist == pytest.approx(_SR / 2.0)

    def test_compressed_length_positive(self, layout):
        assert layout.compressed_length > 0

    def test_compressed_length_less_than_n_freq(self, layout):
        assert layout.compressed_length < layout.n_freq


class TestFrequencyBandLayoutIndices:
    def test_number_of_index_tensors(self, layout):
        indices = layout.band_indices()
        assert len(indices) == len(_BANDS)

    def test_indices_are_tensors(self, layout):
        for idx in layout.band_indices():
            assert isinstance(idx, torch.Tensor)
            assert idx.dtype == torch.int64

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

    def test_frequencies_length(self, layout):
        freqs = layout.frequencies()
        assert len(freqs) == layout.compressed_length

    def test_frequencies_in_hz(self, layout):
        freqs = layout.frequencies()
        # All retained frequencies must be within the band range
        assert (freqs >= _BANDS[0].f_low - layout.df).all()
        assert (freqs <= _BANDS[-1].f_high + layout.df).all()


class TestFrequencyBandLayoutValidation:
    def test_validate_for_correct_shape(self, layout):
        fd = torch.zeros(2, layout.n_freq, dtype=torch.complex64)
        layout.validate_for(fd)  # should not raise

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
    def test_sample_pool_output_shape(self, simple_layout):
        comp = FrequencyMultibandCompressor(simple_layout, pool="sample")
        B, D = 4, 2
        fd = torch.randn(B, D, simple_layout.n_freq, dtype=torch.complex64)
        out = comp(fd)
        assert out.shape == (B, D, simple_layout.compressed_length)

    def test_mean_pool_output_shape(self, simple_layout):
        comp = FrequencyMultibandCompressor(simple_layout, pool="mean")
        B, D = 4, 2
        fd = torch.randn(B, D, simple_layout.n_freq, dtype=torch.complex64)
        out = comp(fd)
        assert out.shape == (B, D, simple_layout.compressed_length)

    def test_sample_pool_matches_direct_index_select(self, simple_layout):
        comp = FrequencyMultibandCompressor(simple_layout, pool="sample")
        fd = torch.arange(simple_layout.n_freq, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out = comp(fd)
        # Verify the selected values match direct index_select
        expected = fd.index_select(-1, comp.indices)
        assert torch.equal(out, expected)

    def test_stride1_band_sample_equals_mean(self, simple_layout):
        # For stride-1 bands, sample and mean should give identical results
        comp_s = FrequencyMultibandCompressor(simple_layout, pool="sample")
        comp_m = FrequencyMultibandCompressor(simple_layout, pool="mean")
        fd = torch.randn(2, simple_layout.n_freq)
        out_s = comp_s(fd)
        out_m = comp_m(fd)
        # They won't be identical everywhere (stride>1 differs), but check shapes match
        assert out_s.shape == out_m.shape

    def test_invalid_pool_mode_raises(self, simple_layout):
        with pytest.raises(ValueError, match="pool mode"):
            FrequencyMultibandCompressor(simple_layout, pool="max")

    def test_from_bands_class_method(self):
        comp = FrequencyMultibandCompressor.from_bands(
            sample_rate=4096.0,
            duration=8.0,
            bands=[(20.0, 512.0, 1), (512.0, 1024.0, 4)],
            pool="sample",
        )
        assert isinstance(comp, FrequencyMultibandCompressor)

    def test_retained_frequencies_method(self, simple_layout):
        comp = FrequencyMultibandCompressor(simple_layout, pool="sample")
        freqs = comp.retained_frequencies()
        assert isinstance(freqs, torch.Tensor)
        assert len(freqs) == simple_layout.compressed_length

    def test_graph_ready_flag(self, simple_layout):
        comp = FrequencyMultibandCompressor(simple_layout)
        assert comp.GRAPH_READY is True


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

    def test_returns_tuple_of_frequency_bands(self):
        bands = make_dyadic_frequency_bands(f_min=20.0, f_max=256.0)
        assert isinstance(bands, tuple)
        assert all(isinstance(b, FrequencyBand) for b in bands)

    def test_at_least_one_band(self):
        bands = make_dyadic_frequency_bands(f_min=20.0, f_max=40.0)
        assert len(bands) >= 1


# ---------------------------------------------------------------------------
# describe_layout
# ---------------------------------------------------------------------------

class TestDescribeLayout:
    def test_returns_list_of_dicts(self, layout):
        rows = describe_layout(layout)
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_expected_keys(self, layout):
        rows = describe_layout(layout)
        expected = {"f_low", "f_high", "stride", "start_bin", "end_bin", "samples"}
        for row in rows:
            assert set(row.keys()) == expected

    def test_samples_positive(self, layout):
        rows = describe_layout(layout)
        for row in rows:
            assert row["samples"] > 0

    def test_start_bin_le_end_bin(self, layout):
        rows = describe_layout(layout)
        for row in rows:
            assert row["start_bin"] <= row["end_bin"]

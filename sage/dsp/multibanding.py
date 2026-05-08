"""Frequency-domain multibanding utilities.

This module compresses uniformly sampled rFFT tensors by keeping a variable
frequency grid.  It is meant as the frequency-domain analogue of time-domain
multirate sampling: keep fine resolution where it matters, and use coarser
frequency spacing elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor


OutputFormat = Literal["channels", "stacked", "complex"]
PoolMode = Literal["sample", "mean"]


@dataclass(frozen=True)
class FrequencyBand:
    """A frequency interval sampled with an integer bin stride.

    Frequencies use half-open intervals ``[f_low, f_high)`` except for the last
    band after conversion to bins, which naturally includes any remaining bins
    up to the requested maximum.
    """

    f_low: float
    f_high: float
    stride: int

    def __post_init__(self) -> None:
        if self.f_low < 0:
            raise ValueError("f_low must be non-negative")
        if self.f_high <= self.f_low:
            raise ValueError("f_high must be greater than f_low")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")


@dataclass(frozen=True)
class FrequencyBandLayout:
    """A concrete multiband layout on an rFFT frequency grid."""

    sample_rate: float
    duration: float
    bands: tuple[FrequencyBand, ...]

    @property
    def n_time(self) -> int:
        return int(round(self.sample_rate * self.duration))

    @property
    def n_freq(self) -> int:
        return self.n_time // 2 + 1

    @property
    def df(self) -> float:
        return 1.0 / self.duration

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    @property
    def compressed_length(self) -> int:
        return sum(len(index) for index in self.band_indices())

    def band_indices(self, device: torch.device | None = None) -> tuple[Tensor, ...]:
        """Return one integer index tensor per band."""

        indices = []
        for band_id, band in enumerate(self.bands):
            start = self.frequency_to_bin(band.f_low, round_up=True)
            end = self.frequency_to_bin(band.f_high, round_up=True)
            if band_id == len(self.bands) - 1:
                end = min(end, self.n_freq)
            start = min(max(start, 0), self.n_freq)
            end = min(max(end, start + 1), self.n_freq)
            indices.append(torch.arange(start, end, band.stride, device=device))
        return tuple(indices)

    def frequencies(self, device: torch.device | None = None) -> Tensor:
        """Return the retained frequencies in Hz."""

        indices = torch.cat(self.band_indices(device=device))
        return indices.to(dtype=torch.float32) * self.df

    def frequency_to_bin(self, frequency: float, *, round_up: bool) -> int:
        raw = frequency / self.df
        return int(torch.ceil(torch.tensor(raw)).item()) if round_up else int(round(raw))

    def validate_for(self, fd: Tensor) -> None:
        if fd.shape[-1] != self.n_freq:
            raise ValueError(
                f"expected {self.n_freq} rFFT bins for {self.sample_rate} Hz and "
                f"{self.duration} s, got {fd.shape[-1]}"
            )
        last = 0.0
        for band in self.bands:
            if band.f_low < last:
                raise ValueError("bands must be sorted and non-overlapping")
            if band.f_high > self.nyquist + self.df:
                raise ValueError("band extends beyond Nyquist")
            last = band.f_high


class FrequencyMultibandCompressor(torch.nn.Module):
    """Compress complex rFFT data with a variable frequency grid.

    Input tensors are expected to have shape ``(..., n_freq)`` and complex dtype.
    For Sage strain batches this is normally ``(batch, detectors, n_freq)``.

    ``pool="sample"`` keeps representative frequency bins exactly.  This is the
    safest first choice for training because it does not average away phase.
    ``pool="mean"`` averages stride-sized chunks inside each band, which is more
    aggressive and can be useful if the model should see smoothed spectra.
    """

    def __init__(
        self,
        layout: FrequencyBandLayout,
        *,
        output_format: OutputFormat = "channels",
        pool: PoolMode = "sample",
    ) -> None:
        super().__init__()
        if output_format not in ("channels", "stacked", "complex"):
            raise ValueError(f"unknown output_format: {output_format}")
        if pool not in ("sample", "mean"):
            raise ValueError(f"unknown pool mode: {pool}")
        self.layout = layout
        self.output_format = output_format
        self.pool = pool

        indices = torch.cat(layout.band_indices())
        self.register_buffer("indices", indices, persistent=False)

    @classmethod
    def from_bands(
        cls,
        sample_rate: float,
        duration: float,
        bands: Sequence[FrequencyBand | tuple[float, float, int]],
        *,
        output_format: OutputFormat = "channels",
        pool: PoolMode = "sample",
    ) -> "FrequencyMultibandCompressor":
        parsed = tuple(
            band if isinstance(band, FrequencyBand) else FrequencyBand(*band)
            for band in bands
        )
        return cls(
            FrequencyBandLayout(sample_rate=sample_rate, duration=duration, bands=parsed),
            output_format=output_format,
            pool=pool,
        )

    def forward(self, fd: Tensor) -> Tensor:
        self.layout.validate_for(fd)
        if not torch.is_complex(fd):
            raise TypeError("FrequencyMultibandCompressor expects a complex rFFT tensor")

        if self.pool == "sample":
            compressed = fd.index_select(-1, self.indices)
        else:
            compressed = self._mean_pool(fd)

        if self.output_format == "complex":
            return compressed
        if self.output_format == "stacked":
            return torch.stack((compressed.real, compressed.imag), dim=-2)
        return torch.cat((compressed.real, compressed.imag), dim=-2)

    def retained_frequencies(self) -> Tensor:
        """Return retained frequencies on the same device as the module."""

        return self.layout.frequencies(device=self.indices.device)

    def _mean_pool(self, fd: Tensor) -> Tensor:
        chunks = []
        for band, indices in zip(self.layout.bands, self.layout.band_indices(fd.device)):
            band_end = min(self.layout.frequency_to_bin(band.f_high, round_up=True), fd.shape[-1])
            windows = []
            for index in indices.tolist():
                end = min(index + band.stride, band_end)
                windows.append(fd[..., index:end].mean(dim=-1))
            chunks.append(torch.stack(windows, dim=-1))
        return torch.cat(chunks, dim=-1)


def make_dyadic_frequency_bands(
    *,
    f_min: float,
    f_max: float,
    base_stride: int = 1,
    max_stride: int = 64,
    first_width: float = 32.0,
) -> tuple[FrequencyBand, ...]:
    """Make bands whose stride doubles with frequency.

    This is intentionally conservative near ``f_min`` and progressively cheaper
    at high frequency.  Pass explicit bands to ``FrequencyMultibandCompressor``
    when you want a layout matched to a PSD, waveform duration, or detector band.
    """

    if f_max <= f_min:
        raise ValueError("f_max must be greater than f_min")
    if base_stride < 1 or max_stride < base_stride:
        raise ValueError("invalid stride range")
    if first_width <= 0:
        raise ValueError("first_width must be positive")

    bands: list[FrequencyBand] = []
    f_low = f_min
    width = first_width
    stride = base_stride
    while f_low < f_max:
        f_high = min(f_low + width, f_max)
        bands.append(FrequencyBand(f_low, f_high, stride))
        f_low = f_high
        width *= 2.0
        stride = min(stride * 2, max_stride)
    return tuple(bands)


def describe_layout(layout: FrequencyBandLayout) -> list[dict[str, float | int]]:
    """Return a compact, notebook-friendly description of a layout."""

    rows: list[dict[str, float | int]] = []
    for band, indices in zip(layout.bands, layout.band_indices()):
        rows.append(
            {
                "f_low": band.f_low,
                "f_high": band.f_high,
                "stride": band.stride,
                "start_bin": int(indices[0].item()),
                "end_bin": int(indices[-1].item()) + 1,
                "samples": len(indices),
            }
        )
    return rows

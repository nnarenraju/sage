"""Frequency-domain multibanding utilities.

This module compresses uniformly sampled rFFT tensors by keeping a variable
frequency grid.  It is meant as the frequency-domain analogue of time-domain
multirate sampling: keep fine resolution where it matters, and use coarser
frequency spacing elsewhere.
"""

from __future__ import annotations

import math
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
        """Return one integer index tensor per band.

        Each index is the start of a complete stride-sized window.  Partial
        trailing windows (where fewer than ``stride`` native bins remain before
        the band boundary) are excluded, matching DINGO's floor-division
        decimation exactly.
        """
        indices = []
        for band_id, band in enumerate(self.bands):
            start = self.frequency_to_bin(band.f_low, round_up=True)
            end = self.frequency_to_bin(band.f_high, round_up=True)
            if band_id == len(self.bands) - 1:
                end = min(end, self.n_freq)
            start = min(max(start, 0), self.n_freq)
            end = min(max(end, start), self.n_freq)
            # Only complete stride-sized windows — matches DINGO's (k2-k1)//stride
            n_complete = (end - start) // band.stride
            indices.append(
                torch.arange(start, start + n_complete * band.stride, band.stride, device=device)
            )
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
            if len(indices) == 0:
                continue
            # band_indices() guarantees all windows are complete; stride is exact.
            windows = [fd[..., int(i):int(i) + band.stride].mean(dim=-1) for i in indices.tolist()]
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
        if len(indices) == 0:
            continue
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


# ---------------------------------------------------------------------------
# Physics-driven band layout
# ---------------------------------------------------------------------------

def _prev_pow2(x: float) -> int:
    """Largest power of 2 that is <= floor(x), minimum 1."""
    n = int(x)
    return 1 if n <= 0 else 1 << (n.bit_length() - 1)


def _first_bin_below_tau(
    m1_kg: float,
    m2_kg: float,
    s1z: float,
    s2z: float,
    tau_threshold: float,
    k_lo: int,
    k_hi: int,
    duration: float,
) -> int:
    """Binary-search for the leftmost bin in [k_lo, k_hi] where τ(f_k) <= tau_threshold.

    τ(f_k) = SimIMRPhenomDChirpTime evaluated at f_k = k / duration, which is
    monotonically decreasing in k.  Returns k_hi + 1 if the condition is never
    met in the search range.
    """
    import lalsimulation as lalsim

    def tau_at(k: int) -> float:
        return lalsim.SimIMRPhenomDChirpTime(m1_kg, m2_kg, s1z, s2z, k / duration)

    if tau_at(k_hi) > tau_threshold:
        return k_hi + 1
    if tau_at(k_lo) <= tau_threshold:
        return k_lo

    while k_lo + 1 < k_hi:
        k_mid = (k_lo + k_hi) // 2
        if tau_at(k_mid) <= tau_threshold:
            k_hi = k_mid
        else:
            k_lo = k_mid

    return k_hi


def make_prior_informed_frequency_bands(
    *,
    m1: float,
    m2: float,
    s1z: float = 0.0,
    s2z: float = 0.0,
    f_min: float,
    f_max: float,
    duration: float,
    n_bins_per_period: int = 32,
    max_stride: int = 128,
) -> tuple[FrequencyBand, ...]:
    """**Method 1 — analytic chirp-time boundary placement.**

    Build a multibanding layout by evaluating the IMRPhenomD chirp time
    τ(f) = SimIMRPhenomDChirpTime analytically for a single worst-case binary.
    Band boundaries are the smallest rFFT bin frequencies at which τ(f) drops
    below the resolution threshold for the next stride level.  Boundaries are
    evaluated exactly on the discrete rFFT grid via binary search — no
    interpolation is performed.

    See also :func:`make_empirical_frequency_bands` (Method 2) which instead
    measures τ(f) empirically from a simulated sample of waveforms.

    The resolution criterion at bin k (f_k = k / duration) is::

        stride(f_k) <= duration / (n_bins_per_period * tau(f_k))

    where tau(f_k) = SimIMRPhenomDChirpTime(f_k) for (m1, m2, s1z, s2z).

    Parameters
    ----------
    m1, m2 : float
        Component masses [M_sun].  Pass the lightest binary in the prior
        (worst case: longest inspiral, finest required resolution everywhere).
    s1z, s2z : float
        Aligned-spin parameters.  Higher aligned spin increases tau(f) at
        every frequency, so pass the prior maximum spin (e.g. 0.99) for the
        most conservative (finest) layout.  This is the same convention as
        TD multirate sampling — both use lightest mass + highest spin as the
        worst case.
    f_min : float
        Lower frequency edge [Hz].
    f_max : float
        Upper frequency edge [Hz] (typically the Nyquist frequency).
    duration : float
        Segment duration [s]; sets native resolution Δf = 1 / duration.
    n_bins_per_period : int
        Minimum multibanded bins per GW signal oscillation period (DINGO: 32).
    max_stride : int
        Maximum stride (power of 2).  The final band runs to f_max at this
        stride regardless of the chirp-time criterion.

    Returns
    -------
    tuple[FrequencyBand, ...]
        Ready to pass to ``FrequencyMultibandCompressor``.  Use
        ``pool="mean"`` to replicate DINGO's bin-averaging decimation.
    """
    import lalsimulation as lalsim

    _MSUN_KG = 1.989e30
    m1_kg = m1 * _MSUN_KG
    m2_kg = m2 * _MSUN_KG

    k_min = int(math.ceil(f_min * duration))
    k_max = int(math.floor(f_max * duration))

    # Correct initial stride so the criterion is already satisfied at f_min.
    tau_0 = lalsim.SimIMRPhenomDChirpTime(m1_kg, m2_kg, s1z, s2z, k_min / duration)
    stride = min(max_stride, _prev_pow2(duration / (n_bins_per_period * tau_0)))

    bands: list[FrequencyBand] = []
    f_band_lo = f_min
    k_current = k_min

    while k_current < k_max:

        if stride >= max_stride:
            bands.append(FrequencyBand(f_band_lo, f_max, stride))
            break

        next_stride = stride * 2
        tau_threshold = duration / (n_bins_per_period * next_stride)

        k_boundary = _first_bin_below_tau(
            m1_kg, m2_kg, s1z, s2z,
            tau_threshold, k_current, k_max, duration,
        )

        if k_boundary > k_max:
            # Criterion for next_stride never met; close out at f_max.
            bands.append(FrequencyBand(f_band_lo, f_max, stride))
            break

        if k_boundary > k_current:
            bands.append(FrequencyBand(f_band_lo, k_boundary / duration, stride))
            f_band_lo = k_boundary / duration

        k_current = k_boundary
        stride = next_stride

    return tuple(bands)


# ---------------------------------------------------------------------------
# Simulation-based band layout (Method 2)
# ---------------------------------------------------------------------------

def make_empirical_frequency_bands(
    *,
    m_min: float,
    m_max: float,
    spin_max: float = 0.99,
    f_min: float,
    f_max: float,
    duration: float,
    n_samples: int = 1000,
    n_bins_per_period: int = 32,
    max_stride: int = 128,
    seed: int = 42,
) -> tuple[FrequencyBand, ...]:
    """**Method 2 — simulation-based boundary placement.**

    Build a multibanding layout by simulating ``n_samples`` FD waveforms drawn
    from the prior, measuring the worst-case local phase gradient |dΨ/df| at
    each rFFT bin, and converting it to an empirical chirp-time envelope
    τ_max(f).  Band boundaries are then placed by the same criterion as
    :func:`make_prior_informed_frequency_bands` (Method 1), but using
    τ_max(f) in place of the analytic SimIMRPhenomDChirpTime.

    This follows the procedure described in the DINGO multibanding paper:
    simulate waveforms from the prior and demand that every oscillation period
    of each signal is covered by at least ``n_bins_per_period`` bins in the
    multibanded domain.  The key difference from Method 1 is that τ_max(f) is
    measured directly from waveform phase rather than from the SPA formula,
    making Method 2 valid for any approximant including precessing or HOM
    models where the analytic chirp time is inaccurate.

    Waveforms are generated with IMRPhenomD (aligned spin) with t_c = 0 so
    that the FD phase Ψ(f) is the pure inspiral chirp phase with no linear
    time-shift term; the phase gradient then directly yields τ(f) without
    a heterodyne step.

    Parameters
    ----------
    m_min, m_max : float
        Component mass range [M_sun].  m1 is drawn from U[m_min, m_max]
        and m2 from U[m_min, m1] (mass ordering enforced).  The analytic
        worst case (m1 = m2 = m_min, s1z = s2z = spin_max) is always
        included as the first sample.
    spin_max : float
        Maximum aligned spin magnitude.  s1z, s2z drawn from
        U[-spin_max, spin_max].  Higher spin increases τ(f); use the prior
        maximum for the most conservative layout.
    f_min, f_max : float
        Frequency range [Hz].
    duration : float
        Segment duration [s]; native resolution Δf = 1 / duration.
    n_samples : int
        Number of IMRPhenomD waveforms to simulate (DINGO paper: 1000).
        Includes the explicit worst-case binary as sample 0.
    n_bins_per_period : int
        Minimum multibanded bins per GW period (DINGO: 32).
    max_stride : int
        Maximum stride (power of 2).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tuple[FrequencyBand, ...]
        Ready to pass to :class:`FrequencyMultibandCompressor` with
        ``pool="mean"``.
    """
    import numpy as np
    from pycbc.waveform import get_fd_waveform

    df = 1.0 / duration
    k_min = int(math.ceil(f_min * duration))
    k_max = int(math.floor(f_max * duration))
    n_intervals = k_max - k_min  # number of adjacent-bin pairs in range

    rng = np.random.default_rng(seed)
    tau_max = np.zeros(n_intervals, dtype=np.float64)

    # Always include the analytic worst case (lightest mass + max spin) so that
    # tau_max is well-defined across [f_min, f_max] even with small n_samples.
    m1_samples = np.concatenate(
        [[m_min], rng.uniform(m_min, m_max, n_samples - 1)]
    )
    m2_samples = np.concatenate(
        [[m_min], np.array([rng.uniform(m_min, m) for m in m1_samples[1:]])]
    )
    s1z_samples = np.concatenate(
        [[spin_max], rng.uniform(-spin_max, spin_max, n_samples - 1)]
    )
    s2z_samples = np.concatenate(
        [[spin_max], rng.uniform(-spin_max, spin_max, n_samples - 1)]
    )

    for m1, m2, s1z, s2z in zip(m1_samples, m2_samples, s1z_samples, s2z_samples):
        try:
            hp, _ = get_fd_waveform(
                approximant="IMRPhenomD",
                mass1=float(m1), mass2=float(m2),
                spin1z=float(s1z), spin2z=float(s2z),
                delta_f=df, f_lower=f_min, f_final=f_max,
                distance=100.0,
            )
        except Exception:
            continue

        h = hp.numpy()
        if k_max + 1 > len(h):
            continue

        h_lo = h[k_min:k_max]
        h_hi = h[k_min + 1:k_max + 1]
        valid = (np.abs(h_lo) > 0) & (np.abs(h_hi) > 0)
        dphi = np.angle(h_hi * np.conj(h_lo))
        # τ(f_k) = |dΨ/df| / (2π) = |Δφ| / (2π Δf) = |Δφ| × duration / (2π)
        tau = np.where(valid, np.abs(dphi) * duration / (2.0 * np.pi), 0.0)
        tau_max = np.maximum(tau_max, tau)

    # Bins where tau_max == 0 (no waveform reached there, i.e. above all merger
    # frequencies) have negligible signal content — leave as 0 so the boundary
    # criterion is trivially satisfied there and max_stride is used.

    tau_0 = tau_max[0] if tau_max[0] > 0 else duration
    stride = min(max_stride, _prev_pow2(duration / (n_bins_per_period * tau_0)))

    bands: list[FrequencyBand] = []
    f_band_lo = f_min
    k_current = k_min

    while k_current < k_max:

        if stride >= max_stride:
            bands.append(FrequencyBand(f_band_lo, f_max, stride))
            break

        next_stride = stride * 2
        tau_threshold = duration / (n_bins_per_period * next_stride)

        # Linear scan for the first interval where τ_max drops below threshold.
        j_start = k_current - k_min
        k_boundary = k_max + 1  # sentinel: condition never met
        for j in range(j_start, n_intervals):
            if tau_max[j] < tau_threshold:
                k_boundary = k_min + j
                break

        if k_boundary > k_max:
            bands.append(FrequencyBand(f_band_lo, f_max, stride))
            break

        if k_boundary > k_current:
            bands.append(FrequencyBand(f_band_lo, k_boundary / duration, stride))
            f_band_lo = k_boundary / duration

        k_current = k_boundary
        stride = next_stride

    return tuple(bands)

"""Frequency-domain heterodyning for GW signals.

Heterodyning removes a reference chirp phase from a waveform, leaving a
slowly-varying residual that can be sampled more coarsely.

    h_het(f) = h(f) × exp(-i φ_ref(f))

where φ_ref(f) = angle(h_ref(f)) is the chirp phase of a chosen reference
binary.  The ``apply_heterodyne`` function mirrors DINGO's
``BaseFrequencyDomain.add_phase`` convention and supports the same data
representations (numpy complex, torch complex, torch real/imag pairs).

For a *search* pipeline the reference binary is not known per-event.  Use
``make_median_reference_binary`` to select a fixed reference from the median
of the prior, then ``compute_reference_phase`` to obtain φ_ref(f).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def apply_heterodyne(
    h: np.ndarray | Tensor,
    phase_ref: np.ndarray | Tensor,
) -> np.ndarray | Tensor:
    """Remove a reference chirp phase from frequency-domain strain data.

    Computes h_het(f) = h(f) × exp(-i phase_ref(f)).  Mirrors DINGO's
    ``BaseFrequencyDomain.add_phase`` convention and supports the same set
    of data representations:

    * ``np.ndarray`` (complex dtype)
    * ``torch.Tensor`` (complex dtype)
    * ``torch.Tensor`` with real/imag encoding (shape ``(..., 2, N_freq)``)

    Parameters
    ----------
    h : np.ndarray or torch.Tensor
        Complex frequency-domain strain, shape ``(..., N_freq)`` for
        complex dtype or ``(..., 2+, N_freq)`` for real/imag encoding.
    phase_ref : np.ndarray or torch.Tensor
        Reference phase φ_ref(f) in radians, shape ``(N_freq,)``.

    Returns
    -------
    Heterodyned strain of the same type and shape as *h*.
    """
    if isinstance(h, np.ndarray):
        if not np.iscomplexobj(h):
            raise TypeError("h must be a complex numpy array")
        return h * np.exp(-1j * np.asarray(phase_ref, dtype=np.float64))

    if not isinstance(h, Tensor):
        raise TypeError(f"Unsupported type {type(h)}")

    phase = torch.as_tensor(phase_ref, dtype=torch.float32, device=h.device)

    if torch.is_complex(h):
        while phase.dim() < h.dim():
            phase = phase[..., None, :]
        return h * torch.exp(torch.tensor(-1j, dtype=h.dtype) * phase)

    # Real/imag encoding: axis -2 carries [re, im, optional extra channels]
    while phase.dim() < h.dim() - 1:
        phase = phase[..., None, :]
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    result = torch.empty_like(h)
    result[..., 0, :] = h[..., 0, :] * cos_p + h[..., 1, :] * sin_p
    result[..., 1, :] = h[..., 1, :] * cos_p - h[..., 0, :] * sin_p
    if h.shape[-2] > 2:
        result[..., 2:, :] = h[..., 2:, :]
    return result


def compute_reference_phase(
    m1: float,
    m2: float,
    s1z: float = 0.0,
    s2z: float = 0.0,
    *,
    sample_rate: float,
    duration: float,
    f_min: float,
    f_max: float,
    approximant: str = "IMRPhenomD",
    distance: float = 100.0,
    coa_phase: float = 0.0,
) -> np.ndarray:
    """Compute the chirp phase of a reference binary on the rFFT grid.

    Generates h_ref(f) via PyCBC and returns φ_ref(f) = angle(h_ref(f)).
    Bins outside [f_min, f_max] or where |h_ref| = 0 are set to 0.

    Parameters
    ----------
    m1, m2 : float
        Component masses [M_sun].
    s1z, s2z : float
        Aligned spin parameters.
    sample_rate : float
        Sample rate [Hz].
    duration : float
        Segment duration [s].
    f_min : float
        Lower frequency cutoff [Hz].
    f_max : float
        Upper frequency cutoff [Hz].
    approximant : str
        Waveform approximant (default IMRPhenomD, fast and accurate for
        aligned-spin binaries).
    distance : float
        Source distance [Mpc].  Affects amplitude only — not the phase.
    coa_phase : float
        Coalescence phase [rad].  With t_c = 0 (PyCBC default) and
        coa_phase = 0 the waveform phase equals the intrinsic SPA chirp
        phase with no additional carrier term.

    Returns
    -------
    np.ndarray, shape (N_freq,)
        Reference phase φ_ref(f) in radians on the rFFT grid.
    """
    from pycbc.waveform import get_fd_waveform

    n_freq = int(round(sample_rate * duration)) // 2 + 1
    delta_f = 1.0 / duration

    hp, _ = get_fd_waveform(
        approximant=approximant,
        mass1=float(m1),
        mass2=float(m2),
        spin1z=float(s1z),
        spin2z=float(s2z),
        delta_f=delta_f,
        f_lower=f_min,
        f_final=f_max,
        distance=float(distance),
        coa_phase=float(coa_phase),
    )
    h = np.array(hp, dtype=np.complex128)
    if len(h) < n_freq:
        h = np.concatenate([h, np.zeros(n_freq - len(h), dtype=np.complex128)])
    h = h[:n_freq]

    # Only assign phase where the waveform is non-zero (inside merger band)
    return np.where(np.abs(h) > 0, np.angle(h), 0.0).astype(np.float64)


def make_median_reference_binary(
    m_min: float,
    m_max: float,
    spin_max: float = 0.99,
    *,
    f_eval: float,
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Find a symmetric zero-spin binary whose chirp time matches the prior median.

    Draws *n_samples* binaries from the flat-in-component-mass prior
    (m1, m2 ∈ [m_min, m_max] with m2 ≤ m1; s1z, s2z ∈ [-spin_max, spin_max])
    and computes the median chirp time τ at *f_eval*.  Then binary-searches
    for the equal-mass, zero-spin binary with the same τ.

    Parameters
    ----------
    m_min, m_max : float
        Component mass range [M_sun].
    spin_max : float
        Maximum aligned spin magnitude.
    f_eval : float
        Frequency [Hz] at which the median chirp time is evaluated.
    n_samples : int
        Number of prior samples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (m_ref, m_ref, 0.0, 0.0)
        Equal-mass, zero-spin reference binary [M_sun].
    """
    import lalsimulation as lalsim

    MSUN_KG = 1.989e30
    rng = np.random.default_rng(seed)

    m1_s = rng.uniform(m_min, m_max, n_samples)
    m2_s = np.array([rng.uniform(m_min, m) for m in m1_s])
    s1z_s = rng.uniform(-spin_max, spin_max, n_samples)
    s2z_s = rng.uniform(-spin_max, spin_max, n_samples)

    taus = np.array([
        lalsim.SimIMRPhenomDChirpTime(
            m1 * MSUN_KG, m2 * MSUN_KG, float(s1z), float(s2z), f_eval
        )
        for m1, m2, s1z, s2z in zip(m1_s, m2_s, s1z_s, s2z_s)
    ])
    tau_med = float(np.median(taus))

    # Binary-search for equal-mass m where τ(m, m, 0, 0) = tau_med.
    # τ decreases with increasing mass.
    lo, hi = m_min, m_max
    for _ in range(60):
        mid = (lo + hi) / 2.0
        t = lalsim.SimIMRPhenomDChirpTime(
            mid * MSUN_KG, mid * MSUN_KG, 0.0, 0.0, f_eval
        )
        if t > tau_med:
            lo = mid   # mass too light → τ too large
        else:
            hi = mid   # mass too heavy → τ too small

    m_ref = (lo + hi) / 2.0
    return m_ref, m_ref, 0.0, 0.0


def residual_chirp_time(
    h_het: np.ndarray,
    duration: float,
    *,
    valid_threshold: float = 0.0,
) -> np.ndarray:
    """Measure the residual chirp time from a heterodyned waveform.

    Computes τ_het(f_k) = |angle(h_het[k+1] × conj(h_het[k]))| × T / (2π)
    which is the local phase-gradient estimate of the remaining chirp rate
    after heterodyning.

    Parameters
    ----------
    h_het : np.ndarray, complex, shape (N_freq,)
        Heterodyned frequency-domain strain.
    duration : float
        Segment duration [s]; sets Δf = 1/T.
    valid_threshold : float
        Bins with |h_het| ≤ valid_threshold are treated as zero.

    Returns
    -------
    np.ndarray, shape (N_freq - 1,)
        Residual chirp time estimate at each native bin boundary.
    """
    h_lo = h_het[:-1]
    h_hi = h_het[1:]
    valid = (np.abs(h_lo) > valid_threshold) & (np.abs(h_hi) > valid_threshold)
    dphi = np.where(valid, np.abs(np.angle(h_hi * np.conj(h_lo))), 0.0)
    return dphi * duration / (2.0 * np.pi)

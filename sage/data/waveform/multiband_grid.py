"""
LAL-equivalent multibanding frequency grid for IMRPhenomX waveforms.

Ports ``XLALSimIMRPhenomXMultibandingGrid`` and its helpers from
``LALSimIMRPhenomXHM_multiband.c`` (lalsuite/lalsimulation/lib/).

Reference: arXiv:2001.10897 (García-Quirós & Husa 2020)
C source  : sage/data/waveform/lalsim_src/LALSimIMRPhenomXHM_multiband.c

Algorithm summary
-----------------
The coarse frequency grid satisfies

    df(f) = dfcoefficient × f^(11/6)            [Eq. 2.8, arXiv:2001.10897]

where the exponent 11/6 matches the PN chirp-rate dφ/df ∝ f^(-11/6).
This gives a coarser grid in the inspiral (slow phase evolution) and a
finer grid near merger (rapid phase evolution).

The full grid is assembled from up to three regions:
  1. Optional pre-grid: uniform at delta_f from f_min up to the frequency
     where the multibanding criterion first triggers.
  2. Inspiral derefinement: a sequence of sub-grids whose df doubles each
     sub-band (factor-of-2 frequency coarsening per sub-band).
  3. Merger + ringdown: separate grids controlled by the Lorentzian/QNM
     structure (only relevant for BBH; BNS mergers are above LIGO band).

The LAL default accuracy threshold is resTest = 1e-3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────
_MSUN_SI = 1.989e30          # kg
_G_SI    = 6.67430e-11       # m^3 kg^-1 s^-2
_C_SI    = 2.99792458e8      # m s^-1
_MTSUN_SI = _G_SI * _MSUN_SI / _C_SI**3  # 1 solar mass in seconds (≈ 4.9255e-6 s)


# ── Sub-grid data structure ───────────────────────────────────────────────
@dataclass
class _SubGrid:
    """One equally-spaced sub-band (mirrors IMRPhenomXMultiBandingGridStruct)."""
    x_start:   float   # start frequency (Mf dimensionless)
    x_end_req: float   # requested end frequency (Mf)
    x_max:     float   # actual end frequency (Mf) = x_start + n*deltax
    deltax:    float   # spacing (Mf)
    n_intervals: int   # number of intervals
    length:    int     # number of points = n_intervals + 1
    int_df_ratio: int  # deltax / evaldMf (integer)


def _grid_comp(f_start: float, f_end: float, mydf: float) -> _SubGrid:
    """
    Build one equally-spaced sub-grid.
    Python equivalent of ``XLALSimIMRPhenomXGridComp``.

    Parameters
    ----------
    f_start, f_end : float
        Start and (requested) end frequency in Mf units.
    mydf : float
        Frequency spacing in Mf units.

    Returns
    -------
    _SubGrid
        Sub-grid spanning [f_start, x_max] with ``n_intervals+1`` points.
    """
    n_intervals = math.ceil((f_end - f_start) / mydf)
    x_max = f_start + mydf * n_intervals
    return _SubGrid(
        x_start      = f_start,
        x_end_req    = f_end,
        x_max        = x_max,
        deltax       = mydf,
        n_intervals  = n_intervals,
        length       = n_intervals + 1,
        int_df_ratio = 0,   # filled in by caller
    )


# ── Merger / ringdown grid spacing ────────────────────────────────────────
def _delta_f_merger_bin(f_damp: float, alpha4: float, res_test: float) -> float:
    """
    Merger bin spacing.  Eq. 2.27 of arXiv:2001.10897.
    C: ``deltaF_mergerBin``.
    """
    aux = math.sqrt(math.sqrt(3.0) * 3.0)
    return 4.0 * f_damp * math.sqrt(res_test / abs(alpha4)) / aux


def _delta_f_ringdown_bin(
    f_damp: float,
    alpha4: float,
    lambda_: float,
    res_test: float,
) -> float:
    """
    Ringdown bin spacing.  Eqs. 2.28, 2.31 of arXiv:2001.10897.
    C: ``deltaF_ringdownBin``.
    """
    df_phase = 5.0 * f_damp * math.sqrt(res_test * 0.5 / abs(alpha4))
    df_amp   = math.sqrt(2.0 * res_test) / abs(lambda_)
    return min(df_phase, df_amp)


# ── df-coefficient for the inspiral power law ────────────────────────────
def _inspiral_df_coefficient(
    eta:      float,
    emm:      int   = 2,
    res_test: float = 1e-3,
) -> float:
    """
    Compute the coefficient in  df(Mf) = dfcoefficient × Mf^(11/6).

    C (line 636 of LALSimIMRPhenomXHM_multiband.c):
        dfcoefficient = 8 * sqrt(3/5) * LAL_PI * PI^(-1/6) * sqrt(2) * cbrt(2)
                        / (cbrt(emm) * emm) * sqrt(resTest * eta)

    Parameters
    ----------
    eta : float
        Symmetric mass ratio η = m1*m2/(m1+m2)².
    emm : int
        Azimuthal mode number (2 for the dominant 22 mode).
    res_test : float
        Multibanding accuracy threshold (LAL default: 1e-3).
    """
    pi          = math.pi
    pi_m_sixth  = pi ** (-1.0 / 6.0)
    cbrt2       = 2.0 ** (1.0 / 3.0)
    cbrt_emm    = float(emm) ** (1.0 / 3.0)

    return (
        8.0 * math.sqrt(3.0 / 5.0)
        * pi * pi_m_sixth
        * math.sqrt(2.0) * cbrt2
        / (cbrt_emm * emm)
        * math.sqrt(res_test * eta)
    )


# ── Main grid builder ─────────────────────────────────────────────────────
def multibanding_grid(
    f_min:    float,
    f_max:    float,
    delta_f:  float,
    m1_msun:  float,
    m2_msun:  float,
    # Merger / ringdown boundary frequencies (Mf dimensionless).
    # For BNS where f_max < f_MECO, these can be left None (no merger/RD grid).
    mf_meco:         float | None = None,
    mf_lorentzian_end: float | None = None,
    mf_max_prime:    float | None = None,
    # For the merger/ringdown grid spacing (22 mode).
    # Required only when mf_meco is provided and f_max > f_meco.
    mf_damp:   float | None = None,  # Mf_DAMP = f_damp * M_total_s
    alpha_l22: float | None = None,  # pPhase22->cLovfda / eta
    gamma2:    float | None = None,  # pAmp22->gamma2
    gamma3:    float | None = None,  # pAmp22->gamma3
    # Grid parameters
    emm:      int   = 2,
    res_test: float = 1e-3,
) -> np.ndarray:
    """
    Build the non-uniform (multibanded) frequency grid in Hz.

    Python equivalent of ``XLALSimIMRPhenomXMultibandingGrid``.

    Parameters
    ----------
    f_min, f_max : float
        Frequency range in Hz.
    delta_f : float
        Target uniform grid spacing in Hz (= 1 / padded_length_in_s).
    m1_msun, m2_msun : float
        Component masses in solar masses.  Used for the Mf ↔ Hz conversion.
    mf_meco : float or None
        MECO (end-of-inspiral) frequency in dimensionless Mf units.
        If None, defaults to Mf corresponding to f_max + 1 Hz (no MECO in band).
    mf_lorentzian_end : float or None
        Start of the merger Lorentzian region (Mf).  Defaults to mf_meco.
    mf_max_prime : float or None
        Maximum frequency for the waveform (Mf).  Defaults to Mf(f_max).
    mf_damp, alpha_l22, gamma2, gamma3 : float or None
        Merger/ringdown parameters needed for the merger and ringdown sub-grid
        spacing.  Only required if the waveform has merger/ringdown in band.
    emm : int
        Azimuthal mode number (2 for the dominant 22 mode).
    res_test : float
        Multibanding accuracy threshold.  LAL default is 1e-3.

    Returns
    -------
    freqs : np.ndarray, shape (N_coarse,)
        Non-uniform frequency grid in Hz.

    Notes
    -----
    Grid is assembled in dimensionless Mf = f × (m1+m2) × G_SI/(c_SI)³ units,
    then converted back to Hz at the end — identical to the LAL convention.
    """
    M_total_s  = (m1_msun + m2_msun) * _MTSUN_SI  # total mass in seconds
    eta        = m1_msun * m2_msun / (m1_msun + m2_msun) ** 2

    # ── Convert inputs to Mf (dimensionless NR units) ────────────────────
    def hz_to_mf(f: float) -> float:
        return f * M_total_s

    eval_dmf  = hz_to_mf(delta_f)          # evaldMf
    mf_start  = hz_to_mf(f_min)            # fstartIn
    mf_fmax   = hz_to_mf(f_max)            # Mfmax

    # Default MECO: for BNS the 3PN MECO is well above the LIGO band.
    # A 1.0+1.0 Msun BNS has f_MECO ≈ 1826 Hz; a 3.0+3.0 Msun BNS ≈ 610 Hz.
    # Using the Newtonian leading-order estimate: f_MECO = c^3/(pi*G*M*6^(3/2)).
    # To be safe we set MfMECO to max(Newtonian estimate, 4×f_max) so the
    # inspiral sub-grids always extend past f_max regardless of mass.
    if mf_meco is None:
        mf_meco_newt = 1.0 / (6.0 ** 1.5 * math.pi)   # Newtonian Mf_MECO
        mf_meco = max(mf_meco_newt, hz_to_mf(f_max * 4.0))
    if mf_lorentzian_end is None:
        # For BNS: ringdown/Lorentzian region starts at fRING >> f_max.
        # Set safely above mf_meco so merger/ringdown grids are never needed.
        mf_lorentzian_end = mf_meco * 2.0
    if mf_max_prime is None:
        mf_max_prime = mf_fmax

    df_power       = 11.0 / 6.0                          # PN chirp-rate exponent
    df_coefficient = _inspiral_df_coefficient(eta, emm, res_test)

    # ── Merger / ringdown grid spacings ───────────────────────────────────
    if (mf_meco <= mf_fmax and mf_damp is not None
            and alpha_l22 is not None and gamma2 is not None and gamma3 is not None):
        df_merger   = _delta_f_merger_bin(mf_damp, alpha_l22, res_test)
        df_ringdown = _delta_f_ringdown_bin(
            mf_damp, alpha_l22,
            gamma2 / (gamma3 * mf_damp),
            res_test,
        )
    else:
        df_merger   = 0.0
        df_ringdown = 0.0

    # ── Determine how many sub-regions are needed ─────────────────────────
    # Ratio of the proposed coarse df at f_start vs the uniform df
    df_ratio = df_coefficient * mf_start ** df_power / eval_dmf

    if df_ratio < 1.0:
        # The multibanding df is finer than the uniform grid at f_min:
        # keep uniform spacing up to the frequency where it first triggers.
        pre_compute_first_grid = True
        int_df_ratio_0         = 1
        df0                    = eval_dmf
        f_end_grid0            = (eval_dmf / df_coefficient) ** (1.0 / df_power)
        f_start_insp_deref     = f_end_grid0 + 2.0 * df0
        df0_orig               = df_coefficient * mf_start ** df_power
    else:
        pre_compute_first_grid = False
        int_df_ratio_0         = int(math.floor(df_ratio))
        df0                    = eval_dmf * int_df_ratio_0
        f_start_insp_deref     = mf_start
        df0_orig               = df_coefficient * mf_start ** df_power

    freq_factor = 2.0 ** (1.0 / df_power)          # ≈ 1.5157 for dfpower=11/6

    if f_start_insp_deref >= mf_meco:
        f_end_insp           = f_start_insp_deref
        n_derefine_inspiral  = 0
    else:
        log_ratio             = math.log(mf_meco / f_start_insp_deref) / math.log(freq_factor)
        n_derefine_inspiral   = math.ceil(log_ratio)
        f_end_insp            = f_start_insp_deref * freq_factor ** n_derefine_inspiral

    # Determine whether merger and ringdown sub-grids are needed
    if f_end_insp + eval_dmf >= mf_lorentzian_end:
        n_merger_grid = 0
        n_rd_grid     = 0 if (f_end_insp + eval_dmf >= mf_max_prime) else 1
    else:
        n_merger_grid = 1
        n_rd_grid     = 0 if mf_lorentzian_end > mf_max_prime else 1

    # ── Assemble sub-grids ────────────────────────────────────────────────
    all_grids: List[_SubGrid] = []
    last_grid = None
    df0_current = df0_orig

    # Pre-grid (uniform section before derefinement)
    if pre_compute_first_grid:
        g = _grid_comp(mf_start, f_end_grid0, eval_dmf)
        g.int_df_ratio = 1
        all_grids.append(g)
        f_start_insp_deref = g.x_max
        df0_current        = 2.0 * df0_orig
        last_grid          = g

    # Inspiral derefinement sub-grids
    if n_derefine_inspiral > 0:
        next_f_start = f_start_insp_deref
        for index in range(n_derefine_inspiral):
            # Compute spacing for this sub-grid
            if df0_current < eval_dmf:
                mydf         = eval_dmf
                int_df_ratio = 1
            else:
                int_df_ratio = int(math.floor(df0_current / eval_dmf))
                mydf         = eval_dmf * int_df_ratio

            if index == 0 and not pre_compute_first_grid:
                f_start_here = next_f_start
            else:
                f_start_here = next_f_start + mydf

            f_end_here = f_start_here * freq_factor

            g = _grid_comp(f_start_here, f_end_here, mydf)
            g.int_df_ratio = int_df_ratio
            all_grids.append(g)
            last_grid    = g

            df0_current  = 2.0 * df0_current
            next_f_start = g.x_max

        f_end_insp = last_grid.x_max

    # Merger sub-grid
    if n_merger_grid > 0 and df_merger > 0:
        df0_current = df_merger
        if last_grid is not None and 2.0 * last_grid.deltax < df_merger:
            df0_current = 2.0 * last_grid.deltax

        if df0_current < eval_dmf:
            mydf         = eval_dmf
            int_df_ratio = 1
        else:
            int_df_ratio = int(math.floor(df0_current / eval_dmf))
            mydf         = eval_dmf * int_df_ratio

        f_start_here = f_end_insp + mydf
        if f_end_insp == mf_start:
            f_start_here = f_end_insp

        if f_start_here > mf_lorentzian_end:
            n_merger_grid = 0
        else:
            g = _grid_comp(f_start_here, mf_lorentzian_end, mydf)
            g.int_df_ratio = int_df_ratio
            all_grids.append(g)
            last_grid    = g
            df0_current  = 2.0 * df0_current

    # Ringdown sub-grid
    if n_rd_grid > 0 and df_ringdown > 0:
        df0_current = df_ringdown
        if df0_current < eval_dmf:
            mydf         = eval_dmf
            int_df_ratio = 1
        else:
            int_df_ratio = int(math.floor(df0_current / eval_dmf))
            mydf         = eval_dmf * int_df_ratio

        f_start_here = last_grid.x_max + mydf
        if last_grid.x_max == mf_start:
            f_start_here = f_end_insp

        if f_start_here > mf_max_prime:
            n_rd_grid = 0
        else:
            g = _grid_comp(f_start_here, mf_max_prime, mydf)
            g.int_df_ratio = int_df_ratio
            all_grids.append(g)

    # ── Convert sub-grids to a flat Hz frequency array ─────────────────────
    freq_segments = []
    for g in all_grids:
        pts = np.arange(g.length, dtype=np.float64) * g.deltax + g.x_start
        freq_segments.append(pts)

    if not freq_segments:
        # Fallback: return single uniform grid (should not normally happen)
        return np.arange(mf_start, mf_fmax + eval_dmf * 0.5, eval_dmf) / M_total_s

    mf_grid = np.concatenate(freq_segments)
    # Remove any duplicate boundary points between adjacent sub-grids
    mf_grid = np.unique(mf_grid)
    # Keep only points in [mf_start, mf_fmax].  In LAL the last sub-grid ends
    # at ceil((Mfmax-fSTART)/mydf)*mydf + fSTART ≥ Mfmax, so Mfmax is always
    # within or at the boundary of the grid.  We replicate this by keeping
    # points up to mf_fmax and then ensuring mf_fmax itself is included.
    mf_grid = mf_grid[(mf_grid >= mf_start - 1e-15) & (mf_grid <= mf_fmax + 1e-15)]
    if len(mf_grid) == 0 or mf_grid[-1] < mf_fmax - eval_dmf * 0.5:
        mf_grid = np.append(mf_grid, mf_fmax)

    return mf_grid / M_total_s   # convert Mf → Hz


# ── Convenience: sub-grid metadata for diagnostics ───────────────────────
def multibanding_grid_info(
    f_min:   float,
    f_max:   float,
    delta_f: float,
    m1_msun: float,
    m2_msun: float,
    res_test: float = 1e-3,
) -> dict:
    """
    Return diagnostic information about the multibanding grid.

    Returns a dict with:
        freqs        : np.ndarray — coarse frequency grid in Hz
        n_coarse     : int        — number of coarse grid points
        n_uniform    : int        — number of points on the uniform grid
        compression  : float      — n_uniform / n_coarse (compression ratio)
        df_at_f_min  : float      — grid spacing at f_min (Hz)
        df_at_f_max  : float      — effective spacing at f_max (Hz)
        eta          : float      — symmetric mass ratio
    """
    freqs    = multibanding_grid(f_min, f_max, delta_f, m1_msun, m2_msun,
                                 res_test=res_test)
    M_total_s = (m1_msun + m2_msun) * _MTSUN_SI
    eta       = m1_msun * m2_msun / (m1_msun + m2_msun) ** 2
    n_unif    = int(round((f_max - f_min) / delta_f)) + 1

    return dict(
        freqs        = freqs,
        n_coarse     = len(freqs),
        n_uniform    = n_unif,
        compression  = n_unif / len(freqs),
        df_at_f_min  = freqs[1] - freqs[0]  if len(freqs) > 1 else delta_f,
        df_at_f_max  = freqs[-1] - freqs[-2] if len(freqs) > 1 else delta_f,
        eta          = eta,
    )

"""
IMRPhenomX multibanding grid, ported from ``LALSimIMRPhenomXHM_multiband.c``
(LALSimulation 6.1.0; arXiv:2001.10897).

Each function reproduces its C counterpart operation for operation (same
arithmetic, order and branches), so with LAL's inputs the results equal LAL's
bit for bit:

    grid_comp            XLALSimIMRPhenomXGridComp
    multibanding_grid    XLALSimIMRPhenomXMultibandingGrid
    merger_bin           deltaF_mergerBin
    ringdown_bin         deltaF_ringdownBin
    df_coefficient       dfcoefficient of IMRPhenomXHMMultiBandOneMode
    grid_arguments_22    the 22-mode grid arguments of IMRPhenomXHMMultiBandOneMode
    coarse_frequencies   its coarse frequency array

A sub-grid is a dict with the fields of ``IMRPhenomXMultiBandingGridStruct``:
``n_intervals``, ``length``, ``int_df_ratio``, ``x_start``, ``x_end_requested``,
``x_end_from_x_start_dx``, ``x_max``, ``deltax``.
"""

import math

import numpy as np

from sage.core.constants import GM

# lengthallGrids: the sub-grids IMRPhenomXHMMultiBandOneMode allocates.
MAX_GRIDS = 20
INT4_MAX = 2**31 - 1


def _int4(value: float) -> int:
    """The C ``(int)`` cast of an already rounded value; outside INT4 it is undefined, so refuse."""
    if not -INT4_MAX - 1 <= value <= INT4_MAX:
        raise ValueError(f"{value!r} overflows LAL's INT4 conversion (undefined in the C)")
    return int(value)


def hz_to_mf(f_hz: float, mtot: float) -> float:
    """XLALSimIMRPhenomXUtilsHztoMf: ``f_hz`` in units of the total mass ``mtot`` (solar masses)."""
    return f_hz * (GM * mtot)


def grid_comp(f_start: float, f_end: float, mydf: float) -> dict:
    """XLALSimIMRPhenomXGridComp: equally spaced sub-grid from ``f_start`` past ``f_end`` (Mf)."""
    delta_f = f_end - f_start
    n_coarse_intervals = _int4(math.ceil(delta_f / mydf))
    max_freq = f_start + mydf * n_coarse_intervals
    return {
        "n_intervals": n_coarse_intervals,
        "x_start": f_start,
        "x_end_requested": f_end,
        "x_end_from_x_start_dx": max_freq,
        "x_max": max_freq,
        "deltax": mydf,
        "length": n_coarse_intervals + 1,
    }


def _spacing(df0_original: float, eval_dmf: float) -> tuple:
    """The (mydf, intdfRatio) choice repeated in XLALSimIMRPhenomXMultibandingGrid."""
    if df0_original < eval_dmf:
        return eval_dmf, 1
    int_df_ratio = _int4(math.floor(df0_original / eval_dmf))
    return eval_dmf * int_df_ratio, int_df_ratio


def multibanding_grid(
    fstart_in: float,
    fend: float,
    mf_lorentzian_end: float,
    mf_max: float,
    eval_dmf: float,
    df_power: float,
    df_coefficient: float,
    df_merger: float,
    df_ringdown: float,
) -> list:
    """
    XLALSimIMRPhenomXMultibandingGrid: the non-uniform coarse frequency grid.

    Parameters
    ----------
    fstart_in : float
        Minimum frequency (Mf).
    fend : float
        End of the inspiral sub-grids (Mf; MECO).
    mf_lorentzian_end : float
        End of the merger sub-grid (Mf).
    mf_max : float
        Maximum frequency (Mf).
    eval_dmf : float
        Spacing of the uniform frequency grid (Mf).
    df_power, df_coefficient : float
        Inspiral spacing law ``df = df_coefficient * f**df_power``.
    df_merger, df_ringdown : float
        Spacing of the merger and ringdown sub-grids (Mf).

    Returns
    -------
    list of dict
        The sub-grids, in frequency order.
    """
    # Before any sub-grid exists the C reads only these two fields of coarseGrid.
    coarse_grid = {"x_max": fstart_in}
    df_ratio = df_coefficient * math.pow(fstart_in, df_power) / eval_dmf
    df0_original = df_ratio * eval_dmf
    coarse_grid["deltax"] = df0_original

    f_end_grid0 = 0.0
    if df_ratio < 1.0:
        pre_compute_first_grid = 1
        f_end_grid0 = math.pow(eval_dmf / df_coefficient, 1.0 / df_power)
        f_start_insp_derefinement = f_end_grid0 + 2 * eval_dmf
    else:
        pre_compute_first_grid = 0
        f_start_insp_derefinement = fstart_in

    frequency_factor = 1.0
    if f_start_insp_derefinement >= fend:
        f_end_insp = f_start_insp_derefinement
        n_derefine_inspiral_grids = 0
    else:
        frequency_factor = math.pow(2.0, 1.0 / df_power)
        orig_log_freq_fact = math.log(fend / f_start_insp_derefinement) / math.log(frequency_factor)
        n_derefine_inspiral_grids = _int4(math.ceil(orig_log_freq_fact))
        f_end_insp = f_start_insp_derefinement * math.pow(frequency_factor, n_derefine_inspiral_grids)

    if f_end_insp + eval_dmf >= mf_lorentzian_end:
        n_merger_grid = 0
        n_ringdown_grid = 0 if f_end_insp + eval_dmf >= mf_max else 1
    else:
        n_merger_grid = 1
        n_ringdown_grid = 0 if mf_lorentzian_end > mf_max else 1

    all_grids = {}
    if pre_compute_first_grid > 0:
        coarse_grid = grid_comp(fstart_in, f_end_grid0, eval_dmf)
        all_grids[0] = {**coarse_grid, "int_df_ratio": 1}
        f_start_insp_derefinement = coarse_grid["x_max"]
        df0_original = 2 * df0_original

    if n_derefine_inspiral_grids > 0:
        next_f_start = f_start_insp_derefinement
        for index in range(n_derefine_inspiral_grids):
            mydf, int_df_ratio = _spacing(df0_original, eval_dmf)
            f_start = next_f_start if index + pre_compute_first_grid == 0 else next_f_start + mydf
            coarse_grid = grid_comp(f_start, f_start * frequency_factor, mydf)
            df0_original = 2 * df0_original
            next_f_start = coarse_grid["x_max"]
            all_grids[index + pre_compute_first_grid] = {**coarse_grid, "int_df_ratio": int_df_ratio}
        f_end_insp = coarse_grid["x_max"]
    elif pre_compute_first_grid > 0:
        f_end_insp = coarse_grid["x_max"]

    if n_merger_grid > 0:
        # The Lorentzian's spacing, unless twice the spacing before the merger is finer.
        df0_original = df_merger
        if 2 * coarse_grid["deltax"] < df_merger:
            df0_original = 2 * coarse_grid["deltax"]
        mydf, int_df_ratio = _spacing(df0_original, eval_dmf)
        f_start = f_end_insp + mydf
        if f_end_insp == fstart_in:
            f_start = f_end_insp
        if f_start > mf_lorentzian_end:
            n_merger_grid = 0
        else:
            coarse_grid = grid_comp(f_start, mf_lorentzian_end, mydf)
            all_grids[pre_compute_first_grid + n_derefine_inspiral_grids] = {**coarse_grid, "int_df_ratio": int_df_ratio}

    if n_ringdown_grid > 0:
        mydf, int_df_ratio = _spacing(df_ringdown, eval_dmf)
        f_start = coarse_grid["x_max"] + mydf
        if coarse_grid["x_max"] == fstart_in:
            f_start = f_end_insp
        if f_start > mf_max:
            n_ringdown_grid = 0
        else:
            coarse_grid = grid_comp(f_start, mf_max, mydf)
            index = pre_compute_first_grid + n_derefine_inspiral_grids + n_merger_grid
            all_grids[index] = {**coarse_grid, "int_df_ratio": int_df_ratio}

    n_grids_used = pre_compute_first_grid + n_derefine_inspiral_grids + n_merger_grid + n_ringdown_grid
    return [all_grids[k] for k in range(n_grids_used)]


def merger_bin(fdamp: float, alpha4: float, abserror: float) -> float:
    """deltaF_mergerBin: merger sub-grid spacing (eq. 2.27 of arXiv:2001.10897)."""
    aux = math.sqrt(math.sqrt(3.0) * 3)
    return 4.0 * fdamp * math.sqrt(abserror / abs(alpha4)) / aux


def ringdown_bin(fdamp: float, alpha4: float, lambda_: float, abserror: float) -> float:
    """deltaF_ringdownBin: ringdown sub-grid spacing (eqs. 2.28, 2.31 of arXiv:2001.10897)."""
    dfphase = 5 * fdamp * math.sqrt(abserror * 0.5 / abs(alpha4))
    dfamp = math.sqrt(2 * abserror) / abs(lambda_)
    if dfphase <= dfamp:
        return dfphase
    return dfamp


def df_coefficient(emm: int, res_test: float, eta: float) -> float:
    """dfcoefficient of IMRPhenomXHMMultiBandOneMode (eqs. 2.8, 2.9 of arXiv:2001.10897)."""
    m_one_sixth = 1.0 / math.pow(math.pi, 1.0 / 6.0)  # powers_of_lalpi.m_one_sixth
    cbrt_emm = float(np.cbrt(float(emm)))
    return (
        8.0 * math.sqrt(3.0 / 5.0) * math.pi * m_one_sixth * math.sqrt(2.0) * float(np.cbrt(2.0))
        / (cbrt_emm * emm) * math.sqrt(res_test * eta)
    )


def grid_arguments_22(
    delta_f: float,
    f_min: float,
    f_max_prime: float,
    mtot: float,
    eta: float,
    f_meco: float,
    f_ring: float,
    f_damp: float,
    c_l_ov_fda: float,
    gamma2: float,
    gamma3: float,
    res_test: float,
) -> dict:
    """
    Arguments of :func:`multibanding_grid` for the 22 mode, as
    IMRPhenomXHMMultiBandOneMode sets them from the waveform struct.

    Parameters
    ----------
    delta_f, f_min, f_max_prime : float
        ``pWF->deltaF``, ``pWF->fMin`` and ``pWF->f_max_prime`` (Hz).
    mtot, eta : float
        ``pWF->Mtot`` (solar masses) and ``pWF->eta``.
    f_meco, f_ring, f_damp : float
        ``pWF->fMECO``, ``pWF->fRING`` and ``pWF->fDAMP`` (Mf).
    c_l_ov_fda : float
        ``pPhase22->cLovfda``.
    gamma2, gamma3 : float
        ``pAmp22->gamma2`` and ``pAmp22->gamma3``.
    res_test : float
        ``PhenomXHMThresholdMband``.

    Returns
    -------
    dict
        Keyword arguments of :func:`multibanding_grid`.
    """
    i_start = int(f_min / delta_f)  # (size_t) truncation
    return {
        "fstart_in": hz_to_mf(i_start * delta_f, mtot),
        "fend": f_meco,
        "mf_lorentzian_end": f_ring + 2 * f_damp,
        "mf_max": hz_to_mf(f_max_prime, mtot),
        "eval_dmf": hz_to_mf(delta_f, mtot),
        "df_power": 11.0 / 6.0,
        "df_coefficient": df_coefficient(2, res_test, eta),
        "df_merger": merger_bin(f_damp, c_l_ov_fda / eta, res_test),
        "df_ringdown": ringdown_bin(f_damp, c_l_ov_fda / eta, gamma2 / (gamma3 * f_damp), res_test),
    }


def coarse_frequencies(grids: list, eval_dmf: float, mf_max: float, mtot: float) -> tuple:
    """
    The coarse frequency array of IMRPhenomXHMMultiBandOneMode: sub-grids up to
    the first that reaches ``mf_max``, the last extended until it does.

    Parameters
    ----------
    grids : list of dict
        Output of :func:`multibanding_grid`.
    eval_dmf, mf_max : float
        Uniform spacing and maximum frequency (Mf).
    mtot : float
        ``pWF->Mtot`` (solar masses).

    Returns
    -------
    used : list of dict
        The sub-grids used, the last one extended.
    points : np.ndarray
        Coarse frequencies in Mf (``IntLawpoints``).
    freqs : np.ndarray
        Coarse frequencies in Hz (``coarseFreqs``).

    Raises
    ------
    ValueError
        If there is no sub-grid, or more than LAL allocates (both undefined in the C).
    """
    if not grids or len(grids) > MAX_GRIDS:
        raise ValueError(
            f"{len(grids)} multibanding sub-grids: IMRPhenomXHMMultiBandOneMode needs 1 to {MAX_GRIDS}"
        )
    used = []
    for grid in grids:
        used.append(dict(grid))
        if grid["x_max"] + eval_dmf >= mf_max:
            break
    while used[-1]["x_max"] < mf_max:
        used[-1]["x_max"] = used[-1]["x_max"] + used[-1]["deltax"]
        used[-1]["length"] = used[-1]["length"] + 1
    points = np.concatenate([g["x_start"] + g["deltax"] * np.arange(g["length"], dtype=np.float64) for g in used])
    divis = 1.0 / (mtot * GM)  # 1 / pWF->M_sec
    return used, points, points * divis

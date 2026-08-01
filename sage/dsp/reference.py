#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : reference.py
Description     : Exact time-frequency tracks and heterodyne reference selection.

    Tools for choosing the SINGLE FIXED heterodyne reference template that
    minimises the worst-case residual bandwidth over a prior — i.e. that
    allows the deepest decimation of the heterodyned baseband.

    Everything here is computed from the EXACT generator waveforms (a one-time
    pre-training calculation): the time-frequency track t(f) is the group
    delay of the exact FD phase (stationary-phase relation), so mass-ratio,
    spin, tidal and all PN effects in the model are included.  Nothing is
    measured off heterodyned time series.

    Long-signal unwrapping: for a segment of length T the raw FD phase can
    only be unwrapped while |t(f)| < T/2; long inspirals alias.  We therefore
    remove the analytic 0PN phase first (an FD heterodyne), unwrap the small
    residual, and add the 0PN group delay back — exact for arbitrarily long
    signals.

    Reference selection (minimax): after heterodyning against a reference R,
    an aligned-merger signal s leaves the residual  f_s(tau) - f_R(tau),
    which is monotone in tau and maximal at the truncation tau_min.  The
    worst case over the prior is therefore set by the extreme instantaneous
    frequencies at tau_min, and the reference that minimises it satisfies

        f_R(tau_min) = ( max_s f_s(tau_min) + min_s f_s(tau_min) ) / 2,

    giving worst-case half-bandwidth B* = (max - min)/2.  The equal-mass
    zero-spin reference chirp mass with that f_R(tau_min) is found by
    bisection on the exact generator.

Created on 2026-07-07

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__status__      = inProgress
"""

import numpy as np
import torch

from sage.core.constants import GM as MSUN_S   # G*MSUN/c^3: one solar mass in seconds


def mchirp(m1, m2):
    """Chirp mass from component masses (same units in, same units out)."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def mc_to_equal_mass(mc):
    """Component mass of the equal-mass binary with chirp mass ``mc``."""
    return mc * 2.0 ** 0.2


def tau_0pn(mc, f):
    """0PN time to merger [s] at GW frequency ``f`` [Hz] for chirp mass ``mc`` [Msun]."""
    return 5.0 / 256.0 * (np.pi * f) ** (-8.0 / 3.0) * (mc * MSUN_S) ** (-5.0 / 3.0)


def f_0pn(mc, tau):
    """0PN GW frequency [Hz] at time-to-merger ``tau`` [s] for chirp mass ``mc`` [Msun]."""
    return (1.0 / np.pi) * (5.0 / (256.0 * tau)) ** (3.0 / 8.0) * (mc * MSUN_S) ** (-5.0 / 8.0)


def merger_frequency(model, theta):
    """
    Physical (tidal) merger frequency [Hz] for one parameter row, from the
    model's own NRTidalv3 merger-frequency fit.

    Parameters
    ----------
    model : IMRPhenomXAS_NRTidalv3
    theta : torch.Tensor, shape (1, 10)
    """
    m1, m2 = float(theta[0, 0]), float(theta[0, 1])
    c1, c2 = float(theta[0, 2]), float(theta[0, 3])
    l1, l2 = float(theta[0, 8]), float(theta[0, 9])
    Mtot = m1 + m2
    one = lambda v: torch.tensor([[v]], dtype=torch.float64)
    Mf = model._merger_freq_v3(one(m1 / Mtot), one(m2 / Mtot),
                               one(l1), one(l2), one(c1), one(c2))
    return float(Mf) / (Mtot * MSUN_S)


def time_frequency_track(h_band, f_grid, mc):
    """
    Exact time-frequency track t(f) [s, arbitrary epoch] from a one-sided FD
    waveform on the signal band, via the group delay of its phase.

    Uses 0PN-assisted unwrapping so that signals longer than the segment do
    not alias: the analytic 0PN phase (chirp mass ``mc``) is removed before
    unwrapping and its group delay added back after differentiation.

    Parameters
    ----------
    h_band : numpy.ndarray, complex, shape (F,)
        FD waveform on the signal band (no DC padding), e.g.
        ``get_hphc(theta, reproduce_lal=True)[0][0, n_pad:]``.
    f_grid : numpy.ndarray, shape (F,)
        Frequencies [Hz] of ``h_band`` (uniform).
    mc : float
        Chirp mass [Msun] of the system (from theta; used only for the
        unwrap assist — the returned track is exact regardless).

    Returns
    -------
    numpy.ndarray, shape (F,)
        t(f) in seconds, up to a constant epoch (monotone increasing over the
        inspiral; merger at the maximum).
    """
    df = float(f_grid[1] - f_grid[0])
    mc_s = mc * MSUN_S
    psi0 = 3.0 / 128.0 * (np.pi * mc_s * f_grid) ** (-5.0 / 3.0)
    # Generator convention: h = A e^{+i phi} with phi ~ -Psi_SPA + const.
    psi_res = np.unwrap(np.angle(h_band * np.exp(1j * psi0)))
    t_res = -np.gradient(psi_res, df) / (2.0 * np.pi)
    tau0 = 5.0 / 256.0 * (np.pi * f_grid) ** (-8.0 / 3.0) * mc_s ** (-5.0 / 3.0)
    return -tau0 + t_res


def invert_track(tf, f_grid, f_merg, taus):
    """Invert a t(f) track to f(tau), capped at the physical merger frequency."""
    i_hi = int(np.searchsorted(f_grid, min(0.98 * f_merg, f_grid[-1])))
    i_peak = int(np.argmax(tf[:i_hi])) if i_hi > 100 else i_hi - 1
    if i_peak < 100:                        # degenerate track: use capped band
        i_peak = i_hi - 1
    tau_f = tf[i_peak] - tf[:i_peak]        # tau(f), decreasing in f
    keep = tau_f > 0
    return np.interp(-np.asarray(taus, dtype=float), -tau_f[keep], f_grid[:i_peak][keep])


def freq_at_tau(model, theta, taus):
    """
    EXACT instantaneous GW frequency [Hz] at times-to-merger ``taus`` [s] for
    one system, from the generator's FD phase (all PN/spin/tidal effects in
    the model included).  One waveform generation per call.

    Parameters
    ----------
    model : IMRPhenomXAS_NRTidalv3
        Built on the analysis grid (its ``self.f``/``n_pad`` are used).
    theta : torch.Tensor, shape (1, 10)
    taus : array-like of float
        Times to merger [s].

    Returns
    -------
    numpy.ndarray, shape (len(taus),)
    """
    f_grid = model.f[0].detach().cpu().numpy()
    hp, _ = model.get_hphc(theta, reproduce_lal=True)
    h_band = hp[0].detach().cpu().numpy()[model.n_pad:]
    mc = mchirp(float(theta[0, 0]), float(theta[0, 1]))
    tf = time_frequency_track(h_band, f_grid, mc)
    return invert_track(tf, f_grid, merger_frequency(model, theta), taus)


def freq_at_tau_batch(model, thetas, taus, progress=None):
    """
    Batched :func:`freq_at_tau`: generates waveforms in chunks of the model's
    build batch size (vectorised ``get_hphc``), then inverts each track.

    Parameters
    ----------
    model : IMRPhenomXAS_NRTidalv3
        Built with batch size ``model.B`` on the analysis grid.
    thetas : torch.Tensor, shape (N, 10)
    taus : array-like of float, length K
    progress : callable or None
        Optional ``progress(done, total)`` callback per chunk.

    Returns
    -------
    numpy.ndarray, shape (N, K)
    """
    f_grid = model.f[0].detach().cpu().numpy()
    N = thetas.shape[0]
    B = model.B
    out = np.empty((N, len(list(taus))), dtype=float)
    for s in range(0, N, B):
        chunk = thetas[s:s + B]
        n = chunk.shape[0]
        if n < B:                            # pad final chunk by repetition
            chunk = torch.cat([chunk, chunk[-1:].expand(B - n, -1)], dim=0)
        hp, _ = model.get_hphc(chunk, reproduce_lal=True)
        hb = hp.detach().cpu().numpy()[:, model.n_pad:]
        for j in range(n):
            th = thetas[s + j:s + j + 1]
            mc = mchirp(float(th[0, 0]), float(th[0, 1]))
            tf = time_frequency_track(hb[j], f_grid, mc)
            out[s + j] = invert_track(tf, f_grid, merger_frequency(model, th), taus)
        if progress is not None:
            progress(min(s + B, N), N)
    return out


def select_reference(model, thetas, tau_min, make_theta=None, tol_hz=0.02,
                     f_trunc=None, max_iter=8):
    """
    Automatically choose the fixed heterodyne reference that minimises the
    worst-case residual bandwidth over a prior (maximises decimation).

    For every prior sample the exact instantaneous frequency at the truncation
    time ``f_i = f(tau_min)`` is computed from the generator.  The minimax
    reference satisfies ``f_R(tau_min) = (max f_i + min f_i)/2``; the
    equal-mass zero-spin reference chirp mass with that property is solved ON
    THE EXACT GENERATOR by a 0PN-scaling fixed-point iteration
    (``Mc <- Mc * (f_exact(Mc)/f_target)^{8/5}``, 2-4 generator calls).

    Parameters
    ----------
    model : IMRPhenomXAS_NRTidalv3
        Built on the analysis grid.
    thetas : torch.Tensor, shape (N, 10)
        Prior samples (include the prior corners for a guaranteed envelope).
    tau_min : float
        Truncation time-to-merger [s]: data at tau < tau_min is discarded.
    make_theta : callable or None
        ``make_theta(m1, m2) -> (1, 10) tensor`` for candidate references.
        Default builds an equal-mass zero-spin zero-tide row at 100 Mpc.
    tol_hz : float
        Solver tolerance on f_R(tau_min).
    f_trunc : numpy.ndarray or None
        Precomputed exact per-sample ``f(tau_min)`` (from :func:`freq_at_tau`).
        Pass this when scanning several ``tau_min`` values so each waveform is
        generated only once.
    max_iter : int
        Maximum fixed-point iterations.

    Returns
    -------
    dict with keys:
        mc_ref, m_ref        : reference chirp mass / equal component mass [Msun]
        f_mid, B             : target midpoint frequency and worst-case
                               half-bandwidth B* [Hz] at tau_min
        f_trunc              : (N,) exact per-sample f(tau_min) [Hz]
        i_lo, i_hi           : indices of the extreme samples
        theta_ref            : (1, 10) tensor of the chosen reference
    """
    if make_theta is None:
        def make_theta(m1, m2):
            return torch.tensor([[m1, m2, 0., 0., 100., 0., 0., 0., 0., 0.]],
                                dtype=torch.float64)

    if f_trunc is None:
        f_trunc = np.array([
            float(freq_at_tau(model, thetas[i:i + 1], [tau_min])[0])
            for i in range(thetas.shape[0])
        ])
    else:
        f_trunc = np.asarray(f_trunc, dtype=float)

    f_lo, f_hi = float(f_trunc.min()), float(f_trunc.max())
    f_mid = 0.5 * (f_lo + f_hi)
    B = 0.5 * (f_hi - f_lo)

    # Solve exact f_ref(tau_min; Mc_ref) = f_mid.  Start from the 0PN inverse,
    # then fixed-point with the 0PN scaling f ~ Mc^{-5/8}.
    mc_ref = ((np.pi * f_mid) ** (-8.0 / 3.0) * 5.0 / (256.0 * tau_min)) ** (3.0 / 5.0) / MSUN_S
    for _ in range(max_iter):
        m = mc_to_equal_mass(mc_ref)
        f_now = float(freq_at_tau(model, make_theta(m, m), [tau_min])[0])
        if abs(f_now - f_mid) < tol_hz:
            break
        mc_ref = mc_ref * (f_now / f_mid) ** (8.0 / 5.0)
    m_ref = mc_to_equal_mass(mc_ref)

    return {
        "mc_ref": mc_ref,
        "m_ref": m_ref,
        "f_mid": f_mid,
        "B": B,
        "f_trunc": f_trunc,
        "i_lo": int(np.argmin(f_trunc)),
        "i_hi": int(np.argmax(f_trunc)),
        "theta_ref": make_theta(m_ref, m_ref),
    }

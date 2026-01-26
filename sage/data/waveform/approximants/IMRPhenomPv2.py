#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomPv2.py
Description     : Short description of the file

Created on 2026-01-21 05:26:04

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


# Packages
import torch

# LOCAL
from sage.core.constants import GM
from sage.core.conversions import mchirp_eta_to_mass1_mass2

from .IMRPhenomD_utils import get_coeffs
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp

from .IMRPhenomPv2_utils import (
    WignerdCoefficients,
    convert_spins,
    ComputeNNLOanglecoeffs,
    SpinWeightedY,
    phP_get_transition_frequencies,
    apply_time_shift_phase_correction,
)

from sage.core.torch import nudge_backward_, nudge_forward_


def PhenomPCoreTwistUp(
    fHz,
    hPhenom,
    eta,
    chi1_l,
    chi2_l,
    chip,
    M,
    angcoeffs,
    Y2m,
    alphaoffset,
    epsilonoffset,
    pv2const,
):
    assert angcoeffs is not None
    assert Y2m is not None

    # here it is used to be LAL_MTSUN_SI
    f = fHz * GM * M  # Frequency in geometric units
    q = (pv2const.ONE + torch.sqrt(pv2const.ONE - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
    # This should prevent NaNs
    nudge_forward_(q, 1.0, 1e-6)
    m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
    m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
    Sperp = chip * (
        m2 * m2
    )  # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp
    # chi_eff = m1 * chi1_l + m2 * chi2_l  # effective spin for M=1

    SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.

    omega = pv2const.PI * f
    logomega = torch.log(omega)
    omega_cbrt = (omega) ** pv2const.ONE_BY_THREE
    omega_cbrt2 = omega_cbrt * omega_cbrt

    alpha = (
        angcoeffs[0] / omega
        + angcoeffs[1] / omega_cbrt2
        + angcoeffs[2] / omega_cbrt
        + angcoeffs[3] * logomega
        + angcoeffs[4] * omega_cbrt
    ) - alphaoffset

    epsilon = (
        angcoeffs[5] / omega
        + angcoeffs[6] / omega_cbrt2
        + angcoeffs[7] / omega_cbrt
        + angcoeffs[8] * logomega
        + angcoeffs[9] * omega_cbrt
    ) - epsilonoffset

    # print("alpha, epsilon: ", alpha, epsilon)
    cBetah, sBetah = WignerdCoefficients(
        omega_cbrt,
        SL,
        eta,
        Sperp,
        pv2const,
    )

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    Y2mA = torch.tensor(
        Y2m, dtype=torch.float64
    )  # need to pass Y2m in a 5-component list
    hp_sum = pv2const.ZERO
    hc_sum = pv2const.ZERO

    cexp_i_alpha = torch.exp(pv2const.ONE_J * alpha)
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    T2m = (
        cexp_2i_alpha * cBetah4 * Y2mA[0]
        - cexp_i_alpha * 2 * cBetah3 * sBetah * Y2mA[1]
        + 1 * pv2const.SQRT_6 * sBetah2 * cBetah2 * Y2mA[2]
        - cexp_mi_alpha * 2 * cBetah * sBetah3 * Y2mA[3]
        + cexp_m2i_alpha * sBetah4 * Y2mA[4]
    )
    Tm2m = (
        cexp_m2i_alpha * sBetah4 * torch.conj(Y2mA[0])
        + cexp_mi_alpha * 2 * cBetah * sBetah3 * torch.conj(Y2mA[1])
        + 1 * pv2const.SQRT_6 * sBetah2 * cBetah2 * torch.conj(Y2mA[2])
        + cexp_i_alpha * 2 * cBetah3 * sBetah * torch.conj(Y2mA[3])
        + cexp_2i_alpha * cBetah4 * torch.conj(Y2mA[4])
    )
    hp_sum = T2m + Tm2m
    hc_sum = pv2const.ONE_J * (T2m - Tm2m)
    eps_phase_hP = torch.exp(-pv2const.TWO_J * epsilon) * hPhenom / 2.0

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc


def PhenomPOneFrequency(
    fs,
    m1,
    m2,
    chi1,
    chi2,
    chip,
    phic,
    M,
    dist_mpc,
    coeffs,
    transition_freqs,
    pv2const,
):
    """
    m1, m2: in solar masses
    phic: Orbital phase at the peak of the underlying non precessing model (rad)
    M: Total mass (Solar masses)
    """
    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    norm = 2.0 * torch.sqrt(pv2const.FIVE / (64.0 * pv2const.PI))
    theta_ripple = torch.tensor([m1, m2, chi1, chi2], dtype=torch.float64)

    phase = PhDPhase(fs, theta_ripple, coeffs, transition_freqs, pv2const)
    # Dphi = lambda f: -PhDPhase(f, theta_ripple, coeffs, transition_freqs, pv2const)

    phase = phase - phic
    Amp = (
        PhDAmp(
            fs, theta_ripple, coeffs, transition_freqs, D=dist_mpc, pv2const=pv2const
        )
        / norm
    )

    # phase -= 2. * phic; # line 1316 ???
    hPhenom = Amp * (torch.exp(-pv2const.ONE_J * phase))
    return hPhenom, phase


def gen_IMRPhenomPv2(fs, theta, f_ref, pv2const):
    """
    Thetas are waveform parameters.
    m1 must be larger than m2.
    """
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phiRef, incl = theta

    # flip m1 m2. For some reason LAL uses this convention for PhenomPv2
    m1, m2 = m2, m1
    s1x, s2x = s2x, s1x
    s1y, s2y = s2y, s1y
    s1z, s2z = s2z, s1z
    # from now on, m1 < m2

    (
        chi1_l,
        chi2_l,
        chip,
        thetaJN,
        alpha0,
        phi_aligned,
        zeta_polariz,
    ) = convert_spins(
        m1,
        m2,
        f_ref,
        phiRef,
        incl,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        pv2const,
    )

    print(
        chi1_l,
        chi2_l,
        chip,
        thetaJN,
        alpha0,
        phi_aligned,
        zeta_polariz,
    )

    phic = 2 * phi_aligned
    q = m2 / m1  # q>=1
    # This should prevents NaNs
    nudge_forward_(q, 1.0, 1e-6)
    M = m1 + m2
    M_s = (m1 + m2) * GM
    chi_eff = (m1 * chi1_l + m2 * chi2_l) / M
    chil = (1.0 + q) / q * chi_eff
    eta = m1 * m2 / (M * M)
    # This should prevents NaNs
    nudge_backward_(eta, 0.25, 1e-6)
    m_sec = M * GM
    piM = torch.pi * m_sec

    omega_ref = piM * f_ref
    logomega_ref = torch.log(omega_ref)
    omega_ref_cbrt = (piM * f_ref) ** (1 / 3)  # == v0
    omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

    # angcoeffs is a torch.stack with the following values in order
    # alphacoeff1, alphacoeff2, alphacoeff3, alphacoeff4, alphacoeff5,
    # epsiloncoeff1, epsiloncoeff2, epsiloncoeff3, epsiloncoeff4, epsiloncoeff5,
    angcoeffs = ComputeNNLOanglecoeffs(
        q,
        chil,
        chip,
        pv2const,
    )

    alphaNNLOoffset = (
        angcoeffs[0] / omega_ref
        + angcoeffs[1] / omega_ref_cbrt2
        + angcoeffs[2] / omega_ref_cbrt
        + angcoeffs[3] * logomega_ref
        + angcoeffs[4] * omega_ref_cbrt
    )

    epsilonNNLOoffset = (
        angcoeffs[5] / omega_ref
        + angcoeffs[6] / omega_ref_cbrt2
        + angcoeffs[7] / omega_ref_cbrt
        + angcoeffs[8] * logomega_ref
        + angcoeffs[9] * omega_ref_cbrt
    )

    Y2m2 = SpinWeightedY(
        thetaJN,
        0,
        -2,
        2,
        -2,
        pv2const,
    )
    Y2m1 = SpinWeightedY(
        thetaJN,
        0,
        -2,
        2,
        -1,
        pv2const,
    )
    Y20 = SpinWeightedY(
        thetaJN,
        0,
        -2,
        2,
        -0,
        pv2const,
    )
    Y21 = SpinWeightedY(
        thetaJN,
        0,
        -2,
        2,
        1,
        pv2const,
    )
    Y22 = SpinWeightedY(
        thetaJN,
        0,
        -2,
        2,
        2,
        pv2const,
    )
    Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]

    # Shift phase so that peak amplitude matches t = 0
    theta_intrinsic = torch.tensor(
        [m2, m1, chi2_l, chi1_l], device=theta.device, dtype=torch.float64
    )
    coeffs = get_coeffs(theta_intrinsic, pv2const)

    transition_freqs = phP_get_transition_frequencies(
        theta_intrinsic,
        coeffs[5],
        coeffs[6],
        chip,
        pv2const,
    )

    hPhenomDs, _ = PhenomPOneFrequency(
        fs,
        m2,
        m1,
        chi2_l,
        chi1_l,
        chip,
        phic,
        M,
        dist_mpc,
        coeffs,
        transition_freqs,
        pv2const,
    )

    hp, hc = PhenomPCoreTwistUp(
        fs,
        hPhenomDs,
        eta,
        chi1_l,
        chi2_l,
        chip,
        M,
        angcoeffs,
        Y2,
        alphaNNLOoffset - alpha0,
        epsilonNNLOoffset,
        pv2const,
    )
    # unpack transition_freqs
    _, _, _, _, f_RD, _ = transition_freqs

    ## ** This is where we do the corrections to phase and time shift **
    # Fixed frequency grid around ringdown frequency for Pv2
    # 10 points should be enough for cubic interpolation
    # Same n_fixed used in LAL version
    n_fixed = 10
    M_s = (theta_intrinsic[0] + theta_intrinsic[1]) * GM
    fcut = pv2const.fM_CUT / M_s
    f_final = f_RD
    freqs_fixed_start = 0.8 * f_final
    freqs_fixed_stop = min(1.2 * f_final, fcut)  # clamp to fCut
    freqs_fixed = torch.linspace(
        freqs_fixed_start,
        freqs_fixed_stop,
        n_fixed,
        device=theta.device,
        dtype=torch.float64,
    )

    # Compute phase on fixed grid
    # We have inverted m1 and m2 back to the convention m1 > m2 for PhenomD call
    phase_fixed = torch.empty(n_fixed, device=fs.device, dtype=torch.float64)

    _, phase_fixed = PhenomPOneFrequency(
        freqs_fixed,
        m2,
        m1,
        chi2_l,
        chi1_l,
        chip,
        phic,
        M,
        dist_mpc,
        coeffs,
        transition_freqs,
        pv2const,
    )

    hp, hc = apply_time_shift_phase_correction(
        hptilde=hp,
        hctilde=hc,
        freqs=fs,
        freqs_fixed=freqs_fixed,
        phase_fixed=phase_fixed,
        f_final=f_final,
        M_s=M_s,
        tc=tc,
        pv2const=pv2const,
    )

    # final touches to hp and hc, stolen from Scott
    c2z = torch.cos(2 * zeta_polariz)
    s2z = torch.sin(2 * zeta_polariz)
    final_hp = c2z * hp + s2z * hc
    final_hc = c2z * hc - s2z * hp

    # Accounting for DC components and zero-padding below f_min
    # Assuming f_ref is f_min
    df = fs[1] - fs[0]
    n_pad = int(f_ref / df) + 1
    hp_pad = torch.zeros(n_pad + hp.numel(), dtype=hp.dtype, device=hp.device)
    hc_pad = torch.zeros_like(hp_pad)

    hp_pad[n_pad:] = final_hp
    hc_pad[n_pad:] = final_hc

    return hp_pad, hc_pad


def gen_IMRPhenomPv2_hphc(f, params, f_ref, pv2const):
    """
    wrapper around gen_Pph but the first two parameters are Mc and eta
    instead of m1 and m2
    """
    Mc = params[0]
    eta = params[1]
    m1, m2 = mchirp_eta_to_mass1_mass2(Mc, eta)
    m1m2params = torch.tensor(
        [
            m1,
            m2,
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
            params[9],
            params[10],
            params[11],
        ],
        device=params.device,
        dtype=torch.float64,
    )

    hp, hc = gen_IMRPhenomPv2(f, m1m2params, f_ref, pv2const)
    return hp, hc

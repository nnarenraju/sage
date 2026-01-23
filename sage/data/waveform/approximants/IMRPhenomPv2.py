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
from sage.core.utils import torch_grad
from sage.core.conversions import mchirp_eta_to_m1_m2

from .IMRPhenomD_utils import get_coeffs
from .IMRPhenomD import Phase as PhDPhase
from .IMRPhenomD import Amp as PhDAmp

from .IMRPhenomPv2_utils import (
    WignerdCoefficients,
    convert_spins,
    ComputeNNLOanglecoeffs,
    SpinWeightedY,
    phP_get_transition_frequencies,
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
):
    assert angcoeffs is not None
    assert Y2m is not None

    # here it is used to be LAL_MTSUN_SI
    f = fHz * GM * M  # Frequency in geometric units
    q = (1.0 + torch.sqrt(1.0 - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
    # This should prevent NaNs
    nudge_forward_(q, 1.0, 1e-6)
    m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
    m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
    Sperp = chip * (
        m2 * m2
    )  # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp
    # chi_eff = m1 * chi1_l + m2 * chi2_l  # effective spin for M=1

    SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.

    omega = torch.pi * f
    logomega = torch.log(omega)
    omega_cbrt = (omega) ** (1 / 3)
    omega_cbrt2 = omega_cbrt * omega_cbrt

    alpha = (
        angcoeffs["alphacoeff1"] / omega
        + angcoeffs["alphacoeff2"] / omega_cbrt2
        + angcoeffs["alphacoeff3"] / omega_cbrt
        + angcoeffs["alphacoeff4"] * logomega
        + angcoeffs["alphacoeff5"] * omega_cbrt
    ) - alphaoffset

    epsilon = (
        angcoeffs["epsiloncoeff1"] / omega
        + angcoeffs["epsiloncoeff2"] / omega_cbrt2
        + angcoeffs["epsiloncoeff3"] / omega_cbrt
        + angcoeffs["epsiloncoeff4"] * logomega
        + angcoeffs["epsiloncoeff5"] * omega_cbrt
    ) - epsilonoffset

    # print("alpha, epsilon: ", alpha, epsilon)
    cBetah, sBetah = WignerdCoefficients(omega_cbrt, SL, eta, Sperp)

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    Y2mA = torch.tensor(Y2m)  # need to pass Y2m in a 5-component list
    hp_sum = 0
    hc_sum = 0

    cexp_i_alpha = torch.exp(1j * alpha)
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    T2m = (
        cexp_2i_alpha * cBetah4 * Y2mA[0]
        - cexp_i_alpha * 2 * cBetah3 * sBetah * Y2mA[1]
        + 1 * torch.sqrt(6) * sBetah2 * cBetah2 * Y2mA[2]
        - cexp_mi_alpha * 2 * cBetah * sBetah3 * Y2mA[3]
        + cexp_m2i_alpha * sBetah4 * Y2mA[4]
    )
    Tm2m = (
        cexp_m2i_alpha * sBetah4 * torch.conj(Y2mA[0])
        + cexp_mi_alpha * 2 * cBetah * sBetah3 * torch.conj(Y2mA[1])
        + 1 * torch.sqrt(6) * sBetah2 * cBetah2 * torch.conj(Y2mA[2])
        + cexp_i_alpha * 2 * cBetah3 * sBetah * torch.conj(Y2mA[3])
        + cexp_2i_alpha * cBetah4 * torch.conj(Y2mA[4])
    )
    hp_sum = T2m + Tm2m
    hc_sum = 1j * (T2m - Tm2m)
    eps_phase_hP = torch.exp(-2j * epsilon) * hPhenom / 2.0

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc


def PhenomPOneFrequency(
    fs, m1, m2, chi1, chi2, chip, phic, M, dist_mpc, coeffs, transition_freqs
):
    """
    m1, m2: in solar masses
    phic: Orbital phase at the peak of the underlying non precessing model (rad)
    M: Total mass (Solar masses)
    """
    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    norm = 2.0 * torch.sqrt(5.0 / (64.0 * torch.pi))
    theta_ripple = torch.array([m1, m2, chi1, chi2])

    phase = PhDPhase(fs, theta_ripple, coeffs, transition_freqs)
    Dphi = lambda f: -PhDPhase(f, theta_ripple, coeffs, transition_freqs)

    phase -= phic
    Amp = PhDAmp(fs, theta_ripple, coeffs, transition_freqs, D=dist_mpc) / norm

    # phase -= 2. * phic; # line 1316 ???
    hPhenom = Amp * (torch.exp(-1j * phase))
    return hPhenom, Dphi


def gen_IMRPhenomPv2(fs, theta, f_ref):
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
    ) = convert_spins(m1, m2, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z)
    phic = 2 * phi_aligned
    q = m2 / m1  # q>=1
    # This should prevents NaNs
    nudge_forward_(q, 1.0, 1e-6)
    M = m1 + m2
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

    angcoeffs = ComputeNNLOanglecoeffs(q, chil, chip)

    alphaNNLOoffset = (
        angcoeffs["alphacoeff1"] / omega_ref
        + angcoeffs["alphacoeff2"] / omega_ref_cbrt2
        + angcoeffs["alphacoeff3"] / omega_ref_cbrt
        + angcoeffs["alphacoeff4"] * logomega_ref
        + angcoeffs["alphacoeff5"] * omega_ref_cbrt
    )

    epsilonNNLOoffset = (
        angcoeffs["epsiloncoeff1"] / omega_ref
        + angcoeffs["epsiloncoeff2"] / omega_ref_cbrt2
        + angcoeffs["epsiloncoeff3"] / omega_ref_cbrt
        + angcoeffs["epsiloncoeff4"] * logomega_ref
        + angcoeffs["epsiloncoeff5"] * omega_ref_cbrt
    )

    Y2m2 = SpinWeightedY(thetaJN, 0, -2, 2, -2)
    Y2m1 = SpinWeightedY(thetaJN, 0, -2, 2, -1)
    Y20 = SpinWeightedY(thetaJN, 0, -2, 2, -0)
    Y21 = SpinWeightedY(thetaJN, 0, -2, 2, 1)
    Y22 = SpinWeightedY(thetaJN, 0, -2, 2, 2)
    Y2 = [Y2m2, Y2m1, Y20, Y21, Y22]

    # Shift phase so that peak amplitude matches t = 0
    theta_intrinsic = torch.tensor([m2, m1, chi2_l, chi1_l])
    coeffs = get_coeffs(theta_intrinsic)

    transition_freqs = phP_get_transition_frequencies(
        theta_intrinsic, coeffs[5], coeffs[6], chip
    )

    hPhenomDs, phi_IIb = PhenomPOneFrequency(
        fs, m2, m1, chi2_l, chi1_l, chip, phic, M, dist_mpc, coeffs, transition_freqs
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
    )
    # unpack transition_freqs
    _, _, _, _, f_RD, _ = transition_freqs

    ## TODO: This is where we do the corrections to phase and time shift
    t0 = torch_grad(phi_IIb, (f_RD,)) / (2 * torch.pi)
    phase_corr = torch.cos(2 * torch.pi * fs * (t0)) - 1j * torch.sin(
        2 * torch.pi * fs * (t0)
    )
    M_s = (m1 + m2) * GM
    phase_corr_tc = torch.exp(-1j * fs * M_s * tc)
    hp *= phase_corr * phase_corr_tc
    hc *= phase_corr * phase_corr_tc

    # final touches to hp and hc, stolen from Scott
    c2z = torch.cos(2 * zeta_polariz)
    s2z = torch.sin(2 * zeta_polariz)
    final_hp = c2z * hp + s2z * hc
    final_hc = c2z * hc - s2z * hp
    return final_hp, final_hc


def gen_IMRPhenomPv2_hphc(f, params, f_ref):
    """
    wrapper around gen_Pph but the first two parameters are Mc and eta
    instead of m1 and m2
    """
    Mc = params[0]
    eta = params[1]
    m1, m2 = mchirp_eta_to_m1_m2(torch.tensor([Mc, eta]))
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
        ]
    )
    hp, hc = gen_IMRPhenomPv2(f, m1m2params, f_ref)
    return hp, hc

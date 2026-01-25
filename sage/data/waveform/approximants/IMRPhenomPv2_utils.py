#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomPv2_utils.py
Description     : Short description of the file

Created on 2026-01-21 05:29:44

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


Modified code from the JAX version on Ripple
Modifications listed below:
    1. Converting all possible jnp functions to torch
    2. Replaced jnp.interp with custom torch interp (see sage.core.utils)
    3. Replaced dict with less readable torch.stack for efficiency
    4. Changed all typing to reflect torch batched operations

"""


# Packages
import torch

from typing import Tuple

# LOCAL
from sage.core.constants import GM
from sage.core.typing import TorchArray
from sage.core.interpolation import torch_linear_interp, torch_natural_cubic_interp

from .IMRPhenomD_QNMdata import (
    QNMData_a,
    QNMData_fRD,
    QNMData_fdamp,
)

from sage.core.torch import nudge_backward_


# helper functions for LALtoPhenomP:
def ROTATEZ(angle, x, y, z):
    # CL: Changed jnp to torch; expanded equation
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    return x * ca - y * sa, x * sa + y * ca, z


def ROTATEY(angle, x, y, z):
    # CL: Changed jnp to torch; expanded equation
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    return x * ca + z * sa, y, -x * sa + z * ca


def EradRational0815_s(eta, s):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    return (
        (
            0.055974469826360077 * eta
            + 0.5809510763115132 * eta2
            - 0.9606726679372312 * eta3
            + 3.352411249771192 * eta4
        )
        * (
            1.0
            + (
                -0.0030302335878845507
                - 2.0066110851351073 * eta
                + 7.7050567802399215 * eta2
            )
            * s
        )
    ) / (
        1.0
        + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta2)
        * s
    )


def EradRational0815(eta, chi1, chi2):
    Seta = torch.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1 * m1
    m2s = m2 * m2
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)

    return EradRational0815_s(eta, s)


def FinalSpin0815_s(eta, S):
    # CL: No changes required
    eta2 = eta * eta
    eta3 = eta2 * eta
    S2 = S * S
    S3 = S2 * S
    return eta * (
        3.4641016151377544
        - 4.399247300629289 * eta
        + 9.397292189321194 * eta2
        - 13.180949901606242 * eta3
        + S
        * (
            (1.0 / eta - 0.0850917821418767 - 5.837029316602263 * eta)
            + (0.1014665242971878 - 2.0967746996832157 * eta) * S
            + (-1.3546806617824356 + 4.108962025369336 * eta) * S2
            + (-0.8676969352555539 + 2.064046835273906 * eta) * S3
        )
    )


def FinalSpin0815(eta, chi1, chi2):
    # CL: Changed jnp to torch; expanded equation
    Seta = torch.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    s = (m1 * m1) * chi1 + (m2 * m2) * chi2
    return FinalSpin0815_s(eta, s)


def convert_spins(
    m1: torch.Tensor,
    m2: torch.Tensor,
    f_ref: torch.Tensor,
    phiRef: torch.Tensor,
    incl: torch.Tensor,
    s1x: torch.Tensor,
    s1y: torch.Tensor,
    s1z: torch.Tensor,
    s2x: torch.Tensor,
    s2y: torch.Tensor,
    s2z: torch.Tensor,
):
    # CL: Changed all jnp to torch equivalent; float to torch.Tensor
    # m1 = m1_SI / MSUN  # Masses in solar masses
    # m2 = m2_SI / MSUN
    M = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    eta = m1 * m2 / (M * M)  # Symmetric mass-ratio
    # This should prevent NaNs
    nudge_backward_(eta, 0.25, 1e-6)

    # From the components in the source frame, we can easily determine
    # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    # We also compute the spherical angles of J,
    # which we need to transform to the J frame

    # Aligned spins
    chi1_l = s1z  # Dimensionless aligned spin on BH 1
    chi2_l = s2z  # Dimensionless aligned spin on BH 2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * torch.sqrt(s1x**2 + s1y**2)
    S2_perp = m2_2 * torch.sqrt(s2x**2 + s2y**2)

    A1 = 2 + (3 * m2) / (2 * m1)
    A2 = 2 + (3 * m1) / (2 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    num = torch.maximum(ASp1, ASp2)
    # Adding this for safety
    # const REAL8 den = (m2 > m1) ? A2*m2_2 : A1*m1_2;
    den = torch.where(m2 > m1, A2 * m2_2, A1 * m1_2)
    chip = num / den

    m_sec = M * GM
    piM = torch.pi * m_sec
    v_ref = (piM * f_ref) ** (1 / 3)
    L0 = M * M * L2PNR(v_ref, eta)
    J0x_sf = m1_2 * s1x + m2_2 * s2x
    J0y_sf = m1_2 * s1y + m2_2 * s2y
    J0z_sf = L0 + m1_2 * s1z + m2_2 * s2z
    J0 = torch.sqrt(J0x_sf * J0x_sf + J0y_sf * J0y_sf + J0z_sf * J0z_sf)

    thetaJ_sf = torch.arccos(J0z_sf / J0)

    phiJ_sf = torch.arctan2(J0y_sf, J0x_sf)

    phi_aligned = -phiJ_sf

    # First we determine kappa
    # in the source frame, the components of N are given in Eq (35c) of T1500606-v6
    Nx_sf = torch.sin(incl) * torch.cos(torch.pi / 2.0 - phiRef)
    Ny_sf = torch.sin(incl) * torch.sin(torch.pi / 2.0 - phiRef)
    Nz_sf = torch.cos(incl)

    tmp_x = Nx_sf
    tmp_y = Ny_sf
    tmp_z = Nz_sf

    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

    kappa = -torch.arctan2(tmp_y, tmp_x)

    # Then we determine alpha0, by rotating LN
    tmp_x, tmp_y, tmp_z = 0, 0, 1
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    alpha0 = torch.arctan2(tmp_y, tmp_x)

    # Finally we determine thetaJ, by rotating N
    tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)
    Nx_Jf, Nz_Jf = tmp_x, tmp_z
    thetaJN = torch.arccos(Nz_Jf)

    # Finally, we need to redefine the polarizations:
    # PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
    # i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ| (see (2.6) in there).
    # By contrast, the triad X,Y,N used in LAL
    # ("waveframe" in the nomenclature of T1500606-v6)
    # is defined in e.g. eq (35) of this document
    # (via its components in the source frame; note we use the defautl Omega=Pi/2).
    # Both triads differ from each other by a rotation around N by an angle \zeta
    # and we need to rotate the polarizations accordingly by 2\zeta

    Xx_sf = -torch.cos(incl) * torch.sin(phiRef)
    Xy_sf = -torch.cos(incl) * torch.cos(phiRef)
    Xz_sf = torch.sin(incl)
    tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    # Now the tmp_a are the components of X in the J frame
    # We need the polar angle of that vector in the P,Q basis of Arun et al
    # P = NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
    PArunx_Jf = 0.0
    PAruny_Jf = -1.0
    PArunz_Jf = 0.0

    # Q = NxP
    QArunx_Jf = Nz_Jf
    QAruny_Jf = 0.0
    QArunz_Jf = -Nx_Jf

    # Calculate the dot products XdotPArun and XdotQArun
    XdotPArun = tmp_x * PArunx_Jf + tmp_y * PAruny_Jf + tmp_z * PArunz_Jf
    XdotQArun = tmp_x * QArunx_Jf + tmp_y * QAruny_Jf + tmp_z * QArunz_Jf

    zeta_polariz = torch.arctan2(XdotQArun, XdotPArun)
    return chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz


def SpinWeightedY(theta, phi, s, l, m):
    "copied from SphericalHarmonics.c in LAL"
    # CL: jnp to torch
    if s == -2:
        if l == 2:
            if m == -2:
                fac = (
                    torch.sqrt(5.0 / (64.0 * torch.pi))
                    * (1.0 - torch.cos(theta))
                    * (1.0 - torch.cos(theta))
                )
            elif m == -1:
                fac = (
                    torch.sqrt(5.0 / (16.0 * torch.pi))
                    * torch.sin(theta)
                    * (1.0 - torch.cos(theta))
                )
            elif m == 0:
                fac = (
                    torch.sqrt(15.0 / (32.0 * torch.pi))
                    * torch.sin(theta)
                    * torch.sin(theta)
                )
            elif m == 1:
                fac = (
                    torch.sqrt(5.0 / (16.0 * torch.pi))
                    * torch.sin(theta)
                    * (1.0 + torch.cos(theta))
                )
            elif m == 2:
                fac = (
                    torch.sqrt(5.0 / (64.0 * torch.pi))
                    * (1.0 + torch.cos(theta))
                    * (1.0 + torch.cos(theta))
                )
            else:
                raise ValueError(f"Invalid mode s={s}, l={l}, m={m} - require |m| <= l")
    return fac * torch.exp(1j * m * phi)


def L2PNR(v: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    # CL: jnp to torch; x**0.5 to torch.sqrt; powers expanded
    eta2 = eta * eta
    x = v * v
    x2 = x * x
    return (
        eta
        * (
            1.0
            + (1.5 + eta / 6.0) * x
            + (3.375 - (19.0 * eta) / 8.0 - eta2 / 24.0) * x2
        )
    ) / torch.sqrt(x)


def WignerdCoefficients(
    v: torch.Tensor, SL: torch.Tensor, eta: torch.Tensor, Sp: torch.Tensor
):
    # CL: jnp to torch; x**0.5 to sqrt; powers expanded
    # We define the shorthand s := Sp / (L + SL)
    L = L2PNR(v, eta)
    s = Sp / (L + SL)
    s2 = s * s
    cos_beta = torch.sqrt(1.0 / (1.0 + s2))
    cos_beta_half = torch.sqrt(((1.0 + cos_beta) / 2.0))  # cos(beta/2)
    sin_beta_half = torch.sqrt(((1.0 - cos_beta) / 2.0))  # sin(beta/2)

    return cos_beta_half, sin_beta_half


def ComputeNNLOanglecoeffs(q, chil, chip):
    m2 = q / (1.0 + q)
    m1 = 1.0 / (1.0 + q)
    dm = m1 - m2
    mtot = 1.0
    eta = m1 * m2  # mtot = 1
    # This should prevent NaNs
    nudge_backward_(eta, 0.25, 1e-6)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    mtot2 = mtot * mtot
    mtot4 = mtot2 * mtot2
    mtot6 = mtot4 * mtot2
    mtot8 = mtot6 * mtot2
    chil2 = chil * chil
    chip2 = chip * chip
    chip4 = chip2 * chip2
    dm2 = dm * dm
    dm3 = dm2 * dm
    m2_2 = m2 * m2
    m2_3 = m2_2 * m2
    m2_4 = m2_3 * m2
    m2_5 = m2_4 * m2
    m2_6 = m2_5 * m2
    m2_7 = m2_6 * m2
    m2_8 = m2_7 * m2

    alphacoeff1 = -0.18229166666666666 - (5 * dm) / (64.0 * m2)

    alphacoeff2 = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
        35 * m2_2 * chil
    ) / (128.0 * mtot2 * eta)

    alphacoeff3 = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (15 * chip2 * dm * m2_3) / (128.0 * mtot4 * eta2)
        - (35 * chip2 * m2_4) / (128.0 * mtot4 * eta2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )

    alphacoeff4 = (
        -(35 * torch.pi) / 48.0
        - (5 * dm * torch.pi) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        - (5 * chip2 * dm * m2_5 * chil) / (128.0 * mtot6 * eta3)
        - (35 * chip2 * m2_6 * chil) / (384.0 * mtot6 * eta3)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )

    alphacoeff5 = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        - (15 * chip4 * dm * m2_7) / (512.0 * mtot8 * eta4)
        - (35 * chip4 * m2_8) / (512.0 * mtot8 * eta4)
        - (485 * chip2 * dm * m2_3) / (14336.0 * mtot4 * eta2)
        + (475 * chip2 * m2_4) / (6144.0 * mtot4 * eta2)
        + (15 * chip2 * dm2 * m2_2) / (256.0 * mtot4 * eta)
        + (145 * chip2 * dm * m2_3) / (512.0 * mtot4 * eta)
        + (575 * chip2 * m2_4) / (1536.0 * mtot4 * eta)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * torch.pi * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * torch.pi * chil) / (16.0 * mtot2 * eta)
        + (15 * chip2 * dm * m2_7 * chil2) / (128.0 * mtot8 * eta4)
        + (35 * chip2 * m2_8 * chil2) / (128.0 * mtot8 * eta4)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )

    epsiloncoeff1 = -0.18229166666666666 - (5 * dm) / (64.0 * m2)
    epsiloncoeff2 = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
        35 * m2_2 * chil
    ) / (128.0 * mtot2 * eta)
    epsiloncoeff3 = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )
    epsiloncoeff4 = (
        -(35 * torch.pi) / 48.0
        - (5 * dm * torch.pi) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )
    epsiloncoeff5 = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * torch.pi * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * torch.pi * chil) / (16.0 * mtot2 * eta)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )

    angcoeffs = torch.stack(
        [
            alphacoeff1,
            alphacoeff2,
            alphacoeff3,
            alphacoeff4,
            alphacoeff5,
            epsiloncoeff1,
            epsiloncoeff2,
            epsiloncoeff3,
            epsiloncoeff4,
            epsiloncoeff5,
        ],
        dim=0,
    ).to(q.device)

    return angcoeffs


def FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip):
    # CL: jnp to torch;
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    # This should prevents NaNs
    nudge_backward_(eta, 0.25, 1e-6)
    # Here I assume m1 > m2, the convention used in phenomD
    # (not the convention of internal phenomP)
    q_factor = m1 / M
    af_parallel = FinalSpin0815(eta, chi1_l, chi2_l)
    Sperp = chip * q_factor * q_factor
    af = torch.copysign(1.0, af_parallel) * torch.sqrt(
        Sperp * Sperp + af_parallel * af_parallel
    )
    return af


def phP_get_fRD_fdamp(m1, m2, chi1_l, chi2_l, chip):
    # CL: jnp.interp to custom torch_interp; see sage.core.utils
    # m1 > m2 should hold here
    finspin = FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip)
    m1_s = m1 * GM
    m2_s = m2 * GM
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s * M_s)
    Erad = EradRational0815(eta_s, chi1_l, chi2_l)
    fRD = torch_linear_interp(finspin, QNMData_a, QNMData_fRD) / (1.0 - Erad)
    fdamp = torch_linear_interp(finspin, QNMData_a, QNMData_fdamp) / (1.0 - Erad)

    return fRD / M_s, fdamp / M_s


def phP_get_transition_frequencies(
    theta,
    gamma2,
    gamma3,
    chip,
):
    # m1 > m2 should hold here
    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    f_RD, f_damp = phP_get_fRD_fdamp(m1, m2, chi1, chi2, chip)

    # Phase transition frequencies
    f1 = 0.018 / (M * GM)
    f2 = 0.5 * f_RD

    # Amplitude transition frequencies
    f3 = 0.014 / (M * GM)
    f4_gammaneg_gtr_1 = torch.abs(f_RD + (-f_damp * gamma3) / gamma2)
    f4_gammaneg_less_1 = torch.abs(
        f_RD + (f_damp * (-1 + torch.sqrt(1 - (gamma2) ** 2.0)) * gamma3) / gamma2
    )

    f4 = torch.where(gamma2 >= 1, f4_gammaneg_gtr_1, f4_gammaneg_less_1)

    return f1, f2, f3, f4, f_RD, f_damp


def apply_time_shift_phase_correction(
    hptilde: torch.Tensor,
    hctilde: torch.Tensor,
    freqs: torch.Tensor,
    freqs_fixed: torch.Tensor,
    phase_fixed: torch.Tensor,
    f_final: float,
    M_s,
    tc,
    offset: int = 0,
):
    """
    Apply time shift correction so the waveform coalesces at t=0.

    Args:
        hptilde: Tensor of shape (n_freq,) with plus polarization.
        hctilde: Tensor of shape (n_freq,) with cross polarization.
        freqs: Tensor of frequencies corresponding to hptilde/hctilde.
        freqs_fixed: Fixed frequency grid used for spline interpolation.
        phase_fixed: Phase values on freqs_fixed.
        f_final: Final frequency (fRD or f_merger) to evaluate derivative.
        offset: Index offset if freqs does not start at zero.
    Returns:
        Tuple of corrected (hptilde, hctilde)
    """

    # Compute derivative of phase at f_final using natural cubic spline
    # torch_natural_cubic_interp returns derivative if deriv=True
    t_corr_fixed = torch_natural_cubic_interp(
        torch.tensor([f_final], device=freqs.device, dtype=freqs.dtype),
        freqs_fixed,
        phase_fixed,
        deriv=True,
    )[0] / (2 * torch.pi)

    # Compute phase correction factor
    # phase_corr = exp(-2.PI.i f t_corr_fixed) = cos(...) - i sin(...)
    # For complex multiplication in PyTorch
    phase_corr = torch.exp(-2j * torch.pi * freqs * t_corr_fixed)

    # This extra correction term can shift the waveform to our desired tc
    # We don't include this for now to try and reproduce LAL closely
    phase_corr_tc = torch.exp(-2j * torch.pi * freqs * tc)

    # Apply to waveform, respecting offset
    hptilde[..., offset : offset + freqs.numel()] *= phase_corr
    hctilde[..., offset : offset + freqs.numel()] *= phase_corr

    return hptilde, hctilde

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomD.py
Description     : Short description of the file

Created on 2026-01-22 10:38:25

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
from sage.core.torch import torch_value_and_grad
from sage.core.torch import nudge_backward_

from sage.data.waveform.approximants import phenom


class IMRPhenomD(phenom.PhenomConstants):

    def __init__(self, f, f_ref):
        # Fixed frequency grid
        self.f = f
        self.f_numel = self.f.numel()
        self.f_ref = f_ref
        # Fix device for all tensors
        self.device = self.f.device

    def __call__(self, theta):
        # Compute derived quantities
        derived = self.compute_derived_parameters(theta)

        # theta = {m1, m2, var2, var3, var4, var5, var6}
        # First four are intrinsic, next 3 are extrinsic
        coeffs = self.get_coeffs(theta[2], theta[3], derived)
        h0 = self.get_h0(theta, coeffs, derived)
        # Compute hp and hc
        # theta[7] is iota
        hp = h0 * (1 / 2 * (1 + torch.cos(theta[7]) ** 2))
        hc = -self.ONE_J * h0 * torch.cos(theta[7])

        return hp, hc

    def compute_derived_parameters(self, theta):
        # Derived parameters are reused a lot; compute once
        # Putting this in self is unfortunately detrimental
        # torch.compile thinks class object is mutating dynamically
        m1_s = theta[0] * self.GM
        m2_s = theta[1] * self.GM
        M_s = m1_s + m2_s
        eta_s = m1_s * m2_s / (M_s * M_s)
        # This should prevents NaNs
        nudge_backward_(eta_s, 0.25, 1e-6)

        return torch.stack([m1_s, m2_s, M_s, eta_s], dim=1)

    def get_coeffs(self, chi1, chi2, derived):
        # Derived quantities used
        eta = derived[:, 3]
        # Definition of chiPN from lalsuite
        chi_s = (chi1 + chi2) / 2.0
        chi_a = (chi1 - chi2) / 2.0
        seta = torch.sqrt(self.ONE - 4 * eta)
        chiPN = chi_s * (self.ONE - 76 * eta / 113) + seta * chi_a

        # chi powers
        chi0 = self.ONE
        chi1 = chiPN - self.ONE
        chi2 = chi1**2
        chi3 = chi1**3

        # eta powers
        eta0 = self.ONE
        eta1 = eta
        eta2 = eta**2

        powers = torch.stack(
            [
                chi0 * eta0,
                chi0 * eta1,
                chi1 * eta0,
                chi1 * eta1,
                chi1 * eta2,
                chi2 * eta0,
                chi2 * eta1,
                chi2 * eta2,
                chi3 * eta0,
                chi3 * eta1,
                chi3 * eta2,
            ],
        )

        # torch stack is compile friendly
        coeff = self.PhenomD_coeff_table @ powers

        return coeff

    def get_h0(self, theta, coeffs, derived):
        ## Shift phase so that peak amplitude matches t = 0
        # Get required derived quantities
        M_s = derived[:, 2]

        # Compute transition frequencies
        # f1, f2, f3, f4, f_RD, f_damp
        trans_fs = self.get_transition_frequencies(theta[:, :4], coeffs[5], coeffs[6])

        # Precomputing required parameters
        f1_Ms = trans_fs[:, 0] * M_s
        f2_Ms = trans_fs[:, 1] * M_s
        f3_Ms = trans_fs[:, 2] * M_s
        f_Ms = self.f * M_s
        fref_Ms = self.f_ref * M_s
        f_RD_Ms = trans_fs[:, 4] * M_s
        f_damp_Ms = trans_fs[:, 5] * M_s
        # Central frequency point (used f_RD and f_damp)
        fmid_Ms = (trans_fs[:, 0] + trans_fs[:, 2]) / 2
        fx_Ms = torch.stack(
            [f_Ms, fref_Ms, f1_Ms, f2_Ms, f3_Ms, f_RD_Ms, f_damp_Ms, fmid_Ms]
        )

        f4_scaled = trans_fs[:, 3] * M_s
        f4_scaled.requires_grad_(True)
        Psi_IIb = IMRPhenomD.get_IIb_raw_phase(f4_scaled, derived[:, 3], coeffs, fx_Ms)
        t0 = torch.autograd.grad(
            Psi_IIb,
            f4_scaled,
            grad_outputs=torch.ones_like(Psi_IIb),
        )[0]

        ## Phase and Amplitude
        # Phase computation
        Psi = self.phase(theta[:, :4], coeffs, derived, f_Ms, fx_Ms, trans_fs[:, 4:])
        Psi_ref = self.phase(
            theta[:, :4], coeffs, derived, fref_Ms, fx_Ms, trans_fs[:, 4:]
        )
        Psi -= t0 * ((f_Ms) - fref_Ms) + Psi_ref
        ext_phase_contrib = self.TWOPI * self.f * theta[:, 4] - 2 * theta[:, 5]
        Psi += ext_phase_contrib

        # And now we can combine them by multiplying by a set of heaviside functions
        fcut_true = self.get_fcut_true(M_s)

        # Get psi and amplitude
        Psi = torch.where(self.f <= fcut_true, Psi, self.TWOPI)

        A = self.amp(
            theta[:, :5],
            coeffs,
            trans_fs,
            derived,
            fx_Ms,
            fcut_true,
        )

        h0 = A * torch.exp(1j * -Psi)
        return h0

    def get_transition_frequencies(self, m1, m2, chi1, chi2, M_s, gamma2, gamma3):
        f_RD, f_damp = self.get_fRD_fdamp(m1, m2, chi1, chi2)

        # Phase transition frequencies
        f1 = 0.018 / M_s
        f2 = 0.5 * f_RD

        # Amplitude transition frequencies
        f3 = 0.014 / M_s
        # Compute both branches
        f4_gammaneg_gtr_1 = torch.abs(f_RD + (-f_damp * gamma3) / gamma2)
        f4_gammaneg_less_1 = torch.abs(
            f_RD + (f_damp * (-1 + torch.sqrt(1 - gamma2**2)) * gamma3) / gamma2
        )

        # Select based on condition
        f4 = torch.where(gamma2 >= 1, f4_gammaneg_gtr_1, f4_gammaneg_less_1)

        return torch.stack([f1, f2, f3, f4, f_RD, f_damp])

    def get_fRD_fdamp(self, chi1, chi2, m1_s, m2_s, M_s, eta_s):
        # Compute Kerr-like total angular momentum
        S = (chi1 * m1_s * m1_s + chi2 * m2_s * m2_s) / (M_s * M_s)
        # Get phenomenological effective spin
        a = IMRPhenomD.final_spin_0815_s(eta_s, S)
        Erad = IMRPhenomD.erad_rational_0815(eta_s, chi1, chi2)

        # Make linear interpolations of frequencies
        # We have already precomputed the slope and intercept
        idx = torch.searchsorted(self.QNMData_a, a) - 1
        idx = idx.clamp(0, len(self.fRD_slope) - 1)

        fRD = self.fRD_slope[idx] * a + self.fRD_intercept[idx]
        fdamp = self.fdamp_slope[idx] * a + self.fdamp_intercept[idx]

        factor = 1.0 / (1.0 - Erad)
        fRD *= factor
        fdamp *= factor

        return fRD / M_s, fdamp / M_s

    @staticmethod
    def final_spin_0815_s(eta, S):
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

    @staticmethod
    def erad_rational_0815(eta, chi1, chi2):
        # Compute the dimensionless mass fractions
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        m1f = 0.5 * (1.0 + Seta)
        m2f = 0.5 * (1.0 - Seta)
        # Compute standard Phenom effective spin combination
        m1sf = m1f * m1f
        m2sf = m2f * m2f
        s = (m1sf * chi1 + m2sf * chi2) / (m1sf + m2sf)

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
            + (
                -0.6714403054720589
                - 1.4756929437702908 * eta
                + 7.304676214885011 * eta2
            )
            * s
        )

    @staticmethod
    def get_phi_IIa(f_Ms, eta, coeffs, beta0, beta1corr):
        return (
            IMRPhenomD.get_IIa_raw_phase(f_Ms, eta, coeffs) + beta1corr * f_Ms + beta0
        )

    @staticmethod
    def get_IIa_raw_phase(self, f_Ms, eta, coeffs):
        phi_IIa_raw = (
            coeffs[11] * f_Ms
            + coeffs[12] * torch.log(f_Ms)
            - coeffs[13] * (f_Ms**-3.0) / 3.0
        ) / eta

        return phi_IIa_raw

    @staticmethod
    def get_IIb_raw_phase(f_Ms, eta, coeffs, fx_Ms):
        phi_IIb_raw = (
            coeffs[14] * f_Ms
            - coeffs[15] * (f_Ms**-1.0)
            + 4.0 * coeffs[16] * (f_Ms ** (3.0 / 4.0)) / 3.0
            + coeffs[17] * torch.arctan((f_Ms - coeffs[18] * fx_Ms[:, 5]) / fx_Ms[:, 6])
        ) / eta

        return phi_IIb_raw

    def phase(self, theta, coeffs, derived, f_Ms, fx_Ms, trans_fs):
        # Get precomputed parameters
        f_RD, f_damp = trans_fs
        _, _, f1_Ms, f2_Ms, _ = fx_Ms

        # Compute inspiral phase
        # Only intrinsic theta has been passed
        phi6corr = self.spin_spin_3pn_correction(*theta, derived[:, 3])
        phi_Ins = self.get_inspiral_phase(
            f_Ms, derived, theta[:, 2:4], coeffs, phi6corr
        )

        # Phase of the late inspiral (region IIa)
        # beta0 is found by matching the phase between the region I and IIa
        # C(1) continuity must be preserved. We therefore need to solve for an
        # additional contribution to beta1
        phi_Ins_f1, dphi_Ins_f1 = torch_value_and_grad(
            self.get_inspiral_phase, (f1_Ms, theta, coeffs)
        )
        phi_IIa_f1, dphi_IIa_f1 = torch_value_and_grad(
            IMRPhenomD.get_IIa_raw_phase, (f1_Ms, derived[:, 3], coeffs)
        )

        beta1corr = dphi_Ins_f1 - dphi_IIa_f1
        beta0 = phi_Ins_f1 - beta1corr * f1_Ms - phi_IIa_f1
        phi_IIa = IMRPhenomD.get_phi_IIa(f_Ms, derived[:, 3], coeffs, beta0, beta1corr)

        # Phase of the merger-ringdown (region IIb)
        phi_IIa_f2, dphi_IIa_f2 = torch_value_and_grad(
            IMRPhenomD.get_phi_IIa, (f2_Ms, derived[:, 3], coeffs, beta0, beta1corr)
        )
        phi_IIb_f2, dphi_IIb_f2 = torch_value_and_grad(
            IMRPhenomD.get_IIb_raw_phase, (f2_Ms, derived[:, 3], coeffs, fx_Ms)
        )

        a1_correction = dphi_IIa_f2 - dphi_IIb_f2
        a0 = phi_IIa_f2 + beta0 - a1_correction * f2_Ms - phi_IIb_f2

        phi_IIb = (
            IMRPhenomD.get_IIb_raw_phase(f_Ms, derived[:, 3], coeffs, fx_Ms)
            + a0
            + a1_correction * f_Ms
        )

        # Combine all regions
        # This is equivalent to the heaviside combine that Ripple had
        # But at frequencies f == f1 and f == f2, we pick one side
        # instead of weighting each side by half.
        # NOTE: Given machine-precision, this should rarely matter
        # Upside, this is faster when vectorised than heaviside
        phase = torch.where(
            f_Ms <= f1_Ms, phi_Ins, torch.where(f_Ms <= f2_Ms, phi_IIa, phi_IIb)
        )

        return phase

    def spin_spin_3pn_correction(self, m1, m2, chi1, chi2, eta_s):
        ## 3PN Spin-Spin Correction from TaylorF2
        # Comments from LALSimIMRPhenomP.c in lalsuite lines 828 - 831
        # // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
        # // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
        # // was not available when PhenomD was tuned.
        # pn->v[6] -= (Subtract3PNSS(m1, m2, M, eta, chi1_l, chi2_l) * pn->v[0]);

        # pn->v[6] corresponds to our phi6 variable
        # pn->v[0] is the leading order coefficient phi0
        # phi0 is simply 1 in the dimensionless PN expansion
        # Subtracting correction before calculating phi_TF2 is sufficient

        # NOTE:
        # The inspiral phase is used to compute the connection coefficients (beta0 and
        # beta1_correction) that enforce C1 continuity between the inspiral to
        # intermediate region. If the 3PN spin–spin term is not subtracted in
        # get_inspiral_phase, phi_Ins will differ from the C implementation, which in
        # turn shifts beta0 and beta1_correction and introduces a phase offset that
        # propagates through the entire waveform (including merger and ringdown).

        # LALSimIMRPhenomD_internals.c; lines 1285 - 1292
        # * Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
        # * (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
        # * was not available when PhenomD was tuned (Subtract3PNSS).

        M = m1 + m2
        m1M = m1 / M
        m2M = m2 / M

        pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta_s) * eta_s * chi1 * chi2

        pn_ss3 = (
            pn_ss3
            + (
                (4703.5 / 8.4 + (2935.0 / 6.0) * m1M - 120.0 * m1M * m1M)
                + (-4108.25 / 6.72 - (108.5 / 1.2) * m1M + (125.5 / 3.6) * m1M * m1M)
            )
            * m1M
            * m1M
            * chi1
            * chi1
        )

        pn_ss3 = (
            pn_ss3
            + (
                (4703.5 / 8.4 + (2935.0 / 6.0) * m2M - 120.0 * m2M * m2M)
                + (-4108.25 / 6.72 - (108.5 / 1.2) * m2M + (125.5 / 3.6) * m2M * m2M)
            )
            * m2M
            * m2M
            * chi2
            * chi2
        )

        return pn_ss3

    def get_inspiral_phase(self, fxi_Ms, derived, chi, coeffs, phi6corr):
        """
        Calculate the inspiral phase for the IMRPhenomD waveform.
        """
        # Expand vars
        m1_s, m2_s, M_s, eta_s = derived
        chi1, chi2 = chi
        # First lets construct the phase in the inspiral (region I)
        m1M = m1_s / M_s
        m2M = m2_s / M_s

        phi0 = self.ONE
        phi1 = self.ZERO
        phi2 = 5.0 * (74.3 / 8.4 + 11.0 * eta_s) / 9.0
        phi3 = -self.SIXTEEN * self.PI + (
            m1M * (25.0 + 38.0 / 3.0 * m1M) * chi1
            + m2M * (25.0 + 38.0 / 3.0 * m2M) * chi2
        )
        phi4 = (
            5.0
            * (3058.673 / 7.056 + 5429.0 / 7.0 * eta_s + 617.0 * eta_s * eta_s)
            / 72.0
        )
        phi4 += (
            (247.0 / 4.8 * eta_s) * chi1 * chi2
            + (-721.0 / 4.8 * eta_s) * chi1 * chi2
            + ((-720.0 / 9.6 * m1M * m1M) + (1.0 / 9.6 * m1M * m1M)) * chi1 * chi1
            + ((-720.0 / 9.6 * m2M * m2M) + (1.0 / 9.6 * m2M * m2M)) * chi2 * chi2
            + ((240.0 / 9.6 * m1M * m1M) + (-7.0 / 9.6 * m1M * m1M)) * chi1 * chi1
            + ((240.0 / 9.6 * m2M * m2M) + (-7.0 / 9.6 * m2M * m2M)) * chi2 * chi2
        )
        phi5 = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta_s) * self.PI
        phi5 += (
            -m1M
            * (
                1391.5 / 8.4
                - m1M * (1.0 - m1M) * 10.0 / 3.0
                + m1M * (1276.0 / 8.1 + m1M * (1.0 - m1M) * 170.0 / 9.0)
            )
        ) * chi1 + (
            -m2M
            * (
                1391.5 / 8.4
                - m2M * (1.0 - m2M) * 10.0 / 3.0
                + m2M * (1276.0 / 8.1 + m2M * (1.0 - m2M) * 170.0 / 9.0)
            )
        ) * chi2
        phi5_log = (5.0 / 3.0) * (772.9 / 8.4 - 13.0 * eta_s) * self.PI
        phi5_log += 3.0 * (
            (
                -m1M
                * (
                    1391.5 / 8.4
                    - m1M * (1.0 - m1M) * 10.0 / 3.0
                    + m1M * (1276.0 / 8.1 + m1M * (1.0 - m1M) * 170.0 / 9.0)
                )
            )
            * chi1
            + (
                -m2M
                * (
                    1391.5 / 8.4
                    - m2M * (1.0 - m2M) * 10.0 / 3.0
                    + m2M * (1276.0 / 8.1 + m2M * (1.0 - m2M) * 170.0 / 9.0)
                )
            )
            * chi2
        )

        phi6 = (
            (
                11583.231236531 / 4.694215680
                - 640.0 / 3.0 * self.PI * self.PI
                - 684.8 / 2.1 * self.EulerGamma
            )
            + eta_s * (-15737.765635 / 3.048192 + 225.5 / 1.2 * self.PI * self.PI)
            + eta_s * eta_s * 76.055 / 1.728
            - eta_s * eta_s * eta_s * 127.825 / 1.296
            + (-684.8 / 2.1) * torch.log(self.FOUR)
        )
        phi6 += (self.PI * m1M * (1490.0 / 3.0 + m1M * 260.0)) * chi1 + (
            self.PI * m2M * (1490.0 / 3.0 + m2M * 260.0)
        ) * chi2

        # Applying the 3PN spin-spin correction
        phi6 = phi6 - phi6corr

        phi6_log = -684.8 / 2.1

        phi7 = self.PI * (
            770.96675 / 2.54016
            + 378.515 / 1.512 * eta_s
            - 740.45 / 7.56 * eta_s * eta_s
        )
        phi7 += (
            m1M
            * (
                -17097.8035 / 4.8384
                + eta_s * 28764.25 / 6.72
                + eta_s * eta_s * 47.35 / 1.44
                + m1M
                * (
                    -7189.233785 / 1.524096
                    + eta_s * 458.555 / 3.024
                    - eta_s * eta_s * 534.5 / 7.2
                )
            )
        ) * chi1 + (
            m2M
            * (
                -17097.8035 / 4.8384
                + eta_s * 28764.25 / 6.72
                + eta_s * eta_s * 47.35 / 1.44
                + m2M
                * (
                    -7189.233785 / 1.524096
                    + eta_s * 458.555 / 3.024
                    - eta_s * eta_s * 534.5 / 7.2
                )
            )
        ) * chi2

        # Add frequency dependence here
        PI_f_Ms = self.PI * fxi_Ms
        v = PI_f_Ms**self.ONE_BY_THREE
        _v = 1.0 / v

        phi_TF2 = (
            phi0 * (PI_f_Ms**-self.FIVE_BY_THREE)
            + phi1 * (PI_f_Ms ** -(4.0 / 3.0))
            + phi2 * (PI_f_Ms**-1.0)
            + phi3 * (PI_f_Ms ** -(2.0 / 3.0))
            + phi4 * _v
            + phi5_log * torch.log(v)
            + phi5
            + phi6_log * torch.log(v) * v
            + phi6 * v
            + phi7 * (PI_f_Ms ** (2.0 / 3.0))
        ) * (3.0 / (128.0 * eta_s)) - self / 4.0
        phi_Ins = (
            phi_TF2
            + (
                coeffs[7] * fxi_Ms
                + (3.0 / 4.0) * coeffs[8] * (fxi_Ms ** (4.0 / 3.0))
                + (3.0 / 5.0) * coeffs[9] * (fxi_Ms ** (5.0 / 3.0))
                + self.HALF * coeffs[10] * (fxi_Ms * fxi_Ms)
            )
            / eta_s
        )

        return phi_Ins

    def get_fcut_true(self, M_s):
        fcut = self.fM_CUT / M_s
        # Find the index where fcut_val would be inserted
        idx = torch.searchsorted(self.f, fcut, right=False) - 1
        idx = torch.clamp(idx, 0, self.f_numel - 1)  # ensure valid index
        # Use torch.where to handle the case when fcut_val is above f[-1]
        return torch.where(fcut > self.f[-1], fcut, self.f[idx])

    def amp(
        self,
        theta,
        coeffs,
        trans_fs,
        derived,
        fx_Ms,
        fcut_true=None,
    ):
        """
        Computes the amplitude of the PhenomD frequency domain waveform following 1508.07253.
        Note that this waveform also assumes that object one is the more massive.
        """

        # Required vars
        _, _, M_s, eta_s = derived
        _, _, chi1, chi2, D = theta

        # First we get the inspiral amplitude
        Amp_Ins = self.get_inspiral_Amp(fx_Ms[:, 0], chi1, chi2, eta_s, coeffs)

        # Next lets construct the phase of the late inspiral (region IIa)
        # Note that this part is a little harder since we need to solve a system of equations for deltas
        Amp_IIa = self.get_IIa_Amp(fx_Ms, theta, derived, coeffs)

        # And finally, we construct the amplitude of the merger-ringdown (region IIb)
        Amp_IIb = IMRPhenomD.get_IIb_Amp(fx_Ms[:, 0], fx_Ms, coeffs)

        # Check for fcut_true
        if fcut_true is None:
            fcut_true = self.get_fcut_true(M_s)

        Amp = torch.where(
            self.f <= trans_fs[:, 2],
            Amp_Ins,
            torch.where(
                self.f <= trans_fs[:, 3],
                Amp_IIa,
                torch.where(self.f <= fcut_true, Amp_IIb, torch.zeros_like(self.f)),
            ),
        )

        # Prefactor (This second factor is from lalsuite)
        Amp0 = self.get_Amp0(fx_Ms[:, 0], eta_s) * (
            2.0 * torch.sqrt(self.FIVE / (64.0 * self.PI))
        )

        # Need to add in an overall scaling of M_s^2 to make the units correct
        dist_s = (D * self.Mpc) / self.C

        return Amp0 * Amp * (M_s * M_s) / dist_s

    def get_inspiral_Amp(self, f_Ms, chi1, chi2, eta_s, coeffs):
        # Below is taken from lalsimulation/lib/LALSimIMRPhenomD_internals.c
        # Lines 302 --> 351
        eta2 = eta_s * eta_s
        eta3 = eta_s * eta2

        Seta = torch.sqrt(1.0 - 4.0 * eta_s)
        SetaPlus1 = 1.0 + Seta

        # Spin variables
        chi12 = chi1 * chi1
        chi22 = chi2 * chi2

        # First lets construct the Amplitude in the inspiral (region I)
        A0 = 1.0
        A2 = ((-969.0 + 1804.0 * eta_s) * self.PI ** (2.0 / 3.0)) / 672.0
        A3 = (
            (
                chi1 * (81.0 * SetaPlus1 - 44.0 * eta_s)
                + chi2 * (81.0 - 81.0 * Seta - 44.0 * eta_s)
            )
            * self.PI
        ) / 48.0
        A4 = (
            (
                -27312085.0
                - 10287648.0 * chi22
                - 10287648.0 * chi12 * SetaPlus1
                + 10287648.0 * chi22 * Seta
                + 24.0
                * (
                    -1975055.0
                    + 857304.0 * chi12
                    - 994896.0 * chi1 * chi2
                    + 857304.0 * chi22
                )
                * eta_s
                + 35371056.0 * eta2
            )
            * (self.PI ** (4.0 / 3.0))
        ) / 8.128512e6

        A5 = (
            (self.PI ** (5.0 / 3.0))
            * (
                chi2
                * (
                    -285197.0 * (-1 + Seta)
                    + 4 * (-91902.0 + 1579.0 * Seta) * eta_s
                    - 35632.0 * eta2
                )
                + chi1
                * (
                    285197.0 * SetaPlus1
                    - 4.0 * (91902.0 + 1579.0 * Seta) * eta_s
                    - 35632.0 * eta2
                )
                + 42840.0 * (-1.0 + 4.0 * eta_s) * self.PI
            )
        ) / 32256.0

        A6 = (
            -(
                (self.PI**2.0)
                * (
                    -336.0
                    * (
                        -3248849057.0
                        + 2943675504.0 * chi12
                        - 3339284256.0 * chi1 * chi2
                        + 2943675504.0 * chi22
                    )
                    * eta2
                    - 324322727232.0 * eta3
                    - 7.0
                    * (
                        -177520268561.0
                        + 107414046432.0 * chi22
                        + 107414046432.0 * chi12 * SetaPlus1
                        - 107414046432.0 * chi22 * Seta
                        + 11087290368.0
                        * (chi1 + chi2 + chi1 * Seta - chi2 * Seta)
                        * self.PI
                    )
                    + 12.0
                    * eta_s
                    * (
                        -545384828789.0
                        - 176491177632.0 * chi1 * chi2
                        + 202603761360.0 * chi22
                        + 77616.0 * chi12 * (2610335.0 + 995766.0 * Seta)
                        - 77287373856.0 * chi22 * Seta
                        + 5841690624.0 * (chi1 + chi2) * self.PI
                        + 21384760320.0 * (self.PI**2.0)
                    )
                )
            )
            / 6.0085960704e10
        )
        A7 = coeffs[0]
        A8 = coeffs[1]
        A9 = coeffs[2]

        Amp_Ins = (
            A0
            # A1 is missed since its zero
            + A2 * (f_Ms ** (2.0 / 3.0))
            + A3 * f_Ms
            + A4 * (f_Ms ** (4.0 / 3.0))
            + A5 * (f_Ms ** (5.0 / 3.0))
            + A6 * (f_Ms**2.0)
            # Now we add the coefficient terms
            + A7 * (f_Ms ** (7.0 / 3.0))
            + A8 * (f_Ms ** (8.0 / 3.0))
            + A9 * (f_Ms**3.0)
        )

        return Amp_Ins

    def get_IIa_Amp(self, fx_Ms, theta, derived, coeffs):
        # Required vars
        # f1, f3, f_RD, f_damp
        _, _, _, eta_s = derived
        _, _, chi1, chi2, _ = theta

        # For this region, we also need to calculate the the values and derivatives
        # of the Ins and IIb regions
        v1, d1 = torch_value_and_grad(
            self.get_inspiral_Amp, (fx_Ms[:, 2], chi1, chi2, eta_s, coeffs)
        )
        v3, d3 = torch_value_and_grad(
            IMRPhenomD.get_IIb_Amp, (fx_Ms[:, 4], fx_Ms, coeffs)
        )

        # Here we need the delta solutions
        delta0 = IMRPhenomD.get_delta0(
            fx_Ms[:, 2], fx_Ms[:, 7], fx_Ms[:, 4], v1, coeffs[3], v3, d1, d3
        )
        delta1 = IMRPhenomD.get_delta1(
            fx_Ms[:, 2], fx_Ms[:, 7], fx_Ms[:, 4], v1, coeffs[3], v3, d1, d3
        )
        delta2 = IMRPhenomD.get_delta2(
            fx_Ms[:, 2], fx_Ms[:, 7], fx_Ms[:, 4], v1, coeffs[3], v3, d1, d3
        )
        delta3 = IMRPhenomD.get_delta3(
            fx_Ms[:, 2], fx_Ms[:, 7], fx_Ms[:, 4], v1, coeffs[3], v3, d1, d3
        )
        delta4 = IMRPhenomD.get_delta4(
            fx_Ms[:, 2], fx_Ms[:, 7], fx_Ms[:, 4], v1, coeffs[3], v3, d1, d3
        )

        Amp_IIa = (
            delta0
            + delta1 * fx_Ms[:, 0]
            + delta2 * (fx_Ms[:, 0] ** 2.0)
            + delta3 * (fx_Ms[:, 0] ** 3.0)
            + delta4 * (fx_Ms[:, 0] ** 4.0)
        )

        return Amp_IIa

    @staticmethod
    def get_IIb_Amp(f_Ms, fx_Ms, coeffs):
        gamma1 = coeffs[4]
        gamma2 = coeffs[5]
        gamma3 = coeffs[6]

        fDMgamma3 = fx_Ms[:, 6] * gamma3
        fminfRD = f_Ms - fx_Ms[:, 5]
        Amp_IIb = (
            torch.exp(-(fminfRD) * gamma2 / (fDMgamma3))
            * (fDMgamma3 * gamma1)
            / ((fminfRD) ** 2.0 + (fDMgamma3) ** 2.0)
        )
        return Amp_IIb

    @staticmethod
    def get_delta0(f1, f2, f3, v1, v2, v3, d1, d3):
        return (
            -(d3 * f1**2 * (f1 - f2) ** 2 * f2 * (f1 - f3) * (f2 - f3) * f3)
            + d1 * f1 * (f1 - f2) * f2 * (f1 - f3) * (f2 - f3) ** 2 * f3**2
            + f3**2
            * (
                f2
                * (f2 - f3) ** 2
                * (-4 * f1**2 + 3 * f1 * f2 + 2 * f1 * f3 - f2 * f3)
                * v1
                + f1**2 * (f1 - f3) ** 3 * v2
            )
            + f1**2
            * (f1 - f2) ** 2
            * f2
            * (f1 * f2 - 2 * f1 * f3 - 3 * f2 * f3 + 4 * f3**2)
            * v3
        ) / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (f2 - f3) ** 2)

    @staticmethod
    def get_delta1(f1, f2, f3, v1, v2, v3, d1, d3):
        return (
            d3 * f1 * (f1 - f3) * (f2 - f3) * (2 * f2 * f3 + f1 * (f2 + f3))
            - (
                f3
                * (
                    d1
                    * (f1 - f2)
                    * (f1 - f3)
                    * (f2 - f3) ** 2
                    * (2 * f1 * f2 + (f1 + f2) * f3)
                    + 2
                    * f1
                    * (
                        f3**4 * (v1 - v2)
                        + 3 * f2**4 * (v1 - v3)
                        + f1**4 * (v2 - v3)
                        + 4 * f2**3 * f3 * (-v1 + v3)
                        + 2 * f1**3 * f3 * (-v2 + v3)
                        + f1
                        * (
                            2 * f3**3 * (-v1 + v2)
                            + 6 * f2**2 * f3 * (v1 - v3)
                            + 4 * f2**3 * (-v1 + v3)
                        )
                    )
                )
            )
            / (f1 - f2) ** 2
        ) / ((f1 - f3) ** 3 * (f2 - f3) ** 2)

    @staticmethod
    def get_delta2(f1, f2, f3, v1, v2, v3, d1, d3):
        return (
            d1
            * (f1 - f2)
            * (f1 - f3)
            * (f2 - f3) ** 2
            * (f1 * f2 + 2 * (f1 + f2) * f3 + f3**2)
            - d3
            * (f1 - f2) ** 2
            * (f1 - f3)
            * (f2 - f3)
            * (f1**2 + f2 * f3 + 2 * f1 * (f2 + f3))
            - 4 * f1**2 * f2**3 * v1
            + 3 * f1 * f2**4 * v1
            - 4 * f1 * f2**3 * f3 * v1
            + 3 * f2**4 * f3 * v1
            + 12 * f1**2 * f2 * f3**2 * v1
            - 4 * f2**3 * f3**2 * v1
            - 8 * f1**2 * f3**3 * v1
            + f1 * f3**4 * v1
            + f3**5 * v1
            + f1**5 * v2
            + f1**4 * f3 * v2
            - 8 * f1**3 * f3**2 * v2
            + 8 * f1**2 * f3**3 * v2
            - f1 * f3**4 * v2
            - f3**5 * v2
            - (f1 - f2) ** 2
            * (
                f1**3
                + f2 * (3 * f2 - 4 * f3) * f3
                + f1**2 * (2 * f2 + f3)
                + f1 * (3 * f2 - 4 * f3) * (f2 + 2 * f3)
            )
            * v3
        ) / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (f2 - f3) ** 2)

    @staticmethod
    def get_delta3(f1, f2, f3, v1, v2, v3, d1, d3):
        return (
            (d3 * (f1 - f3) * (2 * f1 + f2 + f3)) / (f2 - f3)
            - (d1 * (f1 - f3) * (f1 + f2 + 2 * f3)) / (f1 - f2)
            + (
                2
                * (
                    f3**4 * (-v1 + v2)
                    + 2 * f1**2 * (f2 - f3) ** 2 * (v1 - v3)
                    + 2 * f2**2 * f3**2 * (v1 - v3)
                    + 2 * f1**3 * f3 * (v2 - v3)
                    + f2**4 * (-v1 + v3)
                    + f1**4 * (-v2 + v3)
                    + 2
                    * f1
                    * f3
                    * (f3**2 * (v1 - v2) + f2**2 * (v1 - v3) + 2 * f2 * f3 * (-v1 + v3))
                )
            )
            / ((f1 - f2) ** 2 * (f2 - f3) ** 2)
        ) / (f1 - f3) ** 3

    @staticmethod
    def get_delta4(f1, f2, f3, v1, v2, v3, d1, d3):
        return (
            -(d3 * (f1 - f2) ** 2 * (f1 - f3) * (f2 - f3))
            + d1 * (f1 - f2) * (f1 - f3) * (f2 - f3) ** 2
            - 3 * f1 * f2**2 * v1
            + 2 * f2**3 * v1
            + 6 * f1 * f2 * f3 * v1
            - 3 * f2**2 * f3 * v1
            - 3 * f1 * f3**2 * v1
            + f3**3 * v1
            + f1**3 * v2
            - 3 * f1**2 * f3 * v2
            + 3 * f1 * f3**2 * v2
            - f3**3 * v2
            - (f1 - f2) ** 2 * (f1 + 2 * f2 - 3 * f3) * v3
        ) / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (f2 - f3) ** 2)

    def get_Amp0(self, f_Ms, eta):
        Amp0 = (
            (2.0 / 3.0 * eta) ** (1.0 / 2.0)
            * (f_Ms) ** (-7.0 / 6.0)
            * self.PI ** (-1.0 / 6.0)
        )
        return Amp0

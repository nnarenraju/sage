#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomD.py
Description     : Batched PyTorch port of the IMRPhenomD aligned-spin
                  frequency-domain waveform (LALSimIMRPhenomD.c,
                  LALSimIMRPhenomD_internals.c).

Created on 2026-01-22 10:38:25

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

Licensing
---------
This file is part of Sage and is distributed under GPL-3.0-or-later.
Portions derive from the MIT-licensed JAX reference implementation by
Adam Coogan and Thomas Edwards; that notice is retained verbatim in
``THIRD_PARTY_LICENSE`` in this directory and must not be removed.  The MIT
terms cover those derived portions only.

"""


import torch

from sage.core.interpolation import torch_natural_cubic_interp
from sage.data.waveform.approximants import phenom
from sage.data.waveform import taper, waveform_utils


class IMRPhenomD(phenom.PhenomConstants):
    """
    Batched IMRPhenomD aligned-spin waveform on a fixed frequency grid.

    The waveform is computed in float64; the output spans zero
    frequency to the last grid frequency.

    Parameters
    ----------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.
    padding_length_in_s : float
        Leading padding; delta_tc is measured from its end.
    **kwargs
        Ignored.
    """

    def __init__(self, f, f_ref, padding_length_in_s, **kwargs):
        f = f.to(torch.float64)
        f_ref = f_ref.to(torch.float64)
        super().__init__(
            device=f.device,
            batch_size=f.shape[0],
            dtype=f.dtype,
        )
        # Fixed frequency grid
        self.f = f
        _n = self.f[0].numel()
        self.df = (f[0, -1] - f[0, 0]) / (_n - 1)
        self.sample_length_in_s = 1.0 / self.df
        self.f_numel = _n
        self.f_ref = f_ref
        self.padding_length_in_s = padding_length_in_s
        # Batch size
        self.B = f.shape[0]
        # Output buffers from zero frequency to the last grid frequency
        self.n_pad = int(torch.round((self.f[0][0] - self.df) / self.df)) + 1
        self.hp_buffer = torch.empty(
            (self.B, self.n_pad + self.f_numel),
            dtype=torch.complex128,
            device=f.device,
        )
        self.hc_buffer = torch.empty_like(self.hp_buffer)

    # Source: IMRPhenomDGenerateFD (LALSimIMRPhenomD.c).
    def get_hphc(self, theta, reproduce_lal=False):
        """
        FD plus and cross polarisations of a parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip FD tapering, delta_tc shift and df
            normalisation and keep LAL's band (lal_band), so the output
            compares directly with LAL's.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_frequency_bins), complex128

        Raises
        ------
        ValueError
            If any transverse spin is non-zero.
        """
        # [m1, m2, chi1, chi2, distance, delta_tc, phic, inclination]
        theta = waveform_utils.aligned_layout(
            theta, waveform_utils.ALIGNED_COLUMNS,
            waveform_utils.TRANSVERSE_SPIN_PARAMETERS + waveform_utils.TIDAL_PARAMETERS,
        )
        # Swap spins and masses unless m1 >= m2
        swap = ~(theta[:, 0:1] >= theta[:, 1:2])
        theta = torch.where(swap, theta[:, [1, 0, 3, 2, 4, 5, 6, 7]], theta)

        derived = self.compute_derived_parameters(theta)

        coeffs = self.get_coeffs(theta[:, 2:3], theta[:, 3:4], derived[:, 3:4])
        A, Psi, fcut_true = self.get_components(theta, coeffs, derived)

        # h+ = 0.5 (1 + cos^2 i) h and hx = -i cos(i) h
        hp = torch.polar(
            0.5 * A * (1.0 + torch.cos(theta[:, 7:8]) ** 2),
            -Psi,
        )
        hc = torch.polar(
            A * torch.cos(theta[:, 7:8]),
            -Psi - 0.5 * self.PI,
        )

        if not reproduce_lal:
            # Frequency domain tapering
            _taper = taper.fd_taper(
                f=self.f,
                f_min=self.f[0, 0].item(),
                f_cut=fcut_true,
                df=self.df,
            )
            hp *= _taper
            hc *= _taper

            # Phase shift equivalent to a delta_tc time shift
            hp, hc = self.apply_delta_tc(hp, hc, theta[:, 5:6])

            # Continuous to discrete Fourier normalisation
            hp *= self.df
            hc *= self.df
        else:
            hp, hc = self.lal_band(hp, hc, fcut_true)

        # Zeros from zero frequency to the starting frequency
        hp, hc = self.pad_missing_frequencies(hp, hc)

        return hp, hc

    def apply_delta_tc(self, hp, hc, delta_tc):
        """
        Shift the merger delta_tc seconds into the analysis window.

        delta_tc is measured from the end of the leading padding, so the
        merger lands padding_length_in_s + delta_tc into the segment.

        Parameters
        ----------
        hp, hc : torch.Tensor, shape (B, F)
            Plus and cross polarisations.
        delta_tc : torch.Tensor, shape (B, 1)
            Merger time after the leading padding, in s.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, F)
        """
        shift = (delta_tc + self.padding_length_in_s) - self.sample_length_in_s
        hp = torch.polar(torch.abs(hp), torch.angle(hp) - 2 * self.PI * self.f * shift)
        hc = torch.polar(torch.abs(hc), torch.angle(hc) - 2 * self.PI * self.f * shift)
        return hp, hc

    def pad_missing_frequencies(self, hp, hc):
        """Zero-pad hp and hc from zero frequency to the grid start."""
        hp_pad = torch.zeros_like(self.hp_buffer)
        hc_pad = torch.zeros_like(self.hc_buffer)
        hp_pad[:, self.n_pad :] = hp
        hc_pad[:, self.n_pad :] = hc

        return hp_pad, hc_pad

    def compute_derived_parameters(self, theta):
        """
        Mass-derived quantities of a parameter batch.

        Returns
        -------
        torch.Tensor, shape (B, 4)
            m1, m2 and the total mass M in seconds, and eta.
        """
        m1_s = theta[:, 0:1] * self.GM
        m2_s = theta[:, 1:2] * self.GM
        M_s = m1_s + m2_s
        eta_s = m1_s * m2_s / (M_s * M_s)
        # Nudge eta > 0.25 to 0.25 (IMRPhenomDGenerateFD)
        eta_s.clamp_(max=0.25)

        return torch.cat([m1_s, m2_s, M_s, eta_s], dim=1)

    # Source: ComputeIMRPhenomDAmplitudeCoefficients and
    # ComputeIMRPhenomDPhaseCoefficients (LALSimIMRPhenomD_internals.c).
    def get_coeffs(self, chi1, chi2, eta):
        """
        Phenom coefficients, as fits in eta and xi = chiPN - 1.

        Parameters
        ----------
        chi1, chi2 : torch.Tensor, shape (B, 1)
            Aligned spins, with m1 >= m2.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        torch.Tensor, shape (B, 19)
            rho1-3, v2, gamma1-3, sigma1-4, beta1-3 and alpha1-5.
        """
        # PN reduced spin parameter
        chi_s = (chi1 + chi2) / 2.0
        chi_a = (chi1 - chi2) / 2.0
        seta = torch.sqrt(self.ONE - 4 * eta)
        chiPN = chi_s * (self.ONE - 76 * eta / 113) + seta * chi_a

        # Powers of xi = chiPN - 1
        chi0 = self.ONES
        chi1 = chiPN - self.ONE
        chi2 = chi1**2
        chi3 = chi1**3

        # Powers of eta
        eta0 = self.ONES
        eta1 = eta
        eta2 = eta**2

        powers = torch.cat(
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
            dim=1,
        )

        coeff = powers @ self.PhenomD_coeff_table.T

        return coeff

    # Source: IMRPhenomDGenerateFD (LALSimIMRPhenomD.c).
    def get_components(self, theta, coeffs, derived):
        """
        Amplitude, phase and cutoff frequency of a parameter batch.

        Returns
        -------
        A, Psi : torch.Tensor, shape (B, F)
            Amplitude and phase on the frequency grid.
        fcut_true : torch.Tensor, shape (B, 1)
            Cutoff frequency fCut in Hz.
        """
        M_s = derived[:, 2:3]

        # Transition frequencies f1, f2, f3, f4, fRD and fdamp
        trans_fs = self.get_transition_frequencies(
            theta[:, :4], derived, coeffs[:, 5:6], coeffs[:, 6:7]
        )

        f1_Ms = trans_fs[:, 0:1] * M_s
        f2_Ms = trans_fs[:, 1:2] * M_s
        f3_Ms = trans_fs[:, 2:3] * M_s
        f4_Ms = trans_fs[:, 3:4] * M_s
        f_Ms = self.f * M_s
        fref_Ms = self.f_ref * M_s
        f_RD_Ms = trans_fs[:, 4:5] * M_s
        f_damp_Ms = trans_fs[:, 5:6] * M_s
        # Middle of the three amplitude collocation points
        fmid_Ms = ((trans_fs[:, 2:3] + trans_fs[:, 3:4]) / 2) * M_s
        fx_Ms = torch.cat(
            [fref_Ms, f1_Ms, f2_Ms, f3_Ms, f4_Ms, f_RD_Ms, f_damp_Ms, fmid_Ms], dim=1
        )

        # Time shift so that peak amplitude is approximately at t=0
        f4_scaled = trans_fs[:, 3:4] * M_s
        t0 = IMRPhenomD.DPhiMRD(f4_scaled, coeffs, derived[:, 3:4], fx_Ms)

        Psi = self.phase(theta[:, :4], coeffs, derived, f_Ms, fx_Ms)
        Psi_ref = self.phase(theta[:, :4], coeffs, derived, fref_Ms, fx_Ms)
        Psi -= t0 * ((f_Ms) - fref_Ms) + Psi_ref
        # Factor of 2 because phi0 is orbital phase
        ext_phase_contrib = -2 * theta[:, 6:7]
        Psi += ext_phase_contrib

        fcut_true = self.get_fcut_true(M_s)

        Psi = torch.where(self.f <= fcut_true, Psi, self.TWOPI)

        A = self.amp(
            self.f,
            theta[:, :5],
            coeffs,
            trans_fs,
            derived,
            f_Ms,
            fx_Ms,
            fcut_true,
        )

        return A, Psi, fcut_true

    # Source: ComputeIMRPhenDPhaseConnectionCoefficients and fmaxCalc
    # (LALSimIMRPhenomD_internals.c).
    def get_transition_frequencies(self, theta, derived, gamma2, gamma3):
        """
        Transition, ringdown and damping frequencies, in Hz.

        Returns
        -------
        torch.Tensor, shape (B, 6)
            f1, f2, f3, f4, fRD and fdamp.
        """
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]
        m1_s = derived[:, 0:1]
        m2_s = derived[:, 1:2]
        M_s = derived[:, 2:3]
        eta_s = derived[:, 3:4]

        f_RD, f_damp = self.get_fRD_fdamp(chi1, chi2, m1_s, m2_s, M_s, eta_s)

        # Phase transition frequencies
        f1 = 0.018 / M_s
        f2 = 0.5 * f_RD

        # Amplitude transition frequencies
        f3 = 0.014 / M_s
        sqrt_term = torch.sqrt(torch.clamp(1.0 - gamma2**2, min=0.0))

        f4_gammaneg_gtr_1 = torch.abs(f_RD + (-f_damp * gamma3) / gamma2)
        f4_gammaneg_less_1 = torch.abs(
            f_RD + (f_damp * (-1 + sqrt_term) * gamma3) / gamma2
        )

        # If gamma2 >= 1 then set the square root term to zero
        f4 = torch.where(gamma2 >= 1, f4_gammaneg_gtr_1, f4_gammaneg_less_1)

        return torch.cat([f1, f2, f3, f4, f_RD, f_damp], dim=1)

    # Source: fring and fdamp (LALSimIMRPhenomD_internals.c).
    def get_fRD_fdamp(self, chi1, chi2, m1_s, m2_s, M_s, eta_s):
        """
        Ringdown and damping frequencies, in Hz.

        The QNM tables are interpolated at the final spin by natural
        cubic spline and divided by 1 - Erad.
        """
        S = (chi1 * m1_s * m1_s + chi2 * m2_s * m2_s) / (M_s * M_s)
        # Final spin and radiated energy
        a = IMRPhenomD.final_spin_0815_s(eta_s, S)
        Erad = IMRPhenomD.erad_rational_0815(eta_s, chi1, chi2)

        fRD = torch_natural_cubic_interp(
            a, self.QNMData_a, self.QNMData_fRD, self.QNMData_fRD_M
        )
        fdamp = torch_natural_cubic_interp(
            a, self.QNMData_a, self.QNMData_fdamp, self.QNMData_fdamp_M
        )

        factor = 1.0 / (1.0 - Erad)
        fRD *= factor
        fdamp *= factor

        return fRD / M_s, fdamp / M_s

    # Source: FinalSpin0815_s (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def final_spin_0815_s(eta, S):
        """Formula to predict the final spin."""
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

    # Source: PhenomInternal_EradRational0815 and EradRational0815_s
    # (LALSimIMRPhenomInternalUtils.c).
    @staticmethod
    def erad_rational_0815(eta, chi1, chi2):
        """Formula to predict the total radiated energy."""
        # Convention m1 >= m2
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        m1f = 0.5 * (1.0 + Seta)
        m2f = 0.5 * (1.0 - Seta)
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
    def get_phi_IIa(f_Ms, eta, coeffs, beta1corr):
        """
        Intermediate phase PhiIntAnsatz / eta + C2Int f, without C1Int.
        """
        return IMRPhenomD.get_IIa_raw_phase(f_Ms, eta, coeffs) + beta1corr * f_Ms

    # Source: PhiIntAnsatz (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_IIa_raw_phase(f_Ms, eta, coeffs):
        """Ansatz for the intermediate phase, times 1/eta."""
        phi_IIa_raw = (
            coeffs[:, 11:12] * f_Ms
            + coeffs[:, 12:13] * torch.log(f_Ms)
            - coeffs[:, 13:14] * (f_Ms**-3.0) / 3.0
        ) / eta

        return phi_IIa_raw

    # Source: PhiMRDAnsatzInt (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_IIb_raw_phase(f_Ms, eta, coeffs, fx_Ms):
        """Ansatz for the merger-ringdown phase, times 1/eta."""
        phi_IIb_raw = (
            coeffs[:, 14:15] * f_Ms
            - coeffs[:, 15:16] * (f_Ms**-1.0)
            + 4.0 * coeffs[:, 16:17] * (f_Ms ** (3.0 / 4.0)) / 3.0
            + coeffs[:, 17:18]
            * torch.arctan((f_Ms - coeffs[:, 18:19] * fx_Ms[:, 5:6]) / fx_Ms[:, 6:7])
        ) / eta

        return phi_IIb_raw

    # Source: IMRPhenDPhase and
    # ComputeIMRPhenDPhaseConnectionCoefficients
    # (LALSimIMRPhenomD_internals.c).
    def phase(self, theta, coeffs, derived, f_Ms, fx_Ms):
        """
        IMR phase given phenom coefficients.

        The inspiral, intermediate and merger-ringdown parts are joined
        C(1) continuously at the transition frequencies.
        """
        phi6corr = self.spin_spin_3pn_correction(theta, derived[:, 3:4])
        phi_Ins, _, _ = self.get_inspiral_phase(
            f_Ms, derived, theta[:, 2:4], coeffs, phi6corr
        )

        # Compute C1Int (beta0) and C2Int (beta1corr) coeffs:
        # PhiIns (f1) = PhiInt (f1) + C1Int + C2Int f1
        # PhiIns'(f1) = PhiInt'(f1) + C2Int
        phi_Ins_f1, TF2_coeffs, TF2_log_coeffs = self.get_inspiral_phase(
            fx_Ms[:, 1:2], derived, theta[:, 2:4], coeffs, phi6corr
        )
        dphi_Ins_f1 = self.DPhiInsAnsatzInt(
            fx_Ms[:, 1:2], coeffs, TF2_coeffs, TF2_log_coeffs, derived[:, 3:4]
        )

        phi_IIa_f1 = IMRPhenomD.get_IIa_raw_phase(
            fx_Ms[:, 1:2], derived[:, 3:4], coeffs
        )
        dphi_IIa_f1 = IMRPhenomD.DPhiIntAnsatz(fx_Ms[:, 1:2], coeffs, derived[:, 3:4])

        beta1corr = dphi_Ins_f1 - dphi_IIa_f1
        beta0 = phi_Ins_f1 - beta1corr * fx_Ms[:, 1:2] - phi_IIa_f1
        phi_IIa = (
            IMRPhenomD.get_phi_IIa(f_Ms, derived[:, 3:4], coeffs, beta1corr) + beta0
        )

        # Compute C1MRD (a0) and C2MRD (a1_correction) coeffs:
        # PhiInsInt (f2) = PhiMRD (f2) + C1MRD + C2MRD f2
        # PhiInsInt'(f2) = PhiMRD'(f2) + C2MRD
        phi_IIa_f2 = IMRPhenomD.get_phi_IIa(
            fx_Ms[:, 2:3], derived[:, 3:4], coeffs, beta1corr
        )
        dphi_IIa_f2 = self.DPhiIntTemp(
            fx_Ms[:, 2:3], coeffs, derived[:, 3:4], beta1corr
        )

        phi_IIb_f2 = IMRPhenomD.get_IIb_raw_phase(
            fx_Ms[:, 2:3], derived[:, 3:4], coeffs, fx_Ms
        )
        dphi_IIb_f2 = IMRPhenomD.DPhiMRD(fx_Ms[:, 2:3], coeffs, derived[:, 3:4], fx_Ms)

        a1_correction = dphi_IIa_f2 - dphi_IIb_f2
        a0 = phi_IIa_f2 + beta0 - a1_correction * fx_Ms[:, 2:3] - phi_IIb_f2

        phi_IIb = (
            IMRPhenomD.get_IIb_raw_phase(f_Ms, derived[:, 3:4], coeffs, fx_Ms)
            + a0
            + a1_correction * f_Ms
        )

        # Inspiral, intermediate and merger-ringdown ranges
        phase = torch.where(
            f_Ms <= fx_Ms[:, 1:2],
            phi_Ins,
            torch.where(f_Ms <= fx_Ms[:, 2:3], phi_IIa, phi_IIb),
        )

        return phase

    # Source: DPhiInsAnsatzInt (LALSimIMRPhenomD_internals.c).
    def DPhiInsAnsatzInt(
        self,
        fxi_Ms,
        coeffs,
        TF2_coeffs,
        TF2_log_coeffs,
        eta,
    ):
        """
        First frequency derivative of PhiInsAnsatzInt.

        Parameters
        ----------
        fxi_Ms : torch.Tensor, shape (B, 1)
            Geometric frequency.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.
        TF2_coeffs : torch.Tensor, shape (B, 8)
            TaylorF2 coefficients v[0] to v[7], divided by
            pfaN = 3 / (128 eta).
        TF2_log_coeffs : torch.Tensor, shape (B, 2)
            TaylorF2 coefficients vlogv[5], vlogv[6], divided by pfaN.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        torch.Tensor, same shape as fxi_Ms
        """

        sigma1 = coeffs[:, 7:8]
        sigma2 = coeffs[:, 8:9]
        sigma3 = coeffs[:, 9:10]
        sigma4 = coeffs[:, 10:11]

        # Assemble PN phasing series
        v = torch.pow(self.PI * fxi_Ms, self.ONE_BY_THREE)
        logv = torch.log(v)

        v2 = v * v
        v3 = v * v2
        v4 = v * v3
        v5 = v * v4
        v6 = v * v5
        v7 = v * v6
        v8 = v * v7

        # Apply the correct prefactors to LAL phase coefficients to get
        # the phase derivative dphi / dMf = dphi/dv * dv/dMf
        Dphasing = torch.zeros_like(v)

        Dphasing += 2.0 * TF2_coeffs[:, 7:8] * v7
        Dphasing += (TF2_coeffs[:, 6:7] + TF2_log_coeffs[:, 1:2] * (1.0 + logv)) * v6
        Dphasing += TF2_log_coeffs[:, 0:1] * v5
        Dphasing += -1.0 * TF2_coeffs[:, 4:5] * v4
        Dphasing += -2.0 * TF2_coeffs[:, 3:4] * v3
        Dphasing += -3.0 * TF2_coeffs[:, 2:3] * v2
        Dphasing += -4.0 * TF2_coeffs[:, 1:2] * v
        Dphasing += -5.0 * TF2_coeffs[:, 0:1]

        Dphasing /= v8 * 3.0
        Dphasing *= self.PI
        Dphasing *= 3.0 / (128.0 * eta)

        # Now add higher order terms that were calibrated for PhenomD
        Dphasing += (
            sigma1
            + sigma2 * v * (self.PI ** (-1.0 / 3.0))
            + sigma3 * v2 * (self.PI ** (-2.0 / 3.0))
            + sigma4 * v3 * (1.0 / self.PI)
        ) * (1.0 / eta)

        return Dphasing

    # Source: DPhiIntAnsatz (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def DPhiIntAnsatz(fxi_Ms, coeffs, eta):
        """
        First frequency derivative of PhiIntAnsatz, times 1/eta.

        Parameters
        ----------
        fxi_Ms : torch.Tensor, shape (B, 1)
            Geometric frequency.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        torch.Tensor, same shape as fxi_Ms
        """

        Mf4 = fxi_Ms**4

        beta1 = coeffs[:, 11:12]  # p->beta1
        beta2 = coeffs[:, 12:13]  # p->beta2
        beta3 = coeffs[:, 13:14]  # p->beta3

        return (beta1 + beta3 / Mf4 + beta2 / fxi_Ms) * (1.0 / eta)

    # Source: DPhiIntTemp (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def DPhiIntTemp(fxi_Ms, coeffs, eta, beta1corr):
        """
        Temporary instance of DPhiIntAnsatz for the C(1) connection.

        Parameters
        ----------
        fxi_Ms : torch.Tensor, shape (B, 1)
            Geometric frequency.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        beta1corr : torch.Tensor, shape (B, 1)
            Connection coefficient C2Int.

        Returns
        -------
        torch.Tensor, same shape as fxi_Ms
        """

        ff4 = fxi_Ms**4

        beta1 = coeffs[:, 11:12]  # p->beta1
        beta2 = coeffs[:, 12:13]  # p->beta2
        beta3 = coeffs[:, 13:14]  # p->beta3

        return beta1corr + (beta1 + beta3 / ff4 + beta2 / fxi_Ms) * (1.0 / eta)

    # Source: DPhiMRD (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def DPhiMRD(f, coeffs, eta, fx_Ms, Rholm=1.0, Taulm=1.0):
        """
        First frequency derivative of PhiMRDAnsatzInt.

        Parameters
        ----------
        f : torch.Tensor, shape (B, 1)
            Geometric frequency.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        fx_Ms : torch.Tensor, shape (B, 8)
            Geometric fRef, f1, f2, f3, f4, fRD, fdamp and fmid.
        Rholm, Taulm : float
            fRD22/fRDlm and fDMlm/fDM22; 1 for PhenomD.

        Returns
        -------
        torch.Tensor, same shape as f
        """

        f2 = f * f

        alpha1 = coeffs[:, 14:15]
        alpha2 = coeffs[:, 15:16]
        alpha3 = coeffs[:, 16:17]
        alpha4 = coeffs[:, 17:18]
        alpha5 = coeffs[:, 18:19]

        fRD = fx_Ms[:, 5:6]
        fDM = fx_Ms[:, 6:7]

        denom_inner = 1.0 + (f - alpha5 * fRD) ** 2 / ((fDM * Taulm * Rholm) ** 2)

        term4 = alpha4 / (fDM * Taulm * denom_inner)

        return (alpha1 + alpha2 / f2 + alpha3 / torch.pow(f, 0.25) + term4) * (
            1.0 / eta
        )

    # Source: Subtract3PNSS (LALSimIMRPhenomD_internals.c).
    def spin_spin_3pn_correction(self, theta, eta_s):
        """
        3PN spin-spin term subtracted from the TaylorF2 phasing.

        This term is in LAL's TaylorF2 implementation, but was not
        available when PhenomD was tuned.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 4)
            m1, m2, chi1 and chi2.
        eta_s : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        m1 = theta[:, 0:1]
        m2 = theta[:, 1:2]
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]

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

    # Source: PhiInsAnsatzInt (LALSimIMRPhenomD_internals.c),
    # XLALSimInspiralPNPhasing_F2 (LALSimInspiralPNCoefficients.c).
    def get_inspiral_phase(self, fxi_Ms, derived, chi, coeffs, phi6corr):
        """
        Ansatz for the inspiral phase, with the TaylorF2 coefficients.

        Parameters
        ----------
        fxi_Ms : torch.Tensor, shape (B, F) or (B, 1)
            Geometric frequencies.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters.
        chi : torch.Tensor, shape (B, 2)
            Aligned spins chi1 and chi2.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.
        phi6corr : torch.Tensor, shape (B, 1)
            3PN spin-spin term from spin_spin_3pn_correction.

        Returns
        -------
        phi_Ins : torch.Tensor, same shape as fxi_Ms
            Inspiral phase.
        TF2_coeffs : torch.Tensor, shape (B, 8)
            TaylorF2 coefficients v[0] to v[7], divided by
            pfaN = 3 / (128 eta).
        TF2_log_coeffs : torch.Tensor, shape (B, 2)
            TaylorF2 coefficients vlogv[5], vlogv[6], divided by pfaN.
        """
        m1_s, m2_s, M_s, eta_s = (
            derived[:, 0:1],
            derived[:, 1:2],
            derived[:, 2:3],
            derived[:, 3:4],
        )
        chi1, chi2 = chi[:, 0:1], chi[:, 1:2]

        m1M = m1_s / M_s
        m2M = m2_s / M_s
        chi1sq = chi1 * chi1
        chi2sq = chi2 * chi2
        chi12 = chi1 * chi2  # chi1 . chi2 for aligned spins

        # Newtonian and 1PN
        phi0 = self.ONES
        phi1 = self.ZEROS
        phi2 = 5.0 * (743.0 / 84.0 + 11.0 * eta_s) / 9.0

        # 1.5PN
        phi3 = -16.0 * self.PI * self.ONES
        phi3 += m1M * (25.0 + 38.0 / 3.0 * m1M) * chi1
        phi3 += m2M * (25.0 + 38.0 / 3.0 * m2M) * chi2

        # 2PN
        phi4 = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta_s + 617.0 * eta_s**2) / 72.0
        # Add SS, QM, and Self-Spin terms
        phi4 += (247.0 / 4.8 * eta_s) * chi12  # S1S2
        phi4 += (-721.0 / 4.8 * eta_s) * chi1 * chi2  # S1S2O
        phi4 += (
            -720.0 / 9.6 * m1M**2 + 1.0 / 9.6 * m1M**2
        ) * chi1sq  # QM2SO + Self2SO (m1)
        phi4 += (
            -720.0 / 9.6 * m2M**2 + 1.0 / 9.6 * m2M**2
        ) * chi2sq  # QM2SO + Self2SO (m2)
        phi4 += (
            240.0 / 9.6 * m1M**2 - 7.0 / 9.6 * m1M**2
        ) * chi1sq  # QM2S + Self2S (m1)
        phi4 += (
            240.0 / 9.6 * m2M**2 - 7.0 / 9.6 * m2M**2
        ) * chi2sq  # QM2S + Self2S (m2)

        # 2.5PN
        phi5 = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta_s) * self.PI
        so5 = (
            -m1M
            * (
                1391.5 / 8.4
                - m1M * (1.0 - m1M) * 10.0 / 3.0
                + m1M * (1276.0 / 8.1 + m1M * (1.0 - m1M) * 170.0 / 9.0)
            )
            * chi1
        )
        so5 += (
            -m2M
            * (
                1391.5 / 8.4
                - m2M * (1.0 - m2M) * 10.0 / 3.0
                + m2M * (1276.0 / 8.1 + m2M * (1.0 - m2M) * 170.0 / 9.0)
            )
            * chi2
        )
        phi5 += so5
        phi5_log = (
            self.FIVE_BY_THREE * (772.9 / 8.4 - 13.0 * eta_s) * self.PI + 3.0 * so5
        )

        # 3PN
        phi6 = (
            11583.231236531 / 4.694215680
            - 640.0 / 3.0 * self.PI**2
            - 684.8 / 2.1 * self.EulerGamma
        ) * self.ONES
        phi6 += eta_s * (-15737.765635 / 3.048192 + 225.5 / 1.2 * self.PI**2)
        phi6 += eta_s**2 * 76.055 / 1.728 - eta_s**3 * 127.825 / 1.296
        phi6 += (-684.8 / 2.1) * torch.log(self.FOUR)
        # Add 3PN SO/SS/QM terms
        phi6 += (self.PI * m1M * (1490.0 / 3.0 + m1M * 260.0)) * chi1
        phi6 += (self.PI * m2M * (1490.0 / 3.0 + m2M * 260.0)) * chi2
        # S1S2O
        phi6 += ((326.75 / 1.12 + 557.5 / 1.8 * eta_s) * eta_s) * chi1 * chi2
        phi6 += (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M**2) * m1M**2
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M**2) * m1M**2
        ) * chi1sq
        phi6 += (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M**2) * m2M**2
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M**2) * m2M**2
        ) * chi2sq
        # Subtract 3PN spin-spin term (Subtract3PNSS)
        phi6 = phi6 - phi6corr
        phi6_log = self.PHI6LOG

        # 3.5PN
        phi7 = self.PI * (
            770.96675 / 2.54016 + 378.515 / 1.512 * eta_s - 740.45 / 7.56 * eta_s**2
        )
        phi7 += (
            m1M
            * (
                -17097.8035 / 4.8384
                + eta_s * 28764.25 / 6.72
                + eta_s**2 * 47.35 / 1.44
                + m1M
                * (
                    -7189.233785 / 1.524096
                    + eta_s * 458.555 / 3.024
                    - eta_s**2 * 534.5 / 7.2
                )
            )
        ) * chi1
        phi7 += (
            m2M
            * (
                -17097.8035 / 4.8384
                + eta_s * 28764.25 / 6.72
                + eta_s**2 * 47.35 / 1.44
                + m2M
                * (
                    -7189.233785 / 1.524096
                    + eta_s * 458.555 / 3.024
                    - eta_s**2 * 534.5 / 7.2
                )
            )
        ) * chi2

        # Assemble PN phasing series
        TF2_coeffs = torch.cat([phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7], dim=1)
        TF2_log_coeffs = torch.cat([phi5_log, phi6_log], dim=1)

        PI_f_Ms = self.PI * fxi_Ms
        v = PI_f_Ms**self.ONE_BY_THREE

        phi_TF2 = (
            phi0 * (PI_f_Ms**-self.FIVE_BY_THREE)
            + phi2 * (PI_f_Ms**-1.0)
            + phi3 * (PI_f_Ms ** -(2.0 / 3.0))
            + phi4 * (v**-1.0)
            + phi5_log * torch.log(v)
            + phi5
            + phi6_log * torch.log(v) * v
            + phi6 * v
            + phi7 * (PI_f_Ms ** (2.0 / 3.0))
        ) * (3.0 / (128.0 * eta_s)) - self.PI / 4.0

        # Now add higher order terms that were calibrated for PhenomD
        phi_Ins = (
            phi_TF2
            + (
                coeffs[:, 7:8] * fxi_Ms
                + (3.0 / 4.0) * coeffs[:, 8:9] * (fxi_Ms ** (4.0 / 3.0))
                + (3.0 / 5.0) * coeffs[:, 9:10] * (fxi_Ms ** (5.0 / 3.0))
                + 0.5 * coeffs[:, 10:11] * (fxi_Ms**2)
            )
            / eta_s
        )

        return phi_Ins, TF2_coeffs, TF2_log_coeffs

    # Source: IMRPhenomDGenerateFD (LALSimIMRPhenomD.c), PhenomPCore
    # (LALSimIMRPhenomP.c).
    def lal_band(self, hp, hc, f_cut):
        """
        Zero the bins LAL leaves empty.

        LAL fills i * deltaF for i < floor(min(f_max, fCut) / deltaF)
        and zeros the rest.
        """
        stop = torch.floor(torch.minimum(f_cut, self.f[:, -1:]) / self.df)
        keep = torch.round(self.f / self.df) < stop
        return torch.where(keep, hp, 0), torch.where(keep, hc, 0)

    # Source: XLALSimIMRPhenomDGenerateFD (LALSimIMRPhenomD.c).
    def get_fcut_true(self, M_s):
        """Cutoff frequency fCut = f_CUT / M_sec, in Hz."""
        return self.fM_CUT / M_s

    # Source: IMRPhenDAmplitude (LALSimIMRPhenomD_internals.c),
    # IMRPhenomDGenerateFD (LALSimIMRPhenomD.c).
    def amp(
        self,
        f,
        theta,
        coeffs,
        trans_fs,
        derived,
        f_Ms,
        fx_Ms,
        fcut_true=None,
    ):
        """
        IMR amplitude of a parameter batch, with m1 >= m2.

        The inspiral, intermediate and merger-ringdown amplitude parts,
        zero above fCut.
        """

        M_s = derived[:, 2:3]
        eta_s = derived[:, 3:4]
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]
        D = theta[:, 4:5]

        Amp_Ins = self.get_inspiral_Amp(f_Ms, chi1, chi2, eta_s, coeffs)

        Amp_IIa = self.get_IIa_Amp(f_Ms, fx_Ms, theta, derived, coeffs)

        Amp_IIb = IMRPhenomD.get_IIb_Amp(f_Ms, fx_Ms, coeffs)

        if fcut_true is None:
            fcut_true = self.get_fcut_true(M_s)

        Amp = torch.where(
            f <= trans_fs[:, 2:3],
            Amp_Ins,
            torch.where(
                f <= trans_fs[:, 3:4],
                Amp_IIa,
                torch.where(f <= fcut_true, Amp_IIb, torch.zeros_like(f)),
            ),
        )

        # Compute the amplitude pre-factor
        Amp0 = self.get_Amp0(f_Ms, eta_s) * (
            2.0 * torch.sqrt(self.FIVE / (64.0 * self.PI))
        )

        dist_s = (D * self.Mpc) / self.C

        return Amp0 * Amp * (M_s * M_s) / dist_s

    # Source: AmpInsAnsatz and init_amp_ins_prefactors
    # (LALSimIMRPhenomD_internals.c).
    def get_inspiral_Amp(self, f_Ms, chi1, chi2, eta_s, coeffs):
        """
        Inspiral amplitude plus rho phenom coefficients.

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, F) or (B, 1)
            Geometric frequencies.
        chi1 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the larger BH.
        chi2 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the smaller BH.
        eta_s : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients; columns 0 to 2 are rho1, rho2 and rho3.

        Returns
        -------
        Amp_Ins : torch.Tensor, same shape as f_Ms
            Inspiral amplitude without the amp0 prefactor.
        """
        eta2 = eta_s * eta_s
        eta3 = eta_s * eta2

        Seta = torch.sqrt(1.0 - 4.0 * eta_s)
        SetaPlus1 = 1.0 + Seta

        chi12 = chi1 * chi1
        chi22 = chi2 * chi2

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
        A7 = coeffs[:, 0:1]
        A8 = coeffs[:, 1:2]
        A9 = coeffs[:, 2:3]

        Amp_Ins = (
            A0
            # A1 is zero
            + A2 * (f_Ms ** (2.0 / 3.0))
            + A3 * f_Ms
            + A4 * (f_Ms ** (4.0 / 3.0))
            + A5 * (f_Ms ** (5.0 / 3.0))
            + A6 * (f_Ms**2.0)
            # rho phenom coefficients
            + A7 * (f_Ms ** (7.0 / 3.0))
            + A8 * (f_Ms ** (8.0 / 3.0))
            + A9 * (f_Ms**3.0)
        )

        return Amp_Ins

    # Source: AmpIntAnsatz and ComputeDeltasFromCollocation
    # (LALSimIMRPhenomD_internals.c).
    def get_IIa_Amp(self, f_Ms, fx_Ms, theta, derived, coeffs):
        """
        Ansatz for the intermediate amplitude.

        The delta coefficients follow from three collocation points: the
        amplitude transition frequencies and their midpoint.

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, F)
            Geometric frequencies.
        fx_Ms : torch.Tensor, shape (B, 8)
            Geometric fRef, f1, f2, f3, f4, fRD, fdamp and fmid.
        theta : torch.Tensor, shape (B, 5)
            Parameters; columns 2 and 3 are chi1 and chi2.
        derived : torch.Tensor, shape (B, 4)
            Derived parameters; column 3 is eta.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.

        Returns
        -------
        Amp_IIa : torch.Tensor, shape (B, F)
            Intermediate amplitude on f_Ms.
        """
        eta_s = derived[:, 3:4]
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]

        # Inspiral model and its derivative at the first point
        v1 = self.get_inspiral_Amp(fx_Ms[:, 3:4], chi1, chi2, eta_s, coeffs)
        d1 = self.DAmpInsAnsatz(fx_Ms[:, 3:4], chi1, chi2, eta_s, coeffs)

        # Merger-ringdown model and its derivative at the last point
        v3 = IMRPhenomD.get_IIb_Amp(fx_Ms[:, 4:5], fx_Ms, coeffs)
        d3 = IMRPhenomD.DAmpMRDAnsatz(fx_Ms[:, 4:5], coeffs, fx_Ms)

        # Now compute the delta_i's from the collocation coefficients
        delta0 = IMRPhenomD.get_delta0(
            fx_Ms[:, 3:4], fx_Ms[:, 7:8], fx_Ms[:, 4:5], v1, coeffs[:, 3:4], v3, d1, d3
        )
        delta1 = IMRPhenomD.get_delta1(
            fx_Ms[:, 3:4], fx_Ms[:, 7:8], fx_Ms[:, 4:5], v1, coeffs[:, 3:4], v3, d1, d3
        )
        delta2 = IMRPhenomD.get_delta2(
            fx_Ms[:, 3:4], fx_Ms[:, 7:8], fx_Ms[:, 4:5], v1, coeffs[:, 3:4], v3, d1, d3
        )
        delta3 = IMRPhenomD.get_delta3(
            fx_Ms[:, 3:4], fx_Ms[:, 7:8], fx_Ms[:, 4:5], v1, coeffs[:, 3:4], v3, d1, d3
        )
        delta4 = IMRPhenomD.get_delta4(
            fx_Ms[:, 3:4], fx_Ms[:, 7:8], fx_Ms[:, 4:5], v1, coeffs[:, 3:4], v3, d1, d3
        )

        Amp_IIa = (
            delta0
            + delta1 * f_Ms
            + delta2 * (f_Ms**2.0)
            + delta3 * (f_Ms**3.0)
            + delta4 * (f_Ms**4.0)
        )

        return Amp_IIa

    # Source: DAmpInsAnsatz (LALSimIMRPhenomD_internals.c).
    def DAmpInsAnsatz(self, f, chi1, chi2, eta, coeffs):
        """First frequency derivative of AmpInsAnsatz."""
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        SetaPlus1 = 1.0 + Seta
        chi12, chi22 = chi1 * chi1, chi2 * chi2
        eta2, eta3 = eta * eta, eta * eta**2

        PI_2_3 = self.PI ** (2.0 / 3.0)
        PI_4_3 = self.PI ** (4.0 / 3.0)
        PI_5_3 = self.PI ** (5.0 / 3.0)
        PI_2 = self.PI**2

        f_1_3 = f ** (1.0 / 3.0)
        f_2_3 = f ** (2.0 / 3.0)
        f_4_3 = f ** (4.0 / 3.0)
        f_5_3 = f ** (5.0 / 3.0)

        # Term 1: Derivative of A2 * f^(2/3)
        term1 = ((-969.0 + 1804.0 * eta) * PI_2_3) / (1008.0 * f_1_3)

        # Term 2: Derivative of A3 * f
        term2 = (
            (
                chi1 * (81.0 * SetaPlus1 - 44.0 * eta)
                + chi2 * (81.0 - 81.0 * Seta - 44.0 * eta)
            )
            * self.PI
        ) / 48.0

        # Term 3: Derivative of A4 * f^(4/3)
        term3_num = (
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
            * eta
            + 35371056.0 * eta2
        )
        term3 = (term3_num * f_1_3 * PI_4_3) / 6.096384e6

        # Term 4: Derivative of A5 * f^(5/3)
        term4_num = (
            chi2
            * (
                -285197.0 * (-1.0 + Seta)
                + 4.0 * (-91902.0 + 1579.0 * Seta) * eta
                - 35632.0 * eta2
            )
            + chi1
            * (
                285197.0 * SetaPlus1
                - 4.0 * (91902.0 + 1579.0 * Seta) * eta
                - 35632.0 * eta2
            )
            + 42840.0 * (-1.0 + 4.0 * eta) * self.PI
        )
        term4 = (5.0 * f_2_3 * PI_5_3 * term4_num) / 96768.0

        # Term 5: Derivative of A6 * f^2
        term5_num = (
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
                + 11087290368.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * self.PI
            )
            + 12.0
            * eta
            * (
                -545384828789.0
                - 176491177632.0 * chi1 * chi2
                + 202603761360.0 * chi22
                + 77616.0 * chi12 * (2610335.0 + 995766.0 * Seta)
                - 77287373856.0 * chi22 * Seta
                + 5841690624.0 * (chi1 + chi2) * self.PI
                + 21384760320.0 * PI_2
            )
        )
        term5 = -(f * PI_2 * term5_num) / 3.0042980352e10

        # Term 6: Derivative of the rho terms
        rho1, rho2, rho3 = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
        term_rho = (
            (7.0 / 3.0) * f_4_3 * rho1
            + (8.0 / 3.0) * f_5_3 * rho2
            + 3.0 * (f**2) * rho3
        )

        return term1 + term2 + term3 + term4 + term5 + term_rho

    # Source: DAmpMRDAnsatz (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def DAmpMRDAnsatz(f, coeffs, fx_Ms):
        """First frequency derivative of AmpMRDAnsatz."""
        fRD = fx_Ms[:, 5:6]
        fDM = fx_Ms[:, 6:7]
        gamma1 = coeffs[:, 4:5]
        gamma2 = coeffs[:, 5:6]
        gamma3 = coeffs[:, 6:7]

        fDMgamma3 = fDM * gamma3
        pow2_fDMgamma3 = fDMgamma3**2
        fminfRD = f - fRD

        expfactor = torch.exp(fminfRD * gamma2 / fDMgamma3)
        pow2pluspow2 = fminfRD**2 + pow2_fDMgamma3

        numerator = (-2 * fDM * fminfRD * gamma3 * gamma1 / pow2pluspow2) - (
            gamma2 * gamma1
        )
        denominator = expfactor * pow2pluspow2

        return numerator / denominator

    # Source: AmpMRDAnsatz (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_IIb_Amp(f_Ms, fx_Ms, coeffs):
        """
        Ansatz for the merger-ringdown amplitude.

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, F) or (B, 1)
            Geometric frequencies.
        fx_Ms : torch.Tensor, shape (B, 8)
            Geometric fRef, f1, f2, f3, f4, fRD, fdamp and fmid.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients; columns 4 to 6 are gamma1 to gamma3.

        Returns
        -------
        Amp_IIb : torch.Tensor, same shape as f_Ms
            Merger-ringdown amplitude on f_Ms.
        """
        gamma1 = coeffs[:, 4:5]
        gamma2 = coeffs[:, 5:6]
        gamma3 = coeffs[:, 6:7]

        fDMgamma3 = fx_Ms[:, 6:7] * gamma3
        fminfRD = f_Ms - fx_Ms[:, 5:6]
        Amp_IIb = (
            torch.exp(-(fminfRD) * gamma2 / (fDMgamma3))
            * (fDMgamma3 * gamma1)
            / ((fminfRD) ** 2.0 + (fDMgamma3) ** 2.0)
        )
        return Amp_IIb

    # Source: delta0_fun (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_delta0(f1, f2, f3, v1, v2, v3, d1, d3):
        """Coefficient delta0 of the intermediate amplitude ansatz."""
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

    # Source: delta1_fun (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_delta1(f1, f2, f3, v1, v2, v3, d1, d3):
        """Coefficient delta1 of the intermediate amplitude ansatz."""
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

    # Source: delta2_fun (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_delta2(f1, f2, f3, v1, v2, v3, d1, d3):
        """Coefficient delta2 of the intermediate amplitude ansatz."""
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

    # Source: delta3_fun (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_delta3(f1, f2, f3, v1, v2, v3, d1, d3):
        """Coefficient delta3 of the intermediate amplitude ansatz."""
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

    # Source: delta4_fun (LALSimIMRPhenomD_internals.c).
    @staticmethod
    def get_delta4(f1, f2, f3, v1, v2, v3, d1, d3):
        """Coefficient delta4 of the intermediate amplitude ansatz."""
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

    # Source: amp0Func and IMRPhenDAmplitude
    # (LALSimIMRPhenomD_internals.c).
    def get_Amp0(self, f_Ms, eta):
        """
        Amplitude scaling factor times Mf^(-7/6).

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, F)
            Geometric frequencies.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        Amp0 : torch.Tensor, shape (B, F)
            sqrt(2 eta / 3) pi^(-1/6) Mf^(-7/6).
        """
        Amp0 = (
            (2.0 / 3.0 * eta) ** (1.0 / 2.0)
            * (f_Ms) ** (-7.0 / 6.0)
            * self.PI ** (-1.0 / 6.0)
        )
        return Amp0

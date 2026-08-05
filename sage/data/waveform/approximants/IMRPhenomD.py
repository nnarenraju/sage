#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomD.py
Description     : Short description of the file

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
terms cover those derived portions only.  See ``README.md`` here for the
full provenance chain (LALSuite GPL-2+ -> MIT JAX reference -> this PyTorch
reimplementation, validated directly against LALSuite).

"""


# Packages
import torch

# LOCAL
from sage.core.torch import nudge_backward_
from sage.data.waveform.approximants import phenom
from sage.data.waveform import taper


class IMRPhenomD(phenom.PhenomConstants):
    """
    GPU-native batched IMRPhenomD frequency-domain waveform model.

    Implements the IMRPhenomD aligned-spin binary black hole waveform
    approximant entirely in PyTorch, allowing batch generation of ``(hp, hc)``
    polarisations on GPU without any Python-level loops.  Inherits all
    pre-allocated constants and QNM interpolation tables from
    :class:`~sage.data.waveform.approximants.phenom.PhenomConstants`.

    The waveform is computed on the frequency grid ``f`` supplied at
    construction time.  Low-frequency bins below ``f[0]`` are zero-padded
    so that the output arrays span ``[0, f_max]`` inclusive.

    Parameters
    ----------
    f : torch.Tensor, shape ``(B, F)``
        Frequency grid in Hz for the batch.
    f_ref : torch.Tensor, shape ``(B, 1)``
        Reference frequency for the phase calculation.
    **kwargs
        Forwarded to :class:`~sage.data.waveform.approximants.phenom.PhenomConstants`.
    """

    def __init__(self, f, f_ref, **kwargs):
        super().__init__(
            device=f.device,
            batch_size=f.shape[0],
            dtype=f.dtype,
        )
        # Fixed frequency grid
        self.f = f
        # Endpoint formula: avoids catastrophic cancellation when f_l >> del_f.
        _n = self.f[0].numel()
        self.df = (f[0, -1] - f[0, 0]) / (_n - 1)
        self.sample_length_in_s = 1.0 / self.df
        self.f_numel = _n
        self.f_ref = f_ref
        # Batch size
        self.B = f.shape[0]
        # Tensor of zeroes for hp and hc
        # Accounts for freqs from DC to f_upper
        self.n_pad = int(torch.round((self.f[0][0] - self.df) / self.df)) + 1
        self.hp_buffer = torch.empty(
            (self.B, self.n_pad + self.f_numel),
            dtype=torch.complex128,
            device=f.device,
        )
        self.hc_buffer = torch.empty_like(self.hp_buffer)

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute the FD plus and cross polarisations for a parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape ``(B, P)``
            Batch of waveform parameters: ``[m1, m2, chi1, chi2, distance,
            tc, inclination, ...]``.
        reproduce_lal : bool
            If ``True``, skip FD tapering and ``tc`` application to
            reproduce raw LALSuite output (default ``False``).

        Returns
        -------
        hp, hc : torch.Tensor, shape ``(B, F)`` complex
            Plus and cross polarisations in the frequency domain.
        """
        # Compute derived quantities
        derived = self.compute_derived_parameters(theta)

        # theta = {m1, m2, var2, var3, var4, var5, var6, ...}
        # First four are intrinsic, next 3 are extrinsic
        coeffs = self.get_coeffs(theta[:, 2:3], theta[:, 3:4], derived[:, 3:4])
        A, Psi, fcut_true = self.get_components(theta, coeffs, derived)

        # Compute hp and hc
        # Replacing complex exp operations with polar
        # 1J still exists but math ops are fully real
        # Complex exp are not supported by TorchInductor
        # hp = h0 * (1 / 2 * (1 + torch.cos(theta[:, 7:8]) ** 2))
        # hc = -self.ONE_J * h0 * torch.cos(theta[:, 7:8])
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
                f_min=20.0,
                f_cut=fcut_true,
                df=self.df,
            )
            hp *= _taper
            hc *= _taper

            # Apply phase shift equivalent to applying tc
            hp, hc = self.apply_tc(hp, hc, theta[:, 5:6])

            # Make hf consistent with the scale of other data
            # LAL works in continuous Fourier regime
            hp *= self.df
            hc *= self.df

        # Pad missing frequencies from DC to f_low
        # Pad *AFTER* taper; not before
        hp, hc = self.pad_missing_frequencies(hp, hc)

        return hp, hc

    def apply_tc(self, hp, hc, tc):
        """Apply a frequency-domain phase shift equivalent to a time-domain shift by *tc*."""
        # Apply time shift to account for tc
        # Converting from tc in duration space to actual shift
        _tc = tc - self.sample_length_in_s
        # We do this in polar as well without torch exp
        hp = torch.polar(torch.abs(hp), torch.angle(hp) - 2 * self.PI * self.f * _tc)
        hc = torch.polar(torch.abs(hc), torch.angle(hc) - 2 * self.PI * self.f * _tc)
        return hp, hc

    def pad_missing_frequencies(self, hp, hc):
        """Zero-pad *hp* and *hc* below ``f_min`` (DC to the starting frequency)."""
        # Accounting for DC components and zero-padding below f_min
        # We start from 0 Hz, df Hz, 2df Hz; not including f_min
        # Assuming f_min included in fs
        # This accounts for LAL-like handlings of f
        hp_pad = torch.zeros_like(self.hp_buffer)
        hc_pad = torch.zeros_like(self.hc_buffer)
        # Fill empty buffer with hp and hc
        hp_pad[:, self.n_pad :] = hp
        hc_pad[:, self.n_pad :] = hc

        return hp_pad, hc_pad

    def compute_derived_parameters(self, theta):
        """
        Compute mass-related derived quantities from the parameter batch.

        Returns ``[m1_s, m2_s, M_s, eta_s]`` where each column uses SI-scaled
        masses (``m * G / c³``).
        """
        # Derived parameters are reused a lot; compute once
        # Putting this in self (now) is unfortunately detrimental
        # torch.compile thinks class object is mutating dynamically
        # NOTE: Mutating contents of a preallocated tensor in self is fine
        m1_s = theta[:, 0:1] * self.GM
        m2_s = theta[:, 1:2] * self.GM
        M_s = m1_s + m2_s
        eta_s = m1_s * m2_s / (M_s * M_s)
        # Clip to 0.25 (mirrors LALSim: "if(eta > 0.25) eta = 0.25" — corrects
        # floating-point drift above the equal-mass limit, does NOT nudge below).
        # No division by Seta = sqrt(1-4*eta) exists in the fit functions, so
        # eta = 0.25 exactly is safe.
        eta_s.clamp_(max=0.25)

        return torch.cat([m1_s, m2_s, M_s, eta_s], dim=1)

    def get_coeffs(self, chi1, chi2, eta):
        """
        Compute the IMRPhenomD phenomenological coefficient matrix.

        Builds monomials in ``(chiPN - 1)`` and ``eta`` up to third order,
        then multiplies by the pre-loaded coefficient table to produce a
        ``(B, N_coeffs)`` tensor of phenomenological fitting coefficients.
        """
        # Definition of chiPN from lalsuite
        chi_s = (chi1 + chi2) / 2.0
        chi_a = (chi1 - chi2) / 2.0
        seta = torch.sqrt(self.ONE - 4 * eta)
        chiPN = chi_s * (self.ONE - 76 * eta / 113) + seta * chi_a

        # chi powers
        chi0 = self.ONES
        chi1 = chiPN - self.ONE
        chi2 = chi1**2
        chi3 = chi1**3

        # eta powers
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

        # torch cat/stack is compile friendly
        coeff = powers @ self.PhenomD_coeff_table.T

        return coeff

    def get_components(self, theta, coeffs, derived):
        """
        Compute the amplitude *A*, phase *Psi*, and frequency cutoff *fcut_true*.

        Evaluates the full IMRPhenomD frequency-domain waveform components from
        the parameter batch, phenomenological coefficients, and derived mass
        quantities.  Returns tensors ready for combining into h+ and hx.
        """
        ## Shift phase so that peak amplitude matches t = 0
        # Get required derived quantities
        M_s = derived[:, 2:3]

        # Compute transition frequencies
        # f1, f2, f3, f4, f_RD, f_damp
        trans_fs = self.get_transition_frequencies(
            theta[:, :4], derived, coeffs[:, 5:6], coeffs[:, 6:7]
        )

        # Precomputing required parameters
        f1_Ms = trans_fs[:, 0:1] * M_s
        f2_Ms = trans_fs[:, 1:2] * M_s
        f3_Ms = trans_fs[:, 2:3] * M_s
        f4_Ms = trans_fs[:, 3:4] * M_s
        f_Ms = self.f * M_s
        fref_Ms = self.f_ref * M_s
        f_RD_Ms = trans_fs[:, 4:5] * M_s
        f_damp_Ms = trans_fs[:, 5:6] * M_s
        # Central frequency point (used f_RD and f_damp)
        fmid_Ms = ((trans_fs[:, 2:3] + trans_fs[:, 3:4]) / 2) * M_s
        fx_Ms = torch.cat(
            [fref_Ms, f1_Ms, f2_Ms, f3_Ms, f4_Ms, f_RD_Ms, f_damp_Ms, fmid_Ms], dim=1
        )

        # f4_scaled does *not* need requires_grad_()
        f4_scaled = trans_fs[:, 3:4] * M_s
        # Calling autograd here breaks compile; but the following is slow too.
        # The following is a torch grad way of doing the analytical version.
        # But this is much slower than the analytical version; use it as sanity check.
        # y, vjp_fn = vjp(get_IIb_raw_phase, f4_scaled, derived[:, 3:4], coeffs, fx_Ms)
        # Multiply by ones to get gradient per waveform
        # t0 = vjp_fn(torch.ones_like(y))[0]  # shape (B,1)
        # We can instead compute the derivative analytically.
        t0 = IMRPhenomD.DPhiMRD(f4_scaled, coeffs, derived[:, 3:4], fx_Ms)

        ## Phase and Amplitude
        # Phase computation
        Psi = self.phase(theta[:, :4], coeffs, derived, f_Ms, fx_Ms)
        Psi_ref = self.phase(theta[:, :4], coeffs, derived, fref_Ms, fx_Ms)
        Psi -= t0 * ((f_Ms) - fref_Ms) + Psi_ref
        # Originally this term included tc and phic contribution
        # We remove tc here to explicitly apply it later when needed
        # ext_phase_contrib = self.TWOPI * self.f * theta[:, 5:6] - 2 * theta[:, 6:7]
        ext_phase_contrib = -2 * theta[:, 6:7]
        Psi += ext_phase_contrib

        # And now we can combine them by multiplying by a set of heaviside functions
        fcut_true = self.get_fcut_true(M_s)

        # Get psi and amplitude
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

        # Replacing complex exp with polar
        # Complex exp are not supported by TorchInductor
        # h0 = A * torch.exp(self.ONE_J * -Psi)
        # h0 = torch.polar(A, -Psi)

        return A, Psi, fcut_true

    @staticmethod
    def DPhiMRD(f, coeffs, eta, fx_Ms, Rholm: float = 1.0, Taulm: float = 1.0):
        """
        First frequency derivative of PhiMRDAnsatzInt

        Args:
            f: Tensor of frequencies, shape (B,1) or (B,)
            p: object with attributes alpha1, alpha2,
            alpha3, alpha4, alpha5, fDM, fRD, etaInv
            Rholm: ratio of fRD22/fRDlm, default 1.0
            Taulm: ratio of damping times, default 1.0

        Returns:
            Tensor of same shape as f
        """
        f2 = f**2

        term1 = coeffs[:, 14:15]
        term2 = coeffs[:, 15:16] / f2
        term3 = coeffs[:, 16:17] / torch.pow(f, 0.25)

        denom = (
            fx_Ms[:, 6:7]
            * Taulm
            * (
                1
                + (f - coeffs[:, 18:19] * fx_Ms[:, 5:6]) ** 2
                / ((fx_Ms[:, 6:7] * Taulm * Rholm) ** 2)
            )
        )
        term4 = coeffs[:, 17:18] / denom

        return (term1 + term2 + term3 + term4) * (1.0 / eta)

    def get_transition_frequencies(self, theta, derived, gamma2, gamma3):
        """
        Return the six IMRPhenomD transition frequencies
        ``[f1, f2, f3, f4, f_RD, f_damp]`` in Hz for a parameter batch.
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
        # Compute both branches
        sqrt_term = torch.sqrt(torch.clamp(1.0 - gamma2**2, min=0.0))

        f4_gammaneg_gtr_1 = torch.abs(f_RD + (-f_damp * gamma3) / gamma2)
        f4_gammaneg_less_1 = torch.abs(
            f_RD + (f_damp * (-1 + sqrt_term) * gamma3) / gamma2
        )

        # Select based on condition
        f4 = torch.where(gamma2 >= 1, f4_gammaneg_gtr_1, f4_gammaneg_less_1)

        return torch.cat([f1, f2, f3, f4, f_RD, f_damp], dim=1)

    def get_fRD_fdamp(self, chi1, chi2, m1_s, m2_s, M_s, eta_s):
        """
        Return the ringdown frequency and damping frequency by interpolating
        the pre-loaded QNM table with the final-spin estimate.
        """
        # Compute Kerr-like total angular momentum
        S = (chi1 * m1_s * m1_s + chi2 * m2_s * m2_s) / (M_s * M_s)
        # Get phenomenological effective spin
        a = IMRPhenomD.final_spin_0815_s(eta_s, S)
        Erad = IMRPhenomD.erad_rational_0815(eta_s, chi1, chi2)

        # Compute relative position in grid
        # We could precompute slope and intercept but this is faster
        # We have uniform indexing on QNMData_a which allows this to work
        rel_idx = (a - self.QNMData_a[0]) / (self.QNMData_a[1] - self.QNMData_a[0])
        idx_lower = rel_idx.floor().long().clamp(0, len(self.QNMData_a) - 2)
        frac = rel_idx - idx_lower.float()

        fRD = (
            self.QNMData_fRD[idx_lower] * (1 - frac)
            + self.QNMData_fRD[idx_lower + 1] * frac
        )
        fdamp = (
            self.QNMData_fdamp[idx_lower] * (1 - frac)
            + self.QNMData_fdamp[idx_lower + 1] * frac
        )

        factor = 1.0 / (1.0 - Erad)
        fRD *= factor
        fdamp *= factor

        return fRD / M_s, fdamp / M_s

    @staticmethod
    def final_spin_0815_s(eta, S):
        """Phenomological final-spin fit (Eq. 3.6, arXiv:1508.07250)."""
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
        """Rational-function fit for the dimensionless radiated mass (arXiv:1508.07250)."""
        # Compute the dimensionless mass fractions
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        m1f = 0.5 * (1.0 + Seta)
        m2f = 0.5 * (1.0 - Seta)
        # Compute standard Phenom effective spin combination
        m1sf = m1f * m1f
        m2sf = m2f * m2f
        # LALSim PhenomInternal_EradRational0815 (line 320):
        # double s = (m1s * chi1 + m2s * chi2) / (m1s + m2s);
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
        """IMRPhenomD intermediate-region IIa phase including the ``beta1`` continuity correction."""
        return IMRPhenomD.get_IIa_raw_phase(f_Ms, eta, coeffs) + beta1corr * f_Ms

    @staticmethod
    def get_IIa_raw_phase(f_Ms, eta, coeffs):
        """Raw intermediate region IIa phase ansatz (without continuity correction)."""
        phi_IIa_raw = (
            coeffs[:, 11:12] * f_Ms
            + coeffs[:, 12:13] * torch.log(f_Ms)
            - coeffs[:, 13:14] * (f_Ms**-3.0) / 3.0
        ) / eta

        return phi_IIa_raw

    @staticmethod
    def get_IIb_raw_phase(f_Ms, eta, coeffs, fx_Ms):
        """Raw merger-ringdown region IIb phase ansatz."""
        phi_IIb_raw = (
            coeffs[:, 14:15] * f_Ms
            - coeffs[:, 15:16] * (f_Ms**-1.0)
            + 4.0 * coeffs[:, 16:17] * (f_Ms ** (3.0 / 4.0)) / 3.0
            + coeffs[:, 17:18]
            * torch.arctan((f_Ms - coeffs[:, 18:19] * fx_Ms[:, 5:6]) / fx_Ms[:, 6:7])
        ) / eta

        return phi_IIb_raw

    def phase(self, theta, coeffs, derived, f_Ms, fx_Ms):
        """
        Compute the full IMRPhenomD gravitational-wave phase across all frequency regions.

        Stitches together inspiral, intermediate (IIa), and merger-ringdown
        (IIb) phase contributions with C¹-continuity corrections.
        """
        # Compute inspiral phase
        # Only intrinsic theta has been passed
        phi6corr = self.spin_spin_3pn_correction(theta, derived[:, 3:4])
        phi_Ins, _, _ = self.get_inspiral_phase(
            f_Ms, derived, theta[:, 2:4], coeffs, phi6corr
        )

        # Phase of the late inspiral (region IIa)
        # beta0 is found by matching the phase between the region I and IIa
        # C(1) continuity must be preserved. We therefore need to solve for an
        # additional contribution to beta1

        ## Evaluate phase and derivative at f1
        # Implementing DPhiInsAnsatzInt from LAL
        phi_Ins_f1, TF2_coeffs, TF2_log_coeffs = self.get_inspiral_phase(
            fx_Ms[:, 1:2], derived, theta[:, 2:4], coeffs, phi6corr
        )
        dphi_Ins_f1 = self.DPhiInsAnsatzInt(
            fx_Ms[:, 1:2], coeffs, TF2_coeffs, TF2_log_coeffs, derived[:, 3:4]
        )

        # Implementing DPhiIntAnsatz from LAL
        phi_IIa_f1 = IMRPhenomD.get_IIa_raw_phase(
            fx_Ms[:, 1:2], derived[:, 3:4], coeffs
        )
        dphi_IIa_f1 = IMRPhenomD.DPhiIntAnsatz(fx_Ms[:, 1:2], coeffs, derived[:, 3:4])

        beta1corr = dphi_Ins_f1 - dphi_IIa_f1
        beta0 = phi_Ins_f1 - beta1corr * fx_Ms[:, 1:2] - phi_IIa_f1
        phi_IIa = (
            IMRPhenomD.get_phi_IIa(f_Ms, derived[:, 3:4], coeffs, beta1corr) + beta0
        )

        # Phase of the merger-ringdown (region IIb)
        # Evaluate phase and derivative at f2
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

        # Combine all regions
        # This is equivalent to the heaviside combine that Ripple had
        # But at frequencies f == f1 and f == f2, we pick one side
        # instead of weighting each side by half.
        # NOTE: Given machine-precision, this should rarely matter
        # Upside, this is faster when vectorised than heaviside
        phase = torch.where(
            f_Ms <= fx_Ms[:, 1:2],
            phi_Ins,
            torch.where(f_Ms <= fx_Ms[:, 2:3], phi_IIa, phi_IIb),
        )

        return phase

    def DPhiInsAnsatzInt(
        self,
        fxi_Ms,
        coeffs,
        TF2_coeffs,
        TF2_log_coeffs,
        eta,
    ):
        """
        First frequency derivative of PhiInsAnsatzInt

        Args:
            Mf: Tensor, shape (B,1) or (B,)
            coeffs: Tensor, shape (B, Nc)
            pn_v: Tensor, shape (B, 8)
            pn_vlogv: Tensor, shape (B, 8)

        Returns:
            Tensor of same shape as Mf
        """

        # Extract calibrated coefficients
        sigma1 = coeffs[:, 7:8]
        sigma2 = coeffs[:, 8:9]
        sigma3 = coeffs[:, 9:10]
        sigma4 = coeffs[:, 10:11]

        # PN velocity v = (pi * Mf)^(1/3)
        v = torch.pow(self.PI * fxi_Ms, self.ONE_BY_THREE)
        logv = torch.log(v)

        v2 = v * v
        v3 = v * v2
        v4 = v * v3
        v5 = v * v4
        v6 = v * v5
        v7 = v * v6
        v8 = v * v7

        # Assemble PN phasing derivative
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

        # Add PhenomD calibrated higher-order terms
        Dphasing += (
            sigma1
            + sigma2 * v * (self.PI ** (-1.0 / 3.0))
            + sigma3 * v2 * (self.PI ** (-2.0 / 3.0))
            + sigma4 * v3 * (1.0 / self.PI)
        ) * (1.0 / eta)

        return Dphasing

    @staticmethod
    def DPhiIntAnsatz(fxi_Ms, coeffs, eta):
        """
        First frequency derivative of PhiIntAnsatz

        Args:
            Mf:     tensor of shape (B, 1) or (B,)
            coeffs: tensor of shape (B, N) containing beta coefficients
            eta:    tensor of shape (B, 1) or (B,)

        Returns:
            Tensor of same shape as Mf
        """

        Mf4 = fxi_Ms**4

        beta1 = coeffs[:, 11:12]  # p->beta1
        beta2 = coeffs[:, 12:13]  # p->beta2
        beta3 = coeffs[:, 13:14]  # p->beta3

        return (beta1 + beta3 / Mf4 + beta2 / fxi_Ms) * (1.0 / eta)

    @staticmethod
    def DPhiIntTemp(fxi_Ms, coeffs, eta, beta1corr):
        """
        Temporary first frequency derivative of PhiIntAnsatz
        used to enforce C(1) continuity between regions.

        Args:
            ff:     Tensor of shape (B, 1) or (B,)
            coeffs: Tensor of shape (B, N) with PhenomD coefficients
            eta:    Tensor of shape (B, 1) or (B,)

        Returns:
            Tensor of same shape as ff
        """

        ff4 = fxi_Ms**4

        beta1 = coeffs[:, 11:12]  # p->beta1
        beta2 = coeffs[:, 12:13]  # p->beta2
        beta3 = coeffs[:, 13:14]  # p->beta3
        # p->C2Int is beta1corr in our code

        return beta1corr + (beta1 + beta3 / ff4 + beta2 / fxi_Ms) * (1.0 / eta)

    @staticmethod
    def DPhiMRD(f, coeffs, eta, fx_Ms, Rholm=1.0, Taulm=1.0):
        """
        First frequency derivative of PhiMRDAnsatzInt

        Args:
            f:      Tensor (B, 1) or (B,)
            coeffs: Tensor (B, N) containing PhenomD phase coefficients
            eta:    Tensor (B, 1) or (B,)
            fx_Ms:  Tensor (B, K) containing fRD, fDM, etc.
            Rholm:  scalar (default 1.0)
            Taulm:  scalar (default 1.0)

        Returns:
            Tensor of shape (B, 1) or (B,)
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

    def spin_spin_3pn_correction(self, theta, eta_s):
        """
        Compute the 3PN spin-spin TaylorF2 correction term subtracted from
        the PhenomD inspiral phase (LALSimInspiralPNCoefficients.c, v[6]).
        """
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

    def get_inspiral_phase(self, fxi_Ms, derived, chi, coeffs, phi6corr):
        """
        Calculate the inspiral phase for IMRPhenomD exactly equivalent to LAL.
        """
        # Unpack derived parameters
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
        chi12 = chi1 * chi2  # chi1 * chi2 for aligned spin is chi1 . chi2

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
        phi6 += ((326.75 / 1.12 + 557.5 / 1.8 * eta_s) * eta_s) * chi1 * chi2  # S1S2O
        phi6 += (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M**2) * m1M**2
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M**2) * m1M**2
        ) * chi1sq
        phi6 += (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M**2) * m2M**2
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M**2) * m2M**2
        ) * chi2sq
        # Subtract 3PN SS correction as done in driver
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

        # Frequency-domain assembly
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

        # Add higher-order phenomenological sigma terms
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

    def get_fcut_true(self, M_s):
        """Return the physical frequency cutoff in Hz from the dimensionless ``fM_CUT``."""
        # mask = self.f[None, :] <= fcut[:, None]
        # f_grid = self.f[None, :].expand_as(mask)
        # return torch.max(torch.where(mask, f_grid, -torch.inf), dim=1).values
        return self.fM_CUT / M_s

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
        Computes the amplitude of the PhenomD frequency domain waveform
        Refer 1508.07253 for more details
        Note that this waveform also assumes that object one is the more massive.
        """

        # Required vars
        M_s = derived[:, 2:3]
        eta_s = derived[:, 3:4]
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]
        D = theta[:, 4:5]

        # First we get the inspiral amplitude
        Amp_Ins = self.get_inspiral_Amp(f_Ms, chi1, chi2, eta_s, coeffs)

        # Next lets construct the phase of the late inspiral (region IIa)
        # Note that this part is a little harder since we need to
        # solve a system of equations for deltas
        Amp_IIa = self.get_IIa_Amp(f_Ms, fx_Ms, theta, derived, coeffs)

        # And finally, we construct the amplitude of the merger-ringdown (region IIb)
        Amp_IIb = IMRPhenomD.get_IIb_Amp(f_Ms, fx_Ms, coeffs)

        # Check for fcut_true
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

        # Prefactor (This second factor is from lalsuite)
        Amp0 = self.get_Amp0(f_Ms, eta_s) * (
            2.0 * torch.sqrt(self.FIVE / (64.0 * self.PI))
        )

        # Need to add in an overall scaling of M_s^2 to make the units correct
        dist_s = (D * self.Mpc) / self.C

        return Amp0 * Amp * (M_s * M_s) / dist_s

    def get_inspiral_Amp(self, f_Ms, chi1, chi2, eta_s, coeffs):
        """
        Compute the TaylorF2-like inspiral amplitude ansatz (Region I).

        Implements the PN series A0…A6 plus three fitted coefficients A7–A9
        from LALSimIMRPhenomD_internals.c (lines 302–351).

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, n_freq)
            Dimensionless frequency grid f × M_s.
        chi1 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the larger BH.
        chi2 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the smaller BH.
        eta_s : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        coeffs : torch.Tensor, shape (B, 7+)
            PhenomD fit coefficients; columns 0–2 are A7, A8, A9.

        Returns
        -------
        Amp_Ins : torch.Tensor, shape (B, n_freq)
            Inspiral amplitude (unnormalised, dimensionless).
        """
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
        A7 = coeffs[:, 0:1]
        A8 = coeffs[:, 1:2]
        A9 = coeffs[:, 2:3]

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

    def get_IIa_Amp(self, f_Ms, fx_Ms, theta, derived, coeffs):
        """
        Compute the intermediate (IIa) amplitude via the quintic polynomial ansatz.

        Solves for the five delta coefficients by matching the values and first
        derivatives of the inspiral and merger-ringdown amplitudes at the
        transition frequencies f1 and f3, then evaluates the quintic.

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, n_freq)
            Dimensionless frequency grid for the IIa region.
        fx_Ms : torch.Tensor, shape (B, 8)
            Special frequency scale products (fref, f1, f2, f3, f4, fRD, fdamp, fmid).
        theta : torch.Tensor, shape (B, 5+)
            Waveform parameters; columns 2–3 are chi1, chi2.
        derived : torch.Tensor, shape (B, 4+)
            Derived parameters; column 3 is eta.
        coeffs : torch.Tensor, shape (B, 7+)
            PhenomD fit coefficients.

        Returns
        -------
        Amp_IIa : torch.Tensor, shape (B, n_freq)
            Intermediate amplitude on f_Ms.
        """
        # Required vars
        # f1, f3, f_RD, f_damp
        eta_s = derived[:, 3:4]
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]

        # For this region, we also need to calculate the the values and derivatives
        # of the Ins and IIb regions

        ## Evaluate amplitude and derivative
        # Derivative of the inspiral amplitude
        v1 = self.get_inspiral_Amp(fx_Ms[:, 3:4], chi1, chi2, eta_s, coeffs)
        d1 = self.DAmpInsAnsatz(fx_Ms[:, 3:4], chi1, chi2, eta_s, coeffs)

        # Derivative of the merger-ringdown amplitude
        v3 = IMRPhenomD.get_IIb_Amp(fx_Ms[:, 4:5], fx_Ms, coeffs)
        d3 = IMRPhenomD.DAmpMRDAnsatz(fx_Ms[:, 4:5], coeffs, fx_Ms)

        # Here we need the delta solutions
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

    def DAmpInsAnsatz(self, f, chi1, chi2, eta, coeffs):
        """
        Analytical derivative of the Inspiral Amplitude.
        Matches LALSimIMRPhenomD_internals.c: DAmpInsAnsatz.
        """
        # 1. Setup local variables and constants from C code
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        SetaPlus1 = 1.0 + Seta
        chi12, chi22 = chi1 * chi1, chi2 * chi2
        eta2, eta3 = eta * eta, eta * eta**2

        # Numerical constants from C source
        PI_2_3 = self.PI ** (2.0 / 3.0)
        PI_4_3 = self.PI ** (4.0 / 3.0)
        PI_5_3 = self.PI ** (5.0 / 3.0)
        PI_2 = self.PI**2

        # Frequency powers for the derivative calculation
        f_1_3 = f ** (1.0 / 3.0)
        f_2_3 = f ** (2.0 / 3.0)
        f_4_3 = f ** (4.0 / 3.0)
        f_5_3 = f ** (5.0 / 3.0)

        # 2. Ported Terms from LAL Implementation

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

        # Term 6: Calibrated Phenom Rho terms (A7, A8, A9 in your code)
        rho1, rho2, rho3 = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
        term_rho = (
            (7.0 / 3.0) * f_4_3 * rho1
            + (8.0 / 3.0) * f_5_3 * rho2
            + 3.0 * (f**2) * rho3
        )

        return term1 + term2 + term3 + term4 + term5 + term_rho

    @staticmethod
    def DAmpMRDAnsatz(f, coeffs, fx_Ms):
        """
        EXACT analytical derivative of the MRD Amplitude.
        Matches LALSimIMRPhenomD_internals.c: DAmpMRDAnsatz exactly.
        """
        # Unpack from your specific tensor structures
        fRD = fx_Ms[:, 5:6]
        fDM = fx_Ms[:, 6:7]
        gamma1 = coeffs[:, 4:5]
        gamma2 = coeffs[:, 5:6]
        gamma3 = coeffs[:, 6:7]

        # Pre-calculations matching C code [3, 4]
        fDMgamma3 = fDM * gamma3
        pow2_fDMgamma3 = fDMgamma3**2
        fminfRD = f - fRD

        # Note: expfactor is positive in the denominator to represent e^-x [3, 4]
        expfactor = torch.exp(fminfRD * gamma2 / fDMgamma3)
        pow2pluspow2 = fminfRD**2 + pow2_fDMgamma3

        # Analytical Derivative Formula [4]
        numerator = (-2 * fDM * fminfRD * gamma3 * gamma1 / pow2pluspow2) - (
            gamma2 * gamma1
        )
        denominator = expfactor * pow2pluspow2

        return numerator / denominator

    @staticmethod
    def get_IIb_Amp(f_Ms, fx_Ms, coeffs):
        """
        Compute the merger-ringdown (IIb) amplitude via the Lorentzian ansatz.

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, n_freq)
            Dimensionless frequency grid for the IIb region.
        fx_Ms : torch.Tensor, shape (B, 8)
            Special frequency scale products; columns 5–6 are fRD·M_s and fdamp·M_s.
        coeffs : torch.Tensor, shape (B, 7+)
            PhenomD coefficients; columns 4–6 are gamma1, gamma2, gamma3.

        Returns
        -------
        Amp_IIb : torch.Tensor, shape (B, n_freq)
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

    @staticmethod
    def get_delta0(f1, f2, f3, v1, v2, v3, d1, d3):
        """Compute the δ₀ coefficient of the IIa quintic amplitude polynomial."""
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
        """Compute the δ₁ coefficient of the IIa quintic amplitude polynomial."""
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
        """Compute the δ₂ coefficient of the IIa quintic amplitude polynomial."""
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
        """Compute the δ₃ coefficient of the IIa quintic amplitude polynomial."""
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
        """Compute the δ₄ coefficient of the IIa quintic amplitude polynomial."""
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
        """
        Compute the overall GW amplitude prefactor A₀(f, η).

        This is the leading-order Newtonian factor that scales the full
        PhenomD amplitude: A₀ = (2η/3)^(1/2) × (fM_s)^(-7/6) × π^(-1/6).

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, n_freq)
            Dimensionless frequency f × M_s.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        Amp0 : torch.Tensor, shape (B, n_freq)
            Newtonian amplitude prefactor.
        """
        Amp0 = (
            (2.0 / 3.0 * eta) ** (1.0 / 2.0)
            * (f_Ms) ** (-7.0 / 6.0)
            * self.PI ** (-1.0 / 6.0)
        )
        return Amp0

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXAS.py
Description   : GPU-native batched IMRPhenomXAS aligned-spin frequency-domain
                waveform model (BBH baseline, no tidal).

                Implements the IMRPhenomX aligned-spin (22-mode) waveform
                entirely in PyTorch, following García-Quirós et al. (2020),
                arXiv:2001.10914.  Inherits device-resident constants from
                PhenomConstants and adds XAS-specific QNM tables and final
                mass/spin fits (Jimenez-Forteza et al. 2017, arXiv:1611.00332).

                This class is the BBH backbone.  Do not use it directly for
                BNS/NSBH — use IMRPhenomXAS_NRTidalv3 instead.

                Parameters (theta columns)
                --------------------------
                0  : m1          (solar masses, m1 >= m2)
                1  : m2          (solar masses)
                2  : chi1z       (dimensionless aligned spin of body 1)
                3  : chi2z       (dimensionless aligned spin of body 2)
                4  : distance    (Mpc)
                5  : tc          (s, time of coalescence)
                6  : phic        (rad, reference orbital phase)
                7  : inclination (rad)

Created on 2026-05-27

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

References
----------
IMRPhenomXAS  : García-Quirós et al. (2020), arXiv:2001.10914
Final spin/mass: Jimenez-Forteza et al. (2017), arXiv:1611.00332
QNM tables    : Berti, Cardoso & Will (2009), CQG 26, 163001; arXiv:0905.2975
"""

import math

import torch

from sage.data.waveform.approximants.phenom import PhenomConstants
from sage.data.waveform.approximants.phenomx_data import (
    _XAS_QNMData_a,
    _XAS_QNMData_fring22,
    _XAS_QNMData_fdamp22,
)
from sage.core.interpolation import torch_scipylike_cubic_interp
from sage.core.torch import nudge_backward_


class IMRPhenomXAS(PhenomConstants):
    """
    GPU-native batched IMRPhenomXAS aligned-spin BBH waveform.

    Inherits scalar constants and PhenomD QNM tables from PhenomConstants.
    Adds XAS-specific 1200-point QNM tables pre-interpolated to a fine
    uniform grid for O(1) linear-interpolation at runtime.

    Parameters
    ----------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.
    **kwargs
        Forwarded to PhenomConstants.
    """

    def __init__(self, f, f_ref, **kwargs):
        super().__init__(
            device=f.device,
            batch_size=f.shape[0],
            dtype=f.dtype,
            **kwargs,
        )

        self.f      = f
        self.df     = f[0, 1] - f[0, 0]
        self.sample_length_in_s = 1.0 / self.df
        self.f_numel = f.shape[1]
        self.f_ref  = f_ref
        self.B      = f.shape[0]

        # Pre-allocate output buffers (complex128 matches LAL double precision)
        self.n_pad = int(torch.round((self.f[0, 0] - self.df) / self.df).item()) + 1
        self.hp_buffer = torch.empty(
            (self.B, self.n_pad + self.f_numel),
            dtype=torch.complex128,
            device=f.device,
        )
        self.hc_buffer = torch.empty_like(self.hp_buffer)

        # XAS-specific QNM tables (1200 pts, uniform spin grid [-1, 1])
        # Pre-interpolate to a 500k-point grid identical to the PhenomD approach,
        # so that the fast O(1) linear-interp trick works unchanged.
        self._xas_QNMData_a      = _XAS_QNMData_a.to(device=f.device, dtype=f.dtype)
        self._xas_QNMData_fring22 = _XAS_QNMData_fring22.to(device=f.device, dtype=f.dtype)
        self._xas_QNMData_fdamp22 = _XAS_QNMData_fdamp22.to(device=f.device, dtype=f.dtype)

        self.xas_QNMData_a = torch.linspace(
            -1.0, 1.0, 500_000, device=f.device, dtype=f.dtype
        )
        self.xas_QNMData_fring22 = torch_scipylike_cubic_interp(
            self.xas_QNMData_a, self._xas_QNMData_a, self._xas_QNMData_fring22
        )
        self.xas_QNMData_fdamp22 = torch_scipylike_cubic_interp(
            self.xas_QNMData_a, self._xas_QNMData_a, self._xas_QNMData_fdamp22
        )

        # Frequency cutoff: Mf = 0.3 (same as PhenomD convention)
        self.fM_CUT_XAS = torch.tensor(0.3, device=f.device, dtype=f.dtype)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a BBH parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 8+)
            [m1, m2, chi1z, chi2z, distance, tc, phic, inclination, ...]
        reproduce_lal : bool
            Skip tapering/tc/df normalisation to match raw LAL output.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, F), complex
        """
        raise NotImplementedError("get_hphc not yet implemented")

    def apply_tc(self, hp, hc, tc):
        """Apply a frequency-domain time-shift by tc seconds."""
        _tc = tc - self.sample_length_in_s
        hp = torch.polar(torch.abs(hp), torch.angle(hp) - 2 * self.PI * self.f * _tc)
        hc = torch.polar(torch.abs(hc), torch.angle(hc) - 2 * self.PI * self.f * _tc)
        return hp, hc

    def pad_missing_frequencies(self, hp, hc):
        """Zero-pad hp/hc from DC to f_min."""
        hp_pad = torch.zeros_like(self.hp_buffer)
        hc_pad = torch.zeros_like(self.hc_buffer)
        hp_pad[:, self.n_pad:] = hp
        hc_pad[:, self.n_pad:] = hc
        return hp_pad, hc_pad

    def get_fcut(self, M_s):
        """Physical frequency cutoff in Hz from Mf_CUT = 0.3."""
        return self.fM_CUT_XAS / M_s

    # ------------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------------

    def compute_derived_parameters(self, theta):
        """
        Compute mass and spin derived quantities from the parameter batch.

        Convention: m1 >= m2 (enforced by the parameter sampler).
        Mirrors IMRPhenomXSetWaveformVariables in LALSimIMRPhenomX_internals.c.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 4+)
            Columns 0-3: m1, m2, chi1z, chi2z  (masses in solar masses).

        Returns
        -------
        derived : torch.Tensor, shape (B, 13)
            ┌─────┬───────────────────────────────────────────────────────┐
            │  0  │ m1_s   = m1 * GM               (seconds)              │
            │  1  │ m2_s   = m2 * GM                                      │
            │  2  │ M_s    = (m1+m2) * GM                                 │
            │  3  │ eta    = m1*m2 / M²            (symmetric mass ratio) │
            │  4  │ delta  = sqrt(1 - 4*eta)       (PN asymmetry)         │
            │  5  │ Xa     = m1/M  = 0.5*(1+delta) (dimensionless)        │
            │  6  │ Xb     = m2/M  = 0.5*(1-delta)                        │
            │  7  │ chiEff = Xa*chi1 + Xb*chi2                            │
            │  8  │ chiPNHat (= S in all XAS fits)                        │
            │  9  │ STotR  = (Xa²*chi1 + Xb²*chi2)/(Xa²+Xb²)            │
            │ 10  │ dchi   = chi1 - chi2                                  │
            │ 11  │ chi1   = chi1z (raw aligned spin, body 1)             │
            │ 12  │ chi2   = chi2z (raw aligned spin, body 2)             │
            └─────┴───────────────────────────────────────────────────────┘
        """
        m1   = theta[:, 0:1]
        m2   = theta[:, 1:2]
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]

        m1_s = m1 * self.GM
        m2_s = m2 * self.GM
        M_s  = m1_s + m2_s

        # Symmetric mass ratio — nudge below 0.25 to avoid edge-case NaNs
        eta = m1_s * m2_s / (M_s * M_s)
        nudge_backward_(eta, 0.25, 1e-6)

        # PN asymmetry parameter and dimensionless mass fractions
        # delta = sqrt(1 - 4*eta);  Xa = 0.5*(1+delta);  Xb = 0.5*(1-delta)
        delta = torch.sqrt(1.0 - 4.0 * eta)
        Xa    = 0.5 * (1.0 + delta)   # = m1/M
        Xb    = 0.5 * (1.0 - delta)   # = m2/M

        # Spin combinations
        chiEff   = IMRPhenomXAS._chiEff(Xa, Xb, chi1, chi2)
        chiPNHat = IMRPhenomXAS._chiPNHat(eta, chiEff, chi1, chi2)
        STotR    = IMRPhenomXAS._STotR(Xa, Xb, chi1, chi2)
        dchi     = chi1 - chi2

        return torch.cat(
            [m1_s, m2_s, M_s, eta, delta, Xa, Xb,
             chiEff, chiPNHat, STotR, dchi,
             chi1, chi2],   # cols 11-12: raw aligned spins (needed by PN amplitude)
            dim=1,
        )

    # ------------------------------------------------------------------
    # Spin combination helpers
    # (static so they can be called without self on hot path)
    # ------------------------------------------------------------------

    @staticmethod
    def _chiEff(Xa, Xb, chi1, chi2):
        """
        Effective aligned spin  χ_eff = Xa·χ₁ + Xb·χ₂.

        Source: XLALSimIMRPhenomXchiEff, LALSimIMRPhenomXUtilities.c
        Convention: m1 >= m2, so Xa >= Xb.
        """
        return Xa * chi1 + Xb * chi2

    @staticmethod
    def _chiPNHat(eta, chiEff, chi1, chi2):
        """
        Hatted PN effective spin  χ̂_PN — used as the spin variable S in
        all IMRPhenomXAS phenomenological fitting functions.

        Source: XLALSimIMRPhenomXchiPNHat, LALSimIMRPhenomXUtilities.c

            χ̂_PN = (χ_eff − (38/113)·η·(χ₁+χ₂)) / (1 − 76·η/113)

        The denominator never vanishes for η ∈ (0, 0.25].
        """
        num = chiEff - (38.0 / 113.0) * eta * (chi1 + chi2)
        den = 1.0 - (76.0 / 113.0) * eta
        return num / den

    @staticmethod
    def _STotR(Xa, Xb, chi1, chi2):
        """
        Total reduced spin  S_tot/M²  normalised to [-1, 1].

        Source: XLALSimIMRPhenomXSTotR, LALSimIMRPhenomXUtilities.c

            S_totR = (Xa²·χ₁ + Xb²·χ₂) / (Xa² + Xb²)

        Used exclusively in the 2017 final-mass and final-spin fits.
        """
        Xa2 = Xa * Xa
        Xb2 = Xb * Xb
        return (Xa2 * chi1 + Xb2 * chi2) / (Xa2 + Xb2)

    # ------------------------------------------------------------------
    # Final mass and spin (Jimenez-Forteza et al. 2017, arXiv:1611.00332)
    # ------------------------------------------------------------------

    @staticmethod
    def final_mass_2017(eta, S, dchi, delta):
        """
        Remnant mass fraction  Mfinal / M  =  1 − E_rad  from the 2017 fit.

        Source: XLALSimIMRPhenomXFinalMass2017 in LALSimIMRPhenomXUtilities.c
        Reference: Jimenez-Forteza et al. (2017), arXiv:1611.00332

        Parameters
        ----------
        eta   : (B, 1) — symmetric mass ratio
        S     : (B, 1) — STotR  = (Xa²·χ₁ + Xb²·χ₂)/(Xa²+Xb²)
        dchi  : (B, 1) — χ₁ − χ₂
        delta : (B, 1) — sqrt(1 − 4η)

        Returns
        -------
        Mfinal : (B, 1) — dimensionless remnant mass fraction (< 1)
        """
        eta2  = eta  * eta
        eta3  = eta2 * eta
        eta4  = eta3 * eta
        S2    = S    * S
        S3    = S2   * S
        dchi2 = dchi * dchi

        # No-spin contribution (E_rad for equal-mass non-spinning)
        noSpin = (
            0.057190958417936644*eta
            + 0.5609904135313374*eta2
            - 0.84667563764404*eta3
            + 3.145145224278187*eta4
        )

        # Equal-spin correction  (eqSpin = [rational fit] − noSpin)
        # The rational fit is written as noSpin * (spin-modulation) / denom,
        # then noSpin is subtracted so the zero-spin limit is exactly 0.
        eqSpin = (
            noSpin
            * (1.0
               + (-0.13084389181783257 - 1.1387311580238488*eta + 5.49074464410971*eta2)*S
               + (-0.17762802148331427 + 2.176667900182948*eta2)*S2
               + (-0.6320191645391563 + 4.952698546796005*eta - 10.023747993978121*eta2)*S3)
            / (1.0 + (-0.9919475346968611 + 0.367620218664352*eta + 4.274567337924067*eta2)*S)
            - noSpin
        )

        # Unequal-spin correction
        uneqSpin = (
            - 0.09803730445895877  * dchi * delta * (1.0 - 3.2283713377939134*eta)  * eta2
            + 0.01118530335431078  * dchi2 * eta3
            - 0.01978238971523653  * dchi * delta * (1.0 - 4.91667749015812*eta) * eta * S
        )

        # Mfinal = 1 − E_rad
        return 1.0 - (noSpin + eqSpin + uneqSpin)

    @staticmethod
    def final_spin_2017(eta, S, dchi, delta):
        """
        Remnant dimensionless spin  a_f  from the 2017 fit.

        Source: XLALSimIMRPhenomXFinalSpin2017 in LALSimIMRPhenomXUtilities.c
        Reference: Jimenez-Forteza et al. (2017), arXiv:1611.00332

        Parameters
        ----------
        eta   : (B, 1) — symmetric mass ratio
        S     : (B, 1) — STotR  = (Xa²·χ₁ + Xb²·χ₂)/(Xa²+Xb²)
        dchi  : (B, 1) — χ₁ − χ₂
        delta : (B, 1) — sqrt(1 − 4η)

        Returns
        -------
        afinal : (B, 1) — dimensionless remnant spin ∈ (−1, 1)

        Notes
        -----
        Uses the identity  Xa² + Xb² = 1 − 2η  to avoid recomputing
        individual mass fractions (verifiable from Xa = (1+δ)/2).
        """
        eta2  = eta  * eta
        eta3  = eta2 * eta
        S2    = S    * S
        S3    = S2   * S
        dchi2 = dchi * dchi

        # No-spin orbital angular momentum contribution
        noSpin = (
            3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta3
        ) / (1.0 + 7.2388440419467335*eta)

        # Equal-spin correction
        # Leading term is (Xa² + Xb²)*S = (1 − 2η)*S  (identity from mass fractions)
        eqSpin = (1.0 - 2.0*eta) * S + (
            ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta3)*S
             + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta3)*S2
             + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta3)*S3)
            / (1.0 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta3)*S)
        )

        # Unequal-spin correction
        uneqSpin = (
            0.3223660562764661   * dchi * delta * (1.0 + 9.332575956437443*eta)  * eta2
            - 0.059808322561702126 * dchi2 * eta3
            + 2.3170397514509933  * dchi * delta * (1.0 - 3.2624649875884852*eta) * eta3 * S
        )

        return noSpin + eqSpin + uneqSpin

    # ------------------------------------------------------------------
    # QNM ringdown / damping frequencies
    # ------------------------------------------------------------------

    @staticmethod
    def get_fRD_fdamp(af, Mfinal):
        """
        Return (fRING, fDAMP) in dimensionless Mf units  (f_Hz = fRING / M_s).

        Uses the rational polynomial fits from LALSimIMRPhenomX_qnm.c
        (``evaluate_QNMfit_fring22`` / ``evaluate_QNMfit_fdamp22``), which is
        what LAL selects at compile-time with ``QNMfits == 1``.  The fits are
        from Berti, Cardoso & Will (2009), CQG 26, 163001; arXiv:0905.2975.

        The normalization follows LAL convention:
            fRING = evaluate_QNMfit_fring22(afinal) / Mfinal
        so that  f_physical_Hz = fRING / M_s_total.

        Parameters
        ----------
        af     : torch.Tensor, shape (B, 1)  — final dimensionless spin ∈ [−1, 1]
        Mfinal : torch.Tensor, shape (B, 1)  — remnant mass fraction from final_mass_2017

        Returns
        -------
        fRING, fDAMP : torch.Tensor, shape (B, 1)
            Dimensionless frequencies; multiply by ``1 / M_s`` to get Hz.
        """
        a  = af
        a2 = a  * a
        a3 = a2 * a
        a4 = a2 * a2
        a5 = a3 * a2
        a6 = a3 * a3
        a7 = a4 * a3

        # --- ringdown frequency (22-mode) ---
        # evaluate_QNMfit_fring22  in LALSimIMRPhenomX_qnm.c
        fring22 = (
            0.05947169566573468
            - 0.14989771215394762*a  + 0.09535606290986028*a2
            + 0.02260924869042963*a3 - 0.02501704155363241*a4
            - 0.005852438240997211*a5 + 0.0027489038393367993*a6
            + 0.0005821983163192694*a7
        ) / (
            1.0 - 2.8570126619966296*a
            + 2.373335413978394*a2   - 0.6036964688511505*a4
            + 0.0873798215084077*a6
        )

        # --- damping frequency (22-mode) ---
        # evaluate_QNMfit_fdamp22  in LALSimIMRPhenomX_qnm.c
        fdamp22 = (
            0.014158792290965177
            - 0.036989395871554566*a  + 0.026822526296575368*a2
            + 0.0008490933750566702*a3 - 0.004843996907020524*a4
            - 0.00014745235759327472*a5 + 0.0001504546201236794*a6
        ) / (
            1.0 - 2.5900842798681376*a
            + 1.8952576220623967*a2   - 0.31416610693042507*a4
            + 0.009002719412204133*a6
        )

        # Divide by Mfinal to convert from remnant-mass units to initial-mass units
        return fring22 / Mfinal, fdamp22 / Mfinal

    # ------------------------------------------------------------------
    # Special frequencies: MECO and ISCO (region boundaries)
    # ------------------------------------------------------------------

    @staticmethod
    def fMECO(eta, chiPNHat, dchi, delta):
        """
        Hybrid minimum energy circular orbit (MECO) frequency  (dimensionless Mf).

        Source: XLALSimIMRPhenomXfMECO in LALSimIMRPhenomXUtilities.c
        Reference: Cabero et al., Phys.Rev.D95 (2017) 064016.

        Parameters
        ----------
        eta      : (B, 1) — symmetric mass ratio
        chiPNHat : (B, 1) — hatted PN effective spin (used as S in XAS fits)
        dchi     : (B, 1) — χ₁ − χ₂
        delta    : (B, 1) — sqrt(1 − 4η)

        Returns
        -------
        fMECO : (B, 1) — dimensionless frequency Mf at MECO
        """
        eta2  = eta  * eta
        eta3  = eta2 * eta
        eta4  = eta3 * eta

        S  = chiPNHat          # XAS convention: S = chiPNHat for MECO fit
        S2 = S * S
        S3 = S2 * S

        dchi2 = dchi * dchi

        noSpin = (
            0.018744340279608845 + 0.0077903147004616865*eta
            + 0.003940354686136861*eta2 - 0.00006693930988501673*eta3
        ) / (1.0 - 0.10423384680638834*eta)

        eqSpin = (
            S * (
                0.00027180386951683135 - 0.00002585252361022052*S
                + eta4 * (-0.0006807631931297156 + 0.022386313074011715*S - 0.0230825153005985*S2)
                + eta2 * (0.00036556167661117023 - 0.000010021140796150737*S - 0.00038216081981505285*S2)
                + eta  * (0.00024422562796266645 - 0.00001049013062611254*S - 0.00035182990586857726*S2)
                + eta3 * (-0.0005418851224505745 + 0.000030679548774047616*S + 4.038390455349854e-6*S2)
                - 0.00007547517256664526*S2
            )
        ) / (
            0.026666543809890402
            + (-0.014590539285641243 - 0.012429476486138982*eta + 1.4861197211952053*eta4
               + 0.025066696514373803*eta2 + 0.005146809717492324*eta3)*S
            + (-0.0058684526275074025 - 0.02876774751921441*eta - 2.551566872093786*eta4
               - 0.019641378027236502*eta2 - 0.001956646166089053*eta3)*S2
            + (0.003507640638496499 + 0.014176504653145768*eta + 1.0*eta4
               + 0.012622225233586283*eta2 - 0.00767768214056772*eta3)*S3
        )

        uneqSpin = (
            dchi2 * (0.00034375176678815234 + 0.000016343732281057392*eta) * eta2
            + dchi * delta * eta * (
                0.08064665214195679*eta2
                + eta * (-0.028476219509487793 - 0.005746537021035632*S)
                - 0.0011713735642446144*S
            )
        )

        return noSpin + eqSpin + uneqSpin

    @staticmethod
    def fISCO(afinal):
        """
        Innermost stable circular orbit (ISCO) frequency  (dimensionless Mf).

        Source: XLALSimIMRPhenomXfISCO in LALSimIMRPhenomXUtilities.c
        Reference: Ori & Thorne, Phys.Rev.D62 (2000) 124022.

        Returns the Kerr ISCO orbital frequency OmegaISCO / π normalised to
        total initial mass, i.e. f_ISCO_Hz = fISCO / M_s.

        Parameters
        ----------
        afinal : (B, 1) — final dimensionless spin

        Returns
        -------
        fISCO : (B, 1) — dimensionless Mf at ISCO
        """
        # Kerr ISCO radius in units of M (Eq. A2-A4, Bardeen et al. 1972)
        a  = afinal
        a2 = a * a
        Z1 = 1.0 + (1.0 - a2).pow(1.0 / 3.0) * ((1.0 + a).pow(1.0 / 3.0) + (1.0 - a).pow(1.0 / 3.0))
        Z1 = torch.clamp(Z1, max=3.0)           # guard finite-precision edge at a→0
        Z2 = torch.sqrt(3.0*a2 + Z1*Z1)
        rISCO = 3.0 + Z2 - torch.sign(a) * torch.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2))

        # OmegaISCO = 1 / (r^{3/2} + a)  in units of 1/M
        OmegaISCO = 1.0 / (rISCO.pow(1.5) + a)

        return OmegaISCO / math.pi

    # ------------------------------------------------------------------
    # Amplitude: 3 regions
    # ------------------------------------------------------------------

    def get_amp_coeffs(self, derived):
        """
        Compute all IMRPhenomXAS amplitude coefficients for the batch.

        Mirrors ``IMRPhenomXGetAmplitudeCoefficients`` in
        LALSimIMRPhenomX_internals.c.  All versions are fixed to the LAL
        defaults: InspiralAmpVersion=103, IntermediateAmpVersion=104,
        RingdownAmpVersion=103.

        Parameters
        ----------
        derived : (B, 13)  — output of ``compute_derived_parameters``

        Returns
        -------
        ac : dict
            Keys (all (B, 1) tensors unless noted):
            ``pnTT`` through ``pnST``, ``rho1``–``rho3``  — inspiral
            ``delta0``–``delta4``                         — intermediate
            ``gamma1``, ``gammaR``, ``gammaD2``, ``gammaD13``, ``fRING``
                                                          — ringdown
            ``fAmpMatchIN``, ``fAmpRDMin``               — boundaries
        """
        # ------------------------------------------------------------------
        # Unpack derived parameters
        # ------------------------------------------------------------------
        eta      = derived[:, 3:4]
        delta    = derived[:, 4:5]
        chi1     = derived[:, 11:12]   # chi1z (body 1 aligned spin)
        chi2     = derived[:, 12:13]   # chi2z (body 2 aligned spin)
        STotR    = derived[:, 9:10]    # S used in ringdown / intermediate fits
        chiPNHat = derived[:, 8:9]     # S used in inspiral amplitude fits
        dchi     = derived[:, 10:11]

        eta2  = eta  * eta
        eta3  = eta2 * eta
        chi12 = chi1 * chi1
        chi13 = chi12 * chi1
        chi22 = chi2 * chi2

        pi    = math.pi

        # ------------------------------------------------------------------
        # 1. Final mass and spin → fRING, fDAMP  (dimensionless Mf units)
        # ------------------------------------------------------------------
        Mfinal = IMRPhenomXAS.final_mass_2017(eta, STotR, dchi, delta)
        afinal = IMRPhenomXAS.final_spin_2017(eta, STotR, dchi, delta)
        fRING, fDAMP = IMRPhenomXAS.get_fRD_fdamp(afinal, Mfinal)

        # ------------------------------------------------------------------
        # 2. MECO and ISCO frequencies  (dimensionless Mf units)
        # ------------------------------------------------------------------
        fmeco = IMRPhenomXAS.fMECO(eta, chiPNHat, dchi, delta)
        fisco = IMRPhenomXAS.fISCO(afinal)

        # ------------------------------------------------------------------
        # 3. Amplitude region boundaries
        # fAmpMatchIN = fMECO + 0.25*(fISCO - fMECO)   (Eq. 5.16)
        # fAmpRDMin   = ringdown peak frequency          (Eq. 5.14)
        # ------------------------------------------------------------------
        # --- ringdown phenomenological coefficients (STotR spin) ----------
        S  = STotR;  S2 = S*S

        # gamma2  (λ in arXiv:2001.11412)
        gamma2 = (
            (0.8312293675316895 + 7.480371544268765*eta - 18.256121237800397*eta2)
            / (1.0 + 10.915453595496611*eta - 30.578409433912874*eta2)
            + (S*(0.5869408584532747 + eta*(-0.1467158405070222 - 2.8489481072076472*S)
               + 0.031852563636196894*S + eta2*(0.25295441250444334 + 4.6849496672664594*S)))
            / (3.8775263105069953 - 3.41755361841226*S + S2)
            - 0.00548054788508203*dchi*delta*eta
        )

        # gamma3  (σ in arXiv:2001.11412)
        gamma3 = (
            (1.3666000000000007 - 4.091333144596439*eta + 2.109081209912545*eta2 - 4.222259944408823*eta3)
            / (1.0 - 2.7440263888207594*eta)
            + (0.07179105336478316 + eta2*(2.331724812782498 - 0.6330998412809531*S)
               + eta*(-0.8752427297525086 + 0.4168560229353532*S) - 0.05633734476062242*S)*S
        )

        # Ringdown peak frequency: fAmpRDMin  (Eq. 5.14)
        # |fRING + fDAMP*gamma3*(sqrt(1 - gamma2^2) - 1)/gamma2|
        # If gamma2 > 1, use |fRING - fDAMP*gamma3/gamma2| instead
        sqrt_arg = (1.0 - gamma2*gamma2).clamp(min=0.0)
        fAmpRDMin_normal = torch.abs(fRING + fDAMP * gamma3 * (sqrt_arg.sqrt() - 1.0) / gamma2)
        fAmpRDMin_large  = torch.abs(fRING - fDAMP * gamma3 / gamma2)
        fAmpRDMin = torch.where(gamma2 <= 1.0, fAmpRDMin_normal, fAmpRDMin_large)

        # v1RD: amplitude collocation at fAmpRDMin  (STotR spin)
        S3 = S2*S;  S4 = S3*S
        v1RD = (
            (0.03689164742964719 + 25.417967754401182*eta + 162.52904393600332*eta2)
            / (1.0 + 61.19874463331437*eta - 29.628854485544874*eta2)
            + (S*(-0.14352506969368556 + 0.026356911108320547*S + 0.19967405175523437*S2
               - 0.05292913111731128*S3
               + eta3*(-48.31945248941757 - 3.751501972663298*S + 81.9290740950083*S2
                       + 30.491948143930266*S3 - 132.77982622925845*S4)
               + eta*(-4.805034453745424 + 1.11147906765112*S + 6.176053843938542*S2
                      - 0.2874540719094058*S3 - 8.990840289951514*S4)
               - 0.18147275151697131*S4
               + eta2*(27.675454081988036 - 2.398327419614959*S - 47.99096500250743*S2
                       - 5.104257870393138*S3 + 72.08174136362386*S4)))
            / (-1.4160870461211452 + S)
            - 0.04426571511345366*dchi*delta*eta2
        )

        # gamma1 (solved analytically from ansatzRD(fAmpRDMin) == v1RD)
        # gammaR = gamma2 / (fDAMP * gamma3);  gammaD2 = (fDAMP*gamma3)^2
        # gammaD13 = fDAMP * gamma1 * gamma3
        # gamma1 = v1RD * (fAmpRDMin - fRING)^2 + gammaD2) / (fDAMP*gamma3)
        #           * exp((fAmpRDMin - fRING)*gamma2 / (fDAMP*gamma3))
        fDMg3  = fDAMP * gamma3                   # fDAMP * gamma3
        gammaR  = gamma2 / fDMg3
        gammaD2 = fDMg3 * fDMg3
        dfr_rd  = fAmpRDMin - fRING
        gamma1  = (v1RD / fDMg3) * (dfr_rd*dfr_rd + gammaD2) * torch.exp(dfr_rd * gammaR)
        gammaD13 = fDMg3 * gamma1

        # Inspiral boundary frequency
        fAmpMatchIN = fmeco + 0.25 * (fisco - fmeco)

        # ------------------------------------------------------------------
        # 4. TaylorF2 PN amplitude coefficients
        #    Source: IMRPhenomXGetAmplitudeCoefficients in LAL internals.c
        #    (Section V.A of arXiv:2001.11412)
        # ------------------------------------------------------------------
        p2o3 = pi**(2.0/3.0)
        p1o3 = pi**(1.0/3.0)
        p4o3 = pi**(4.0/3.0)
        p5o3 = pi**(5.0/3.0)
        p2   = pi * pi

        pnTwoThirds  = ((-969.0 + 1804.0*eta) / 672.0) * p2o3

        pnThreeThirds = (
            (81.0*(chi1 + chi2) + 81.0*chi1*delta - 81.0*chi2*delta
             - 44.0*(chi1 + chi2)*eta) / 48.0
        ) * pi

        pnFourThirds  = (
            (-27312085.0 - 10287648.0*chi12*(1.0 + delta)
             + 24.0*(428652.0*chi22*(-1.0 + delta)
                     + (-1975055.0 + 10584.0*(81.0*chi12 - 94.0*chi1*chi2 + 81.0*chi22))*eta
                     + 1473794.0*eta2))
            / 8.128512e6
        ) * p1o3 * pi

        pnFiveThirds  = (
            (-6048.0*chi13*(-1.0 - delta + (3.0 + delta)*eta)
             + chi2*(-(287213.0 + 6048.0*chi22)*(-1.0 + delta)
                     + 4.0*(-93414.0 + 1512.0*chi22*(-3.0 + delta) + 2083.0*delta)*eta
                     - 35632.0*eta2)
             + chi1*(287213.0*(1.0 + delta) - 4.0*eta*(93414.0 + 2083.0*delta + 8908.0*eta))
             + 42840.0*(-1.0 + 4.0*eta)*pi)
            / 32256.0
        ) * p5o3

        pnSixThirds   = (
            (-1242641879927.0
             + 12.0*(28.0*(-3248849057.0
                           + 11088.0*(163199.0*chi12 - 266498.0*chi1*chi2 + 163199.0*chi22))*eta2
                    + 27026893936.0*eta3
                    - 116424.0*(147117.0*(-(chi22*(-1.0 + delta)) + chi12*(1.0 + delta))
                                + 60928.0*(chi1 + chi2 + chi1*delta - chi2*delta)*pi)
                    + eta*(545384828789.0
                           - 77616.0*(638642.0*chi1*chi2
                                      + chi12*(-158633.0 + 282718.0*delta)
                                      - chi22*(158633.0 + 282718.0*delta)
                                      - 107520.0*(chi1 + chi2)*pi
                                      + 275520.0*p2))))
            / 6.0085960704e10
        ) * p2

        # ------------------------------------------------------------------
        # 5. Inspiral pseudo-PN collocation points (version 103, chiPNHat spin)
        #    F1 = 0.50 * fAmpMatchIN   F2 = 0.75   F3 = 1.00
        # ------------------------------------------------------------------
        S_ins = chiPNHat;  S2_ins = S_ins*S_ins;  S3_ins = S2_ins*S_ins
        eta4  = eta3 * eta

        V2 = (   # v2 at 0.5 * fAmpMatchIN
            (-0.015178276424448592 - 0.06098548699809163*eta + 0.4845148547154606*eta2)
            / (1.0 + 0.09799277215675059*eta)
            + ((0.02300153747158323 + 0.10495263104245876*eta2)*S_ins
               + (0.04834642258922544 - 0.14189350657140673*eta)*eta*S3_ins
               + (0.01761591799745109 - 0.14404522791467844*eta2)*S2_ins)
            / (1.0 - 0.7340448493183307*S_ins)
            + dchi*delta*eta4*(0.0018724905795891192 + 34.90874132485147*eta)
        )

        V3 = (   # v3 at 0.75 * fAmpMatchIN
            (-0.058572000924124644 - 1.1970535595488723*eta + 8.4630293045015*eta2)
            / (1.0 + 15.430818840453686*eta)
            + ((-0.08746408292050666 + eta*(-0.20646621646484237 - 0.21291764491897636*S_ins)
                + eta2*(0.788717372588848 + 0.8282888482429105*S_ins)
                - 0.018924013869130434*S_ins)*S_ins)
            / (-1.332123330797879 + S_ins)
            + dchi*delta*eta4*(0.004389995099201855 + 105.84553997647659*eta)
        )

        V4 = (   # v4 at 1.00 * fAmpMatchIN
            (-0.16212854591357853 + 1.617404703616985*eta - 3.186012733446088*eta2 + 5.629598195000046*eta3)
            / (1.0 + 0.04507019231274476*eta)
            + (S_ins*(1.0055835408962206
               + eta2*(18.353433894421833 - 18.80590889704093*S_ins)
               - 0.31443470118113853*S_ins
               + eta*(-4.127597118865669 + 5.215501942120774*S_ins)
               + eta3*(-41.0378120175805 + 19.099315016873643*S_ins)))
            / (5.852706459485663 - 5.717874483424523*S_ins + S2_ins)
            + dchi*delta*eta4*(0.05575955418803233 + 208.92352600701068*eta)
        )

        # Collocation frequencies (Mf)
        F1 = 0.50 * fAmpMatchIN
        F2 = 0.75 * fAmpMatchIN
        F3 = 1.00 * fAmpMatchIN          # = fAmpMatchIN itself

        # Solve for rho1, rho2, rho3 (pseudo-PN amplitude correction coefficients)
        # Source: IMRPhenomX_Inspiral_Amp_22_rho1/2/3 in LAL inspiral.c (case 103)
        F1p1o3 = F1.pow(1.0/3.0);  F2p1o3 = F2.pow(1.0/3.0);  F3p1o3 = F3.pow(1.0/3.0)
        F1p7o3 = F1p1o3.pow(7);    F2p7o3 = F2p1o3.pow(7);    F3p7o3 = F3p1o3.pow(7)
        F1p8o3 = F1p7o3 * F1p1o3;  F2p8o3 = F2p7o3 * F2p1o3;  F3p8o3 = F3p7o3 * F3p1o3
        F13    = F1*F1*F1;          F23    = F2*F2*F2;          F33    = F3*F3*F3

        D = F1p7o3*(F1p1o3 - F2p1o3)*F2p7o3*(F1p1o3 - F3p1o3)*(F2p1o3 - F3p1o3)*F3p7o3

        rho1 = (
            -F2p8o3*F33*V2 + F23*F3p8o3*V2 + F1p8o3*F33*V3 - F13*F3p8o3*V3
            - F1p8o3*F23*V4 + F13*F2p8o3*V4
        ) / D

        rho2 = (
            F2p7o3*F33*V2 - F23*F3p7o3*V2 - F1p7o3*F33*V3 + F13*F3p7o3*V3
            + F1p7o3*F23*V4 - F13*F2p7o3*V4
        ) / D

        rho3 = (
            F2p8o3*F3p7o3*V2 - F2p7o3*F3p8o3*V2 - F1p8o3*F3p7o3*V3 + F1p7o3*F3p8o3*V3
            + F1p8o3*F2p7o3*V4 - F1p7o3*F2p8o3*V4
        ) / D

        # ------------------------------------------------------------------
        # 6. Inspiral amplitude at F1 = fAmpMatchIN  (needed for d1 and V1)
        # pnAmp(Mf) = 1 + pnTT*Mf^{2/3} + pnThT*Mf + pnFoT*Mf^{4/3}
        #           + pnFiT*Mf^{5/3} + pnST*Mf^2
        #           + rho1*Mf^{7/3} + rho2*Mf^{8/3} + rho3*Mf^3
        # ------------------------------------------------------------------
        Fma = fAmpMatchIN                         # short alias for F1 = fAmpMatchIN
        inspF1 = (
            1.0
            + pnTwoThirds  * Fma.pow(2.0/3.0)
            + pnThreeThirds * Fma
            + pnFourThirds  * Fma.pow(4.0/3.0)
            + pnFiveThirds  * Fma.pow(5.0/3.0)
            + pnSixThirds   * Fma * Fma
            + rho1 * Fma.pow(7.0/3.0)
            + rho2 * Fma.pow(8.0/3.0)
            + rho3 * Fma * Fma * Fma
        )

        # Inspiral amplitude derivative (d pnAmp / d Mf) at Fma  (case 103)
        Fma_m1o3 = Fma.pow(-1.0/3.0)
        Fma_p1o3 = Fma.pow(1.0/3.0)
        Fma_p2o3 = Fma.pow(2.0/3.0)
        Fma_p2   = Fma * Fma

        d_inspF1 = (
            ((chi2*(81.0 - 81.0*delta - 44.0*eta) + chi1*(81.0*(1.0 + delta) - 44.0*eta))*pi) / 48.0
            + ((-969.0 + 1804.0*eta)*p2o3) / (1008.0 * Fma_m1o3)
            + ((-27312085.0 - 10287648.0*chi22 + 10287648.0*chi22*delta
               - 10287648.0*chi12*(1.0 + delta)
               + 24.0*(-1975055.0 + 857304.0*chi12 - 994896.0*chi1*chi2 + 857304.0*chi22)*eta
               + 35371056.0*eta2) * p4o3 * Fma_p1o3) / 6.096384e6
            + (5.0 * p5o3 * (-6048.0*chi13*(-1.0 - delta + (3.0 + delta)*eta)
               + chi1*(287213.0*(1.0 + delta) - 4.0*(93414.0 + 2083.0*delta)*eta - 35632.0*eta2)
               + chi2*(-(287213.0 + 6048.0*chi22)*(-1.0 + delta)
                       + 4.0*(-93414.0 + 1512.0*chi22*(-3.0 + delta) + 2083.0*delta)*eta
                       - 35632.0*eta2)
               + 42840.0*(-1.0 + 4.0*eta)*pi) * Fma_p2o3) / 96768.0
            - (p2 * (-336.0*(-3248849057.0 + 1809550512.0*chi12 - 2954929824.0*chi1*chi2
                              + 1809550512.0*chi22)*eta2
                      - 324322727232.0*eta3
                      + 7.0*(177520268561.0 + 29362199328.0*chi22
                             - 29362199328.0*chi22*delta
                             + 29362199328.0*chi12*(1.0 + delta)
                             + 12160253952.0*(chi1 + chi2 + chi1*delta - chi2*delta)*pi)
                      + 12.0*eta*(-545384828789.0 + 49568837472.0*chi1*chi2
                                  - 12312458928.0*chi22 - 21943440288.0*chi22*delta
                                  + 77616.0*chi12*(-158633.0 + 282718.0*delta)
                                  - 8345272320.0*(chi1 + chi2)*pi
                                  + 21384760320.0*p2)) * Fma) / 3.0042980352e10
            + (7.0/3.0) * Fma.pow(4.0/3.0) * rho1
            + (8.0/3.0) * Fma.pow(5.0/3.0) * rho2
            + 3.0 * Fma_p2 * rho3
        )

        # d1 = d/dMf [Mf^{7/6} / inspAmp(Mf)] at Mf = Fma
        d1 = (7.0/6.0) * Fma.pow(1.0/6.0) / inspF1 - Fma.pow(7.0/6.0) * d_inspF1 / (inspF1*inspF1)

        # ------------------------------------------------------------------
        # 7. Ringdown amplitude at F4 = fAmpRDMin  (needed for d4 and V4_rd)
        # ------------------------------------------------------------------
        F4 = fAmpRDMin
        dfr4 = F4 - fRING
        rdF4  = torch.exp(-dfr4 * gammaR) * gammaD13 / (dfr4*dfr4 + gammaD2)

        # Ringdown amplitude derivative at F4
        d_rdF4 = (
            -torch.exp(-gamma2 * dfr4 / fDMg3) * gamma1
            * (dfr4*dfr4*gamma2 + 2.0*fDAMP*dfr4*gamma3 + fDAMP*fDAMP*gamma2*gamma3*gamma3)
            / ((dfr4*dfr4 + gammaD2) * (dfr4*dfr4 + gammaD2))
        )

        # d4 = d/dMf [Mf^{7/6} / rdAmp(Mf)] at Mf = F4
        d4 = (7.0/6.0) * F4.pow(1.0/6.0) / rdF4 - F4.pow(7.0/6.0) * d_rdF4 / (rdF4*rdF4)

        # ------------------------------------------------------------------
        # 8. Intermediate amplitude collocation  (version 104, STotR spin)
        #    F1 = fAmpMatchIN,  F2 = F1 + 0.5*(F4 - F1),  F4 = fAmpRDMin
        #    V1 = F1^{7/6} / inspAmp(F1)  (rho polynomial at F1)
        #    V2 = 1 / vA(F2)              (rho polynomial at F2, from fit)
        #    V4 = F4^{7/6} / rdAmp(F4)   (rho polynomial at F4)
        # ------------------------------------------------------------------
        F2_int = F1 + 0.5*(F4 - F1)   # midpoint of intermediate region (F1 = fAmpMatchIN, F4 = fAmpRDMin)
        F1_int = fAmpMatchIN            # alias for clarity

        V1_int = F1_int.pow(7.0/6.0) / inspF1     # rho at F1
        V4_int = F4.pow(7.0/6.0) / rdF4           # rho at F4

        # vA: intermediate amplitude at F2 from the fit (version 104, STotR spin)
        S3   = S2*S  # reuse S = STotR from above
        vA = (
            (1.4873184918202145 + 1974.6112656679577*eta + 27563.641024162127*eta2
             - 19837.908020966777*eta3)
            / (1.0 + 143.29004876335128*eta + 458.4097306093354*eta2)
            + (S*(27.952730865904343 + eta*(-365.55631765202895 - 260.3494489873286*S)
               + 3.2646808851249016*S + 3011.446602208493*eta2*S
               - 19.38970173389662*S2 + eta3*(1612.2681322644232 - 6962.675551371755*S
                                               + 1486.4658089990298*S2)))
            / (12.647425554323242 - 10.540154508599963*S + S2)
            + dchi*delta*(-0.016404056649860943 - 296.473359655246*eta)*eta2
        )
        V2_int = 1.0 / vA      # rho at F2 (inverse of amplitude-at-F2)

        # ------------------------------------------------------------------
        # 9. Solve for delta coefficients (4th-order polynomial, case 104)
        #    Source: IMRPhenomX_Intermediate_Amp_22_delta0..4 in intermediate.c
        #    F1, F2, F4 → f1, f2, f4  (f3=0, v3=0 not used in case 104)
        # ------------------------------------------------------------------
        f1 = F1_int;  f2 = F2_int;  f4 = F4
        v1 = V1_int;  v2 = V2_int;  v4 = V4_int

        f12 = f1*f1;  f13 = f12*f1;  f14 = f13*f1;  f15 = f14*f1
        f22 = f2*f2;  f23 = f22*f2;  f24 = f23*f2
        f42 = f4*f4;  f43 = f42*f4;  f44 = f43*f4;  f45 = f44*f4

        f1mf2 = f1 - f2;  f1mf4 = f1 - f4;  f2mf4 = f2 - f4
        f1mf22 = f1mf2*f1mf2
        f2mf42 = f2mf4*f2mf4
        f1mf43 = f1mf4*f1mf4*f1mf4

        delta0 = (
            (-(d4*f12*f1mf22*f1mf4*f2*f2mf4*f4)
             + d1*f1*f1mf2*f1mf4*f2*f2mf42*f42
             + f42*(f2*f2mf42*(-4.0*f12 + 3.0*f1*f2 + 2.0*f1*f4 - f2*f4)*v1
                    + f12*f1mf43*v2)
             + f12*f1mf22*f2*(f1*f2 - 2.0*f1*f4 - 3.0*f2*f4 + 4.0*f42)*v4)
            / (f1mf22*f1mf43*f2mf42)
        )

        delta1 = (
            (d4*f1*f1mf22*f1mf4*f2mf4*(2.0*f2*f4 + f1*(f2 + f4))
             + f4*(-(d1*f1mf2*f1mf4*f2mf42*(2.0*f1*f2 + (f1 + f2)*f4))
                  - 2.0*f1*(f44*(v1 - v2) + 3.0*f24*(v1 - v4) + f14*(v2 - v4)
                             + 4.0*f23*f4*(-v1 + v4)
                             + 2.0*f13*f4*(-v2 + v4)
                             + f1*(2.0*f43*(-v1 + v2) + 6.0*f22*f4*(v1 - v4)
                                   + 4.0*f23*(-v1 + v4)))))
            / (f1mf22*f1mf43*f2mf42)
        )

        # delta2: source: IMRPhenomX_Intermediate_Amp_22_delta2 case 104
        # f15 = f1^5, f45 = f4^5 are both needed here
        delta2 = (
            (-(d4*f1mf22*f1mf4*f2mf4*(f12 + f2*f4 + 2.0*f1*(f2 + f4)))
             + d1*f1mf2*f1mf4*f2mf42*(f1*f2 + 2.0*(f1 + f2)*f4 + f42)
             - 4.0*f12*f23*v1 + 3.0*f1*f24*v1 - 4.0*f1*f23*f4*v1 + 3.0*f24*f4*v1
             + 12.0*f12*f2*f42*v1 - 4.0*f23*f42*v1 - 8.0*f12*f43*v1
             + f1*f44*v1 + f45*v1
             + f15*v2 + f14*f4*v2 - 8.0*f13*f42*v2 + 8.0*f12*f43*v2
             - f1*f44*v2 - f45*v2
             - f1mf22*(f13 + f2*(3.0*f2 - 4.0*f4)*f4 + f12*(2.0*f2 + f4)
                       + f1*(3.0*f2 - 4.0*f4)*(f2 + 2.0*f4))*v4)
            / (f1mf22*f1mf43*f2mf42)
        )

        delta3 = (
            (d4*f1mf22*f1mf4*f2mf4*(2.0*f1 + f2 + f4)
             - d1*f1mf2*f1mf4*f2mf42*(f1 + f2 + 2.0*f4)
             + 2.0*(f44*(-v1 + v2) + 2.0*f12*f2mf42*(v1 - v4)
                    + 2.0*f22*f42*(v1 - v4)
                    + 2.0*f13*f4*(v2 - v4) + f24*(-v1 + v4) + f14*(-v2 + v4)
                    + 2.0*f1*f4*(f42*(v1 - v2) + f22*(v1 - v4) + 2.0*f2*f4*(-v1 + v4))))
            / (f1mf22*f1mf43*f2mf42)
        )

        delta4 = (
            (-(d4*f1mf22*f1mf4*f2mf4) + d1*f1mf2*f1mf4*f2mf42
             - 3.0*f1*f22*v1 + 2.0*f23*v1 + 6.0*f1*f2*f4*v1 - 3.0*f22*f4*v1
             - 3.0*f1*f42*v1 + f43*v1 + f13*v2 - 3.0*f12*f4*v2 + 3.0*f1*f42*v2 - f43*v2
             - f1mf22*(f1 + 2.0*f2 - 3.0*f4)*v4)
            / (f1mf22*f1mf43*f2mf42)
        )

        return {
            # Inspiral
            'pnTwoThirds':   pnTwoThirds,
            'pnThreeThirds': pnThreeThirds,
            'pnFourThirds':  pnFourThirds,
            'pnFiveThirds':  pnFiveThirds,
            'pnSixThirds':   pnSixThirds,
            'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
            'fAmpMatchIN': fAmpMatchIN,
            # Intermediate
            'delta0': delta0, 'delta1': delta1, 'delta2': delta2,
            'delta3': delta3, 'delta4': delta4,
            'fAmpRDMin': fAmpRDMin,
            # Ringdown (pre-cached combinations)
            'fRING':   fRING,
            'gammaR':  gammaR,   # gamma2 / (fDAMP * gamma3)
            'gammaD2': gammaD2,  # (fDAMP * gamma3)^2
            'gammaD13': gammaD13, # fDAMP * gamma1 * gamma3
        }

    def amp(self, f_Ms, amp_coeffs, derived):
        """
        Evaluate the IMRPhenomXAS normalised amplitude A(Mf) over the grid.

        The full FD waveform amplitude is:
            |h(f)| = Amp0 * Mf^{-7/6} * A(Mf)
        where  Amp0 = sqrt(2η/3) * π^{-1/6}  and the physical prefactor
        M_s² / dist_s is applied in get_hphc.

        Region definitions (boundary frequencies in Mf units):
          Inspiral   :  Mf ≤ fAmpMatchIN
          Intermediate:  fAmpMatchIN < Mf ≤ fAmpRDMin
          Ringdown   :  Mf > fAmpRDMin

        Inspiral ansatz (Eq. 5.2 of arXiv:2001.11412):
            A_ins(Mf) = 1 + pnTT·Mf^{2/3} + pnThT·Mf + pnFoT·Mf^{4/3}
                        + pnFiT·Mf^{5/3} + pnST·Mf²
                        + ρ₁·Mf^{7/3} + ρ₂·Mf^{8/3} + ρ₃·Mf³

        Intermediate ansatz (Eq. 6.12):
            A_int(Mf) = Mf^{7/6} / (δ₀ + δ₁·Mf + δ₂·Mf² + δ₃·Mf³ + δ₄·Mf⁴)

        Ringdown ansatz (Eq. 6.17):
            A_rd(Mf) = γ₁·fDAMP·γ₃ · exp(-(Mf-fRING)·γ₂/(fDAMP·γ₃))
                       / ((Mf-fRING)² + (fDAMP·γ₃)²)

        Parameters
        ----------
        f_Ms      : (B, F) — dimensionless frequency grid Mf = f·M_s
        amp_coeffs: dict   — output of get_amp_coeffs
        derived   : (B, 13) — (unused here; kept for API symmetry)

        Returns
        -------
        amp : (B, F) — normalised amplitude A(Mf)
        """
        ac = amp_coeffs

        # ---- Inspiral ----
        Mf = f_Ms
        A_ins = (
            1.0
            + ac['pnTwoThirds']  * Mf.pow(2.0/3.0)
            + ac['pnThreeThirds'] * Mf
            + ac['pnFourThirds']  * Mf.pow(4.0/3.0)
            + ac['pnFiveThirds']  * Mf.pow(5.0/3.0)
            + ac['pnSixThirds']   * Mf * Mf
            + ac['rho1'] * Mf.pow(7.0/3.0)
            + ac['rho2'] * Mf.pow(8.0/3.0)
            + ac['rho3'] * Mf * Mf * Mf
        )

        # ---- Intermediate  (polynomial in 1/A, then inverted back) ----
        rho_int = (
            ac['delta0']
            + ac['delta1'] * Mf
            + ac['delta2'] * Mf * Mf
            + ac['delta3'] * Mf * Mf * Mf
            + ac['delta4'] * Mf * Mf * Mf * Mf
        )
        A_int = Mf.pow(7.0/6.0) / rho_int

        # ---- Ringdown  (Lorentzian) ----
        dfr   = Mf - ac['fRING']
        A_rd  = (
            torch.exp(-dfr * ac['gammaR'])
            * ac['gammaD13']
            / (dfr*dfr + ac['gammaD2'])
        )

        # ---- Stitch regions with torch.where ----
        fIN = ac['fAmpMatchIN']    # (B, 1) — inspiral/intermediate boundary
        fRD = ac['fAmpRDMin']      # (B, 1) — intermediate/ringdown boundary
        return torch.where(Mf <= fIN, A_ins,
               torch.where(Mf <= fRD, A_int, A_rd))

    # ------------------------------------------------------------------
    # Phase: 3 regions
    # ------------------------------------------------------------------

    def get_phase_coeffs(self, derived):
        """
        Compute all IMRPhenomXAS phase coefficients for the batch.
        """
        raise NotImplementedError

    def phase(self, f_Ms, phase_coeffs, derived):
        """
        Evaluate the full IMRPhenomXAS phase over the frequency grid.

        Stitches inspiral / intermediate / ringdown with C¹ continuity.
        """
        raise NotImplementedError

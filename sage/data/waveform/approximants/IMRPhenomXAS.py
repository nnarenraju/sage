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
        derived : torch.Tensor, shape (B, 11)
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
             chiEff, chiPNHat, STotR, dchi],
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

        Returns a dict or NamedTuple of tensors, one per coefficient.
        """
        raise NotImplementedError

    def amp(self, f_Ms, amp_coeffs, derived):
        """
        Evaluate the full IMRPhenomXAS amplitude over the frequency grid.

        Stitches inspiral / intermediate / ringdown with torch.where.
        """
        raise NotImplementedError

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

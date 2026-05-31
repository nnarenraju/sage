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
from sage.data.waveform import taper as _taper_mod
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

    # ------------------------------------------------------------------
    # Time-shift and psi4-to-strain fits
    # Source: XLALSimIMRPhenomXLinb / XLALSimIMRPhenomXPsi4ToStrain
    #         in LALSimIMRPhenomXUtilities.c
    # ------------------------------------------------------------------

    @staticmethod
    def _linb_fit(eta, STotR, dchi, delta):
        """
        Empirical time-alignment shift fit.

        Mirrors XLALSimIMRPhenomXLinb (LALSimIMRPhenomXUtilities.c).
        Returns ``linb`` in units of 1/Mf (the linear phase slope that
        aligns the model's peak to the hybrid waveform peak).

        Parameters
        ----------
        eta, STotR, dchi, delta : torch.Tensor, shape (B, 1)

        Returns
        -------
        linb : torch.Tensor, shape (B, 1)
        """
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = STotR * STotR
        S3 = S2 * STotR
        S4 = S3 * STotR

        noSpin = (
            3155.1635543201924
            + 1257.9949740608242   * eta
            - 32243.28428870599   * eta2
            + 347213.65466875216  * eta3
            - 1.9223851649491738e6 * eta4
            + 5.3035911346921865e6 * eta5
            - 5.789128656876938e6 * eta6
        )
        eqSpin = (
            (-24.181508118588667 + 115.49264174560281*eta - 380.19778216022763*eta2) * STotR
            + (24.72585609641552 - 328.3762360751952*eta + 725.6024119989094*eta2)   * S2
            + (23.404604124552   - 646.3410199799737*eta + 1941.8836639529036*eta2)  * S3
            + (-12.814828278938885 - 325.92980012408367*eta + 1320.102640190539*eta2) * S4
        )
        uneqSpin = -148.17317525117338 * dchi * delta * eta2

        return noSpin + eqSpin + uneqSpin

    @staticmethod
    def _psi4tostrain_fit(eta, STotR, dchi):
        """
        Psi4-to-strain time correction fit.

        Mirrors XLALSimIMRPhenomXPsi4ToStrain (LALSimIMRPhenomXUtilities.c).
        Returns the number of M cycles (psi4tostrain) by which the psi4 peak
        precedes the strain peak.

        Parameters
        ----------
        eta, STotR, dchi : torch.Tensor, shape (B, 1)

        Returns
        -------
        psi4tostrain : torch.Tensor, shape (B, 1)
        """
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = STotR * STotR
        S3 = S2 * STotR
        S4 = S3 * STotR

        noSpin = (
            13.39320482758057
            - 175.42481512989315 * eta
            + 2097.425116152503  * eta2
            - 9862.84178637907   * eta3
            + 16026.897939722587 * eta4
        )
        eqSpin = (
            (4.7895602776763 - 163.04871764530466*eta + 609.5575850476959*eta2) * STotR
            + (1.3934428041390161 - 97.51812681228478*eta + 376.9200932531847*eta2) * S2
            + (15.649521097877374 + 137.33317057388916*eta - 755.9566456906406*eta2) * S3
            + (13.097315867845788 + 149.30405703643288*eta - 764.5242164872267*eta2) * S4
        )
        uneqSpin = 105.37711654943146 * dchi * torch.sqrt(
            torch.clamp(1.0 - 4.0*eta, min=0.0)
        ) * eta2

        return noSpin + eqSpin + uneqSpin

    # ------------------------------------------------------------------
    # Phase derivative (three-region, needed for linb computation)
    # ------------------------------------------------------------------

    def dphase(self, Mf, phase_coeffs, derived=None):  # noqa: ARG002 – derived unused, kept for API symmetry
        """
        Evaluate the IMRPhenomXAS phase *derivative* dΨ₂₂/d(Mf) over the grid.

        This is used by ``get_hphc`` to compute the time-alignment shift
        (``linb``) at ``Mf = fRING - fDAMP``.  The returned value is the
        raw derivative (not scaled by 1/η).

        Region definitions match ``phase()``.

        Parameters
        ----------
        Mf          : (B, F) or (B, 1) — dimensionless frequency Mf
        phase_coeffs: dict — output of get_phase_coeffs
        derived     : (B, 13) — unused, kept for API symmetry

        Returns
        -------
        dphi : (B, F) or (B, 1)
        """
        pc   = phase_coeffs
        fIN  = pc['fPhaseMatchIN']
        fIM  = pc['fPhaseMatchIM']

        dphase0 = pc['dphase0']
        fRING = pc['fRING']
        fDAMP = pc['fDAMP']
        C2Int = pc['C2Int']
        C2MRD = pc['C2MRD']

        # ---- Inspiral derivative ----
        logMf = torch.log(Mf)
        dphi_ins = (
            pc['dphi0']
            + pc['dphi1']  * Mf.pow(1.0/3.0)
            + pc['dphi2']  * Mf.pow(2.0/3.0)
            + pc['dphi3']  * Mf
            + pc['dphi4']  * Mf.pow(4.0/3.0)
            + pc['dphi5']  * Mf.pow(5.0/3.0)
            + pc['dphi6']  * Mf * Mf
            + pc['dphi6L'] * Mf * Mf * logMf
            + pc['dphi7']  * Mf.pow(7.0/3.0)
            + pc['dphi8']  * Mf.pow(8.0/3.0)
            + pc['dphi8L'] * Mf.pow(8.0/3.0) * logMf
            # pseudo-PN inspiral terms (a0..a3 are derivatives, not integrals)
            + pc['a0'] * Mf.pow(8.0/3.0)
            + pc['a1'] * Mf.pow(3.0)
            + pc['a2'] * Mf.pow(10.0/3.0)
            + pc['a3'] * Mf.pow(11.0/3.0)
        ) * Mf.pow(-8.0/3.0) * dphase0

        # ---- Intermediate derivative (case 105: includes b3/f^3) ----
        inv1 = 1.0 / Mf
        inv2 = inv1 * inv1
        inv3 = inv2 * inv1
        inv4 = inv2 * inv2
        lorentz_int = 4.0 * pc['cL'] / (
            4.0*fDAMP*fDAMP + (Mf - fRING)*(Mf - fRING)
        )
        dphi_int = pc['b0'] + pc['b1']*inv1 + pc['b2']*inv2 + pc['b3']*inv3 + pc['b4']*inv4 + lorentz_int + C2Int

        # ---- Ringdown derivative ----
        lorentz_rd = pc['cL'] / (fDAMP*fDAMP + (Mf - fRING)*(Mf - fRING))
        dphi_rd = (
            pc['c0']
            + pc['c1'] * Mf.pow(-1.0/3.0)
            + pc['c2'] * inv2
            + pc['c4'] * inv4
            + lorentz_rd
            + C2MRD
        )

        return torch.where(Mf < fIN, dphi_ins,
               torch.where(Mf < fIM, dphi_int, dphi_rd))

    # ------------------------------------------------------------------
    # Full waveform assembly
    # ------------------------------------------------------------------

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a BBH parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 8+)
            Columns: [m1, m2, chi1z, chi2z, distance, tc, phic, inclination, ...]
            Masses in solar masses, distance in Mpc, angles in radians.
        reproduce_lal : bool
            If True, skip FD tapering, tc shift, and df normalisation so
            the output can be compared directly with raw LALSim output at
            the same frequency grid.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex
            Plus and cross polarisations.  The first ``n_pad`` bins
            (DC to f_min) are always zero.
        """
        # ----------------------------------------------------------------
        # 1. Parameters and derived quantities
        # ----------------------------------------------------------------
        dist_Mpc = theta[:, 4:5]   # Mpc
        tc       = theta[:, 5:6]   # s
        phic     = theta[:, 6:7]   # rad  (orbital phase at f_ref)
        iota     = theta[:, 7:8]   # rad  (inclination)

        derived  = self.compute_derived_parameters(theta)
        M_s  = derived[:, 2:3]     # (B, 1) total mass in seconds
        eta  = derived[:, 3:4]
        delta= derived[:, 4:5]
        STotR= derived[:, 9:10]
        dchi = derived[:, 10:11]

        # ----------------------------------------------------------------
        # 2. Amplitude and phase coefficients
        # ----------------------------------------------------------------
        amp_coeffs   = self.get_amp_coeffs(derived)
        phase_coeffs = self.get_phase_coeffs(derived)

        fRING = phase_coeffs['fRING']   # (B, 1) in Mf
        fDAMP = phase_coeffs['fDAMP']   # (B, 1)

        # ----------------------------------------------------------------
        # 3. Overall amplitude prefactor  Amp0 = amp0 * ampNorm
        #    amp0    = M_meters * M_s / dist_m  (LALSim internals.c line 609)
        #    ampNorm = sqrt(2*eta/3) * pi^{-1/6}
        # ----------------------------------------------------------------
        M_m    = M_s * self.C                    # total mass in metres
        dist_m = dist_Mpc * self.Mpc             # luminosity distance in metres
        amp0   = M_m * M_s / dist_m              # (B, 1)
        ampNorm = torch.sqrt(2.0*eta / 3.0) * (self.PI ** (-1.0/6.0))
        Amp0   = amp0 * ampNorm                  # (B, 1)

        # ----------------------------------------------------------------
        # 4. Time-alignment shift  linb  (= tshift from TimeShift_22)
        #    Source: IMRPhenomX_TimeShift_22 in internals.c lines 2624-2643
        #
        #    linb_fit  = XLALSimIMRPhenomXLinb(eta, STotR, dchi, delta)
        #    frefFit   = fRING - fDAMP
        #    dphi22Ref = (1/eta) * dPhase_22(frefFit)  [raw, without C2 terms]
        #    psi4      = XLALSimIMRPhenomXPsi4ToStrain(eta, STotR, dchi)
        #    linb      = linb_fit - dphi22Ref - 2π*(500 + psi4)
        # ----------------------------------------------------------------
        linb_fit_val = IMRPhenomXAS._linb_fit(eta, STotR, dchi, delta)     # (B,1)
        psi4val      = IMRPhenomXAS._psi4tostrain_fit(eta, STotR, dchi)    # (B,1)

        # Phase derivative at fRING-fDAMP (always in ringdown region)
        frefFit   = fRING - fDAMP                                           # (B,1) in Mf
        dphi22Ref = (1.0 / eta) * self.dphase(frefFit, phase_coeffs, derived)  # (B,1)

        linb = linb_fit_val - dphi22Ref - 2.0*math.pi*(500.0 + psi4val)    # (B,1)

        # ----------------------------------------------------------------
        # 5. Reference phase  phifRef
        #    = -(1/η * Ψ₂₂(MfRef) + linb*MfRef) + 2*phic + π/4
        # ----------------------------------------------------------------
        MfRef   = self.f_ref * M_s                                    # (B,1)
        phi22ref = self.phase(MfRef, phase_coeffs, derived)           # (B,1)
        phifRef  = -(phi22ref / eta + linb * MfRef) + 2.0*phic + 0.25*math.pi  # (B,1)

        # ----------------------------------------------------------------
        # 6. Build h22(f) over the full frequency grid
        #    φ_total(Mf) = (1/η) Ψ₂₂(Mf) + linb·Mf + phifRef
        #    |h22|(Mf)   = Amp0 · Mf^{-7/6} · A(Mf)
        # ----------------------------------------------------------------
        Mf = self.f * M_s                                             # (B,F)

        phi22 = self.phase(Mf, phase_coeffs, derived)                # (B,F)
        phi_total = phi22 / eta + linb * Mf + phifRef                # (B,F)

        A_mf   = self.amp(Mf, amp_coeffs, derived)                   # (B,F)
        A_total = Amp0 * Mf.pow(-7.0/6.0) * A_mf                    # (B,F)

        # ----------------------------------------------------------------
        # 7. hp / hc from inclination
        #
        #    LALSim convention: h22 = A * exp(+i φ)
        #    hp = -(1/2) * sqrt(5/(4π)) * (1 + cos²ι) * h22
        #       = YLM * (1+cos²ι)/2 * A * exp(+i(φ + π))
        #    hc = -(1/2) * sqrt(5/(4π)) * cos(ι) * (-i) * h22
        #       = YLM * cos(ι)       * A * exp(+i(φ + π/2))
        #
        #    where YLM = sqrt(5/(4π))/2 ≈ 0.31539
        #    Source: LALSimIMRPhenomX.c line 893, standard mode decomposition
        # ----------------------------------------------------------------
        YLM = math.sqrt(5.0 / (4.0 * math.pi)) / 2.0               # ≈ 0.31539
        cos_iota = torch.cos(iota)                                   # (B,1)
        hp = torch.polar(
            YLM * 0.5 * A_total * (1.0 + cos_iota * cos_iota),
            phi_total + math.pi,
        )
        hc = torch.polar(
            YLM * A_total * cos_iota,
            phi_total + 0.5 * math.pi,
        )

        if not reproduce_lal:
            # ---- FD taper (Planck roll-on at f_min, roll-off at fcut) ----
            fcut = self.fM_CUT_XAS / M_s                             # (B,1) Hz
            _win = _taper_mod.fd_taper(
                f=self.f,
                f_min=self.f[0, 0].item(),
                f_cut=fcut,
                df=self.df,
            )
            hp = hp * _win
            hc = hc * _win

            # ---- Apply tc (time of coalescence) ----
            hp, hc = self.apply_tc(hp, hc, tc)

            # ---- df normalisation (match LALSim continuous-FT convention) ----
            hp = hp * self.df
            hc = hc * self.df

        # ---- Zero-pad from DC to f_min ----
        hp, hc = self.pad_missing_frequencies(hp, hc)
        return hp, hc

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

        # Symmetric mass ratio — clip at 0.25 to handle roundoff (mirrors LALSim:
        # "if(eta > 0.25) eta = 0.25;")
        # delta = sqrt(1 - 4*eta) is 0 for equal-mass; no divisions by delta
        # exist in the fit functions, so eta = 0.25 is safe.
        eta = m1_s * m2_s / (M_s * M_s)
        eta.clamp_(max=0.25)

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
        F1_int = fAmpMatchIN            # left  boundary of intermediate region
        F2_int = F1_int + 0.5*(F4 - F1_int)   # midpoint  (LAL: F2 = F1 + 0.5*(F4-F1))

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

        Mirrors ``IMRPhenomXGetPhaseCoefficients`` +
        ``IMRPhenomX_Phase_22_ConnectionCoefficients`` in
        LALSimIMRPhenomX_internals.c.  Fixed to default LAL flags:
          IMRPhenomXInspiralPhaseVersion   = 104
          IMRPhenomXIntermediatePhaseVersion = 104
          IMRPhenomXRingdownPhaseVersion    = 105

        Parameters
        ----------
        derived : (B, 13) — output of compute_derived_parameters

        Returns
        -------
        pc : dict  — all tensors (B, 1) unless noted:
            inspiral: phi0..phi9L (PN integrals),
                      sigma1..sigma5 (pseudo-PN integral coefficients),
                      a0..a3 (pseudo-PN derivative coefficients),
                      phiNorm, dphase0
            intermediate: b0..b4
            ringdown: c0, c1, c2, c4, cL, cLovfda, c4ov3
            boundaries: fPhaseMatchIN, fPhaseMatchIM
            continuity: C1Int, C2Int, C1MRD, C2MRD
        """
        # ---------------------------------------------------------------
        # Unpack derived parameters
        # ---------------------------------------------------------------
        eta      = derived[:, 3:4]
        delta    = derived[:, 4:5]
        chi1     = derived[:, 11:12]   # chi1z
        chi2     = derived[:, 12:13]   # chi2z
        STotR    = derived[:, 9:10]
        chiPNHat = derived[:, 8:9]
        dchi     = derived[:, 10:11]

        eta2  = eta  * eta
        eta3  = eta2 * eta
        eta4  = eta3 * eta

        chi1L  = chi1    # aligned-spin = projected on L
        chi2L  = chi2
        chi1L2 = chi1L * chi1L
        chi1L3 = chi1L2 * chi1L
        chi2L2 = chi2L * chi2L
        chi2L3 = chi2L2 * chi2L
        chi1L2L = chi1L * chi2L  # cross term

        pi    = math.pi
        GAMMA = 0.5772156649015328606   # Euler–Mascheroni constant
        log2  = math.log(2.0)
        logpi = math.log(pi)

        # Powers of pi used in PN coefficients (scalar constants)
        pi2o3 = pi ** (2.0/3.0)
        pi4o3 = pi ** (4.0/3.0)
        pi5o3 = pi ** (5.0/3.0)
        pi7o3 = pi ** (7.0/3.0)
        pi2   = pi * pi
        pi8o3 = pi ** (8.0/3.0)

        # ---------------------------------------------------------------
        # Normalisations
        # ---------------------------------------------------------------
        phiNorm  = -(3.0 / (128.0 * pi5o3))    # = phiNorm in LAL
        dphase0  = 5.0 / (128.0 * pi5o3)        # = dphase0 in LAL

        # ---------------------------------------------------------------
        # 1. Final mass / spin → QNM frequencies (same as amplitude)
        # ---------------------------------------------------------------
        Mfinal = IMRPhenomXAS.final_mass_2017(eta, STotR, dchi, delta)
        afinal = IMRPhenomXAS.final_spin_2017(eta, STotR, dchi, delta)
        fRING, fDAMP = IMRPhenomXAS.get_fRD_fdamp(afinal, Mfinal)
        fmeco  = IMRPhenomXAS.fMECO(eta, chiPNHat, dchi, delta)
        fisco  = IMRPhenomXAS.fISCO(afinal)

        # ---------------------------------------------------------------
        # 2. Phase region boundaries (Sect. VII of arXiv:2001.11412)
        # ---------------------------------------------------------------
        fIMmatch = 0.6 * (0.5 * fRING + fisco)      # Eq. 5.11
        fINmatch = fmeco                              # MECO
        deltaf   = (fIMmatch - fINmatch) * 0.03      # Eq. 5.10

        fPhaseMatchIN  = fINmatch - deltaf            # Eq. 7.7 f_L
        fPhaseMatchIM  = fIMmatch + 0.5 * deltaf      # Eq. 7.7 f_H
        fPhaseInsMin   = torch.full_like(fRING, 0.0026)
        fPhaseInsMax   = 1.020 * fmeco
        fPhaseRDMin    = fIMmatch
        fPhaseRDMax    = fRING + 1.25 * fDAMP

        # ---------------------------------------------------------------
        # 3. TaylorF2 PN phase coefficients (phi0..phi9L)
        #    Source: Lines 1754-1887 of LALSimIMRPhenomX_internals.c
        #    Convention: coefficients are *already* multiplied by powers
        #    of pi so that phi_PN(Mf) = phiNorm * Mf^{-5/3} * sum_n phi_n*Mf^{n/3}
        # ---------------------------------------------------------------

        # --- 0.0 PN (Newtonian) ---
        phi0 = torch.ones_like(eta)   # = 1.0

        # --- 0.5 PN ---
        phi1 = torch.zeros_like(eta)  # = 0

        # --- 1.0 PN ---
        phi2 = (3715.0/756.0 + (55.0 * eta) / 9.0) * pi2o3

        # --- 1.5 PN ---
        phi3NS = -16.0 * pi2
        phi3S  = (
            (113.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
             - 76.0*(chi1L + chi2L)*eta) / 6.0
        ) * pi
        phi3 = phi3NS + phi3S

        # --- 2.0 PN ---
        phi4NS = (
            15293365.0/508032.0 + (27145.0 * eta)/504.0 + (3085.0 * eta2)/72.0
        ) * pi4o3
        phi4S  = (
            -5.0*(81.0*chi1L2*(1.0 + delta - 2.0*eta)
                  + 316.0*chi1L2L*eta
                  - 81.0*chi2L2*(-1.0 + delta + 2.0*eta)) / 16.0
        ) * pi4o3
        phi4 = phi4NS + phi4S

        # --- 2.5 PN (phi5 = 0; only log term survives) ---
        phi5 = torch.zeros_like(eta)

        phi5LNS = (5.0*(46374.0 - 6552.0*eta)*pi / 4536.0) * pi5o3
        phi5LS  = (
            (-732985.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
             - 560.0*(-1213.0*(chi1L + chi2L) + 63.0*(chi1L - chi2L)*delta)*eta
             + 85680.0*(chi1L + chi2L)*eta2) / 4536.0
        ) * pi5o3
        phi5L = phi5LNS + phi5LS

        # --- 3.0 PN ---
        phi6NS = (
            11583231236531.0 / 4.69421568e9
            - (5.0*eta*(3147553127.0 + 588.0*eta*(-45633.0 + 102260.0*eta)))/3.048192e6
            - (6848.0*GAMMA)/21.0
            - (640.0*pi2)/3.0
            + (2255.0*eta*pi2)/12.0
            - (13696.0*log2)/21.0
            - (6848.0*logpi)/63.0
        ) * pi2
        phi6S  = (
            (5.0*(227.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
                  - 156.0*(chi1L + chi2L)*eta)*pi) / 3.0
            + (5.0*(20.0*chi1L2L*eta*(11763.0 + 12488.0*eta)
                    + 7.0*chi2L2*(-15103.0*(-1.0 + delta)
                                   + 2.0*(-21683.0 + 6580.0*delta)*eta - 9808.0*eta2)
                    - 7.0*chi1L2*(-15103.0*(1.0 + delta)
                                   + 2.0*(21683.0 + 6580.0*delta)*eta + 9808.0*eta2)))/4032.0
        ) * pi2
        phi6 = phi6NS + phi6S

        phi6LNS = (-6848.0/63.0) * pi2
        phi6L   = phi6LNS        # phi6LS = 0

        # --- 3.5 PN ---
        phi7NS = (
            5.0*(15419335.0 + 168.0*(75703.0 - 29618.0*eta)*eta)*pi / 254016.0
        ) * pi7o3
        phi7S  = (
            (5.0*(-5030016755.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
                   + 4.0*(2113331119.0*(chi1L + chi2L)
                          + 675484362.0*(chi1L - chi2L)*delta)*eta
                   - 1008.0*(208433.0*(chi1L + chi2L) + 25011.0*(chi1L - chi2L)*delta)*eta2
                   + 90514368.0*(chi1L + chi2L)*eta3)) / 6.096384e6
            - 5.0*(57.0*chi1L2*(1.0 + delta - 2.0*eta)
                    + 220.0*chi1L2L*eta
                    - 57.0*chi2L2*(-1.0 + delta + 2.0*eta))*pi
            + (14585.0*(-(chi2L3*(-1.0 + delta)) + chi1L3*(1.0 + delta))
               - 5.0*(chi2L3*(8819.0 - 2985.0*delta)
                      + 8439.0*chi1L*chi2L2*(-1.0 + delta)
                      - 8439.0*chi1L2*chi2L*(1.0 + delta)
                      + chi1L3*(8819.0 + 2985.0*delta))*eta
               + 40.0*(chi1L + chi2L)*(17.0*chi1L2 - 14.0*chi1L2L + 17.0*chi2L2)*eta2) / 48.0
        ) * pi7o3
        phi7 = phi7NS + phi7S

        # --- 4.0 PN (phi8NS = 0 for 104/105) ---
        # phi8S uses pi8o3 and (log(pi) - 1) factor
        spin_combo_8 = (
            1263141.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
            - 2.0*(794075.0*(chi1L + chi2L) + 178533.0*(chi1L - chi2L)*delta)*eta
            + 94344.0*(chi1L + chi2L)*eta2
        )
        # phi8: coefficient of Mf^{8/3} in integral: phi8S * pi^{8/3}
        # phi8S uses LAL_PI * (log(pi) - 1)
        phi8S  = (
            -5.0 * spin_combo_8 * pi * (logpi - 1.0) / 9072.0
        ) * pi8o3
        phi8L_S = (
            -5.0 * spin_combo_8 * pi / 9072.0
        ) * pi8o3
        phi8   = phi8S        # phi8NS = 0
        phi8L  = phi8L_S      # phi8LNS = 0

        # --- 4.5 PN: phi9 = phi9L = 0 for versions 104/105 ---
        phi9  = torch.zeros_like(eta)
        phi9L = torch.zeros_like(eta)

        # ---------------------------------------------------------------
        # 4. Derivative coefficients dphi (from d/dMf of phiNorm*Mf^{-5/3}*sum)
        #    Relation:  dphi_n = factor_n * phi_n
        #    where factor_n = -(v_n - 5/3)/1 = (5/3 - v_n) / (5/3)
        #    Source: lines 1916-1929 of LALSimIMRPhenomX_internals.c
        # ---------------------------------------------------------------
        dphi0  = phi0
        dphi1  = (4.0/5.0) * phi1
        dphi2  = (3.0/5.0) * phi2
        dphi3  = (2.0/5.0) * phi3
        dphi4  = (1.0/5.0) * phi4
        dphi5  = -(3.0/5.0) * phi5L          # phi5 = 0
        dphi6  = -(1.0/5.0) * phi6 - (3.0/5.0) * phi6L
        dphi6L = -(1.0/5.0) * phi6L
        dphi7  = -(2.0/5.0) * phi7
        dphi8  = -(3.0/5.0) * phi8 - (3.0/5.0) * phi8L
        dphi8L = -(3.0/5.0) * phi8L
        # phi9 = phi9L = 0  →  dphi9 = dphi9L = 0

        # ---------------------------------------------------------------
        # 5. Inspiral pseudo-PN phase coefficients (case 104: 4 terms)
        #    Source: LALSimIMRPhenomX_inspiral.c – v3, d13, d23, d43
        #    Effective spin: chiPNHat
        # ---------------------------------------------------------------
        S    = chiPNHat
        S2   = S  * S
        S3   = S2 * S
        S4   = S3 * S
        dchi2 = dchi * dchi

        # v3  (absolute collocation value at the 3rd GC point)
        v3_ins = (
            (15415.0 + 873401.6255736464*eta + 376665.64637025696*eta2
             - 3.9719980569125614e6*eta3 + 8.913612508054944e6*eta4)
            / (1.0 + 46.83697749859996*eta)
            + (S*(397951.95299014193 - 207180.42746987*S
                  + eta3*(4.662143741417853e6 - 584728.050612325*S - 1.6894189124921719e6*S2)
                  + eta*(-1.0053073129700898e6 + 1.235279439281927e6*S - 174952.69161683554*S2)
                  - 130668.37221912303*S2
                  + eta2*(-1.9826323844247842e6 + 208349.45742548333*S + 895372.155565861*S2)))
            / (-9.675704197652225 + 3.5804521763363075*S + 2.5298346636273306*S2 + S3)
            + dchi2*(-1296.9289110696955*eta)
            + dchi*delta*eta*(-24708.109411857182 + 24703.28267342699*eta + 47752.17032707405*S)
        )

        # d13 = v1 - v3  (difference)
        d13_ins = (
            (-17294.0 - 19943.076428555978*eta + 483033.0998073767*eta2)
            / (1.0 + 4.460294035404433*eta)
            + (S*(68384.62786426462 + 67663.42759836042*S - 2179.3505885609297*S2
                  + eta*(-58475.33302037833 + 62190.404951852535*S + 18298.307770807573*S2 - 303141.1945565486*S3)
                  + 19703.894135534803*S3
                  + eta2*(-148368.4954044637 - 758386.5685734496*S - 137991.37032619823*S2 + 1.0765877367729193e6*S3)
                  + 32614.091002011017*S4))
            / (2.0412979553629143 + S)
            + 12017.062595934838*dchi*delta*eta
        )

        # d23 = v2 - v3  (difference)
        d23_ins = (
            (-7579.3 - 120297.86185566607*eta + 1.1694356931282217e6*eta2 - 557253.0066989232*eta3)
            / (1.0 + 18.53018618227582*eta)
            + (S*(-27089.36915061857 - 66228.9369155027*S
                  + eta2*(150022.21343386435 - 50166.382087278434*S - 399712.22891153296*S2)
                  - 44331.41741405198*S2
                  + eta*(50644.13475990821 + 157036.45676788126*S + 126736.43159783827*S2)
                  + eta3*(-593633.5370110178 - 325423.99477314285*S + 847483.2999508682*S2)))
            / (-1.5232497464826662 - 3.062957826830017*S - 1.130185486082531*S2 + S3)
            + 3843.083992827935*dchi*delta*eta
        )

        # d43 = v4 - v3  (difference)
        d43_ins = (
            (2439.0 - 31133.52170083207*eta + 28867.73328134167*eta2)
            / (1.0 + 0.41143032589262585*eta)
            + (S*(16116.057657391262
                  + eta3*(-375818.0132734753 - 386247.80765802023*S)
                  + eta*(-82355.86732027541 - 25843.06175439942*S)
                  + 9861.635308837876*S
                  + eta2*(229284.04542668918 + 117410.37432997991*S)))
            / (-3.7385208695213668 + 0.25294420589064653*S + S2)
            + 194.5554531509207*dchi*delta*eta
        )

        # Reconstruct absolute values: v1 = d13 + v3, v2 = d23 + v3, v4 = d43 + v3
        v1_ins = d13_ins + v3_ins
        v2_ins = d23_ins + v3_ins
        v4_ins = d43_ins + v3_ins

        # Gauss–Chebyshev collocation points in [fPhaseInsMin, fPhaseInsMax]
        # gpoints4 = [0, 1/4, 3/4, 1]
        dx_ins = fPhaseInsMax - fPhaseInsMin
        f1_ins = fPhaseInsMin                          # gpoints4[0] = 0
        f2_ins = fPhaseInsMin + 0.25 * dx_ins         # gpoints4[1] = 1/4
        f3_ins = fPhaseInsMin + 0.75 * dx_ins         # gpoints4[2] = 3/4
        f4_ins = fPhaseInsMax                          # gpoints4[3] = 1

        # Build 4×4 matrix A: ansatz a0 + a1*f^{1/3} + a2*f^{2/3} + a3*f
        def _ins_row(fi):
            fi13 = fi.pow(1.0/3.0)
            fi23 = fi13 * fi13
            return torch.cat([torch.ones_like(fi), fi13, fi23, fi], dim=-1)

        A_ins = torch.stack([
            _ins_row(f1_ins), _ins_row(f2_ins),
            _ins_row(f3_ins), _ins_row(f4_ins),
        ], dim=-2)  # (B, 4, 4)

        b_ins = torch.stack(
            [v1_ins, v2_ins, v3_ins, v4_ins], dim=-2
        )  # (B, 4, 1)

        x_ins = torch.linalg.solve(A_ins, b_ins)  # (B, 4, 1)
        a0 = x_ins[:, 0, :]   # (B, 1)
        a1 = x_ins[:, 1, :]
        a2 = x_ins[:, 2, :]
        a3 = x_ins[:, 3, :]
        # a4 = 0 for case 104

        # Pseudo-PN sigma coefficients for the phase *integral*
        sigma1 = (-5.0/3.0) * a0    # contributes Mf term in integral
        sigma2 = (-5.0/4.0) * a1    # contributes Mf^{4/3}
        sigma3 = (-5.0/5.0) * a2    # = -a2, contributes Mf^{5/3}
        sigma4 = (-5.0/6.0) * a3    # contributes Mf^2
        sigma5 = torch.zeros_like(a0)  # a4 = 0

        # ---------------------------------------------------------------
        # 6. Ringdown phase coefficients (case 105: 5 terms)
        #    Source: LALSimIMRPhenomX_ringdown.c – v4, d12, d24, d34, d54
        #    Effective spin: STotR
        # ---------------------------------------------------------------
        S   = STotR
        S2  = S  * S
        S3  = S2 * S
        S4  = S3 * S
        S5  = S4 * S

        eta5 = eta4 * eta

        # v4 (direct collocation value)
        v4_rd = (
            (-85.86062966719405 - 4616.740713893726*eta - 4925.756920247186*eta2
             + 7732.064464348168*eta3 + 12828.269960300782*eta4 - 39783.51698102803*eta5)
            / (1.0 + 50.206318806624004*eta)
            + (S*(33.335857451144356 - 36.49019206094966*S
                  + eta3*(1497.3545918387515 - 101.72731770500685*S)*S
                  - 3.835967351280833*S2 + 2.302712009652155*S3
                  + eta2*(93.64156367505917 - 18.184492163348665*S + 423.48863373726243*S2
                          - 104.36120236420928*S3 - 719.8775484010988*S4)
                  + 1.6533417657003922*S4
                  + eta*(-69.19412903018717 + 26.580344399838758*S - 15.399770764623746*S2
                          + 31.231253209893488*S3 + 97.69027029734173*S4)
                  + eta4*(1075.8686153198323 - 3443.0233614187396*S - 4253.974688619423*S2
                          - 608.2901586790335*S3 + 5064.173605639933*S4)))
            / (-1.3705601055555852 + S)
            + dchi*delta*eta*(22.363215261437862 + 156.08206945239374*eta)
        )

        # d12 = v1 - v2
        d12_rd = (
            (eta*(0.7207992174994245 - 1.237332073800276*eta + 6.086871214811216*eta2))
            / (0.006851189888541745 + 0.06099184229137391*eta - 0.15500218299268662*eta2 + eta3)
            + ((0.06519048552628343 - 25.25397971063995*eta - 308.62513664956975*eta4
                + 58.59408241189781*eta2 + 160.14971486043524*eta3)*S
               + eta*(-5.215945111216946 + 153.95945758807616*eta - 693.0504179144295*eta2
                       + 835.1725103648205*eta3)*S2
               + (0.20035146870472367 - 0.28745205203100666*eta - 47.56042058800358*eta4)*S3
               + eta*(5.7756520242745735 - 43.97332874253772*eta + 338.7263666984089*eta3)*S4
               + (-0.2697933899920511 + 4.917070939324979*eta - 22.384949087140086*eta4
                  - 11.61488280763592*eta2)*S5)
            / (1.0 - 0.6628745847248266*S)
            - 23.504907495268824*dchi*delta*eta2
        )

        # d24 = v2 - v4
        d24_rd = (
            (eta*(-9.460253118496386 + 9.429314399633007*eta + 64.69109972468395*eta2))
            / (-0.0670554310666559 - 0.09987544893382533*eta + eta2)
            + (17.36495157980372*eta*S
               + eta3*S*(930.3458437154668 + 808.457330742532*S)
               + eta4*S*(-774.3633787391745 - 2177.554979351284*S - 1031.846477275069*S2)
               + eta2*S*(-191.00932194869588 - 62.997389062600035*S + 64.42947340363101*S2)
               + 0.04497628581617564*S3)
            / (1.0 - 0.7267610313751913*S)
            + dchi*delta*(-36.66374091965371 + 91.60477826830407*eta)*eta2
        )

        # d34 = v3 - v4
        d34_rd = (
            (eta*(-8.506898502692536 + 13.936621412517798*eta)) / (-0.40919671232073945 + eta)
            + (eta*(1.7280582989361533*S + 18.41570325463385*S3 - 13.743271480938104*S4)
               + eta2*(73.8367329022058*S - 95.57802408341716*S3 + 215.78111099820157*S4)
               + 0.046849371468156265*S2
               + eta3*S*(-27.976989112929353 + 6.404060932334562*S - 633.1966645925428*S3 + 109.04824706217418*S2))
            / (1.0 - 0.6862449113932192*S)
            + 641.8965762829259*dchi*delta*eta5
        )

        # d54 = v5 - v4
        d54_rd = (
            (eta*(7.05731400277692 + 22.455288821807095*eta + 119.43820622871043*eta2))
            / (0.26026709603623255 + eta)
            + (eta2*(134.88158268621922 - 56.05992404859163*S)*S
               + eta*S*(-7.9407123129681425 + 9.486783128047414*S)
               + eta3*S*(-316.26970506215554 + 90.31815139272628*S))
            / (1.0 - 0.7162058321905909*S)
            + 43.82713604567481*dchi*delta*eta3
        )

        # Reconstruct absolute values: vj = d_j4 + v4
        v2_rd = d24_rd + v4_rd
        v3_rd = d34_rd + v4_rd
        v5_rd = d54_rd + v4_rd
        v1_rd = d12_rd + v2_rd    # v1 = d12 + v2

        # GC collocation points for ringdown in [fPhaseRDMin, fPhaseRDMax]
        # gpoints5 = [0, 1/2 - 1/(2*sqrt(2)), 1/2, 1/2 + 1/(2*sqrt(2)), 1]
        half_inv_sqrt2 = 1.0 / (2.0 * math.sqrt(2.0))
        gp5 = [0.0, 0.5 - half_inv_sqrt2, 0.5, 0.5 + half_inv_sqrt2, 1.0]
        dx_rd = fPhaseRDMax - fPhaseRDMin
        fi_rd = [fPhaseRDMin + gp * dx_rd for gp in gp5]
        fi_rd[3] = fRING    # override point 4 (0-indexed) to fRING exactly

        # Build 5×5 matrix for RD phase derivative ansatz:
        #   dphase_RD(f) = c0 + c1*f^{-1/3} + c2*f^{-2} + c4*f^{-4}
        #                + cRD * [ -dphase0 / (fDAMP^2 + (f-fRING)^2) ]
        def _rd_row(fi):
            inv_fi   = 1.0 / fi
            fi_m1o3  = fi.pow(-1.0/3.0)
            fi_m2    = inv_fi * inv_fi
            fi_m4    = fi_m2 * fi_m2
            lorentz  = -dphase0 / (fDAMP*fDAMP + (fi - fRING)*(fi - fRING))
            return torch.cat([torch.ones_like(fi), fi_m1o3, fi_m2, fi_m4, lorentz], dim=-1)

        A_rd = torch.stack(
            [_rd_row(fi_rd[i]) for i in range(5)], dim=-2
        )  # (B, 5, 5)

        b_rd = torch.stack(
            [v1_rd, v2_rd, v3_rd, v4_rd, v5_rd], dim=-2
        )  # (B, 5, 1)

        x_rd = torch.linalg.solve(A_rd, b_rd)  # (B, 5, 1)
        c0   = x_rd[:, 0, :]
        c1   = x_rd[:, 1, :]
        c2   = x_rd[:, 2, :]
        c4   = x_rd[:, 3, :]
        cRD  = x_rd[:, 4, :]
        cL   = -dphase0 * cRD            # Eq. 7.12: cL = -dphase0 * aRD
        cLovfda = cL / fDAMP
        c4ov3   = c4 / 3.0

        # phaseRD = v1_rd  (used in intermediate fit)
        phaseRD = v1_rd

        # ---------------------------------------------------------------
        # 7. Inspiral phase derivative at fPhaseMatchIN (= phaseIN in LAL)
        #    Used as collocation value for the intermediate phase fit.
        #    Source: lines 1959-1985 of LALSimIMRPhenomX_internals.c
        # ---------------------------------------------------------------
        fIN = fPhaseMatchIN
        logfIN = torch.log(fIN)

        phaseIN = (
            dphi0
            + dphi1  * fIN.pow(1.0/3.0)
            + dphi2  * fIN.pow(2.0/3.0)
            + dphi3  * fIN
            + dphi4  * fIN.pow(4.0/3.0)
            + dphi5  * fIN.pow(5.0/3.0)
            + dphi6  * fIN * fIN
            + dphi6L * fIN * fIN * logfIN
            + dphi7  * fIN.pow(7.0/3.0)
            + dphi8  * fIN.pow(8.0/3.0)
            + dphi8L * fIN.pow(8.0/3.0) * logfIN
            # phi9 = phi9L = 0 for versions 104/105
            # pseudo-PN a terms (contribute Mf^{8/3}, Mf^3, Mf^{10/3}, Mf^{11/3})
            + a0 * fIN.pow(8.0/3.0)
            + a1 * fIN.pow(3.0)
            + a2 * fIN.pow(10.0/3.0)
            + a3 * fIN.pow(11.0/3.0)
        ) * fIN.pow(-8.0/3.0) * dphase0

        # ---------------------------------------------------------------
        # 8. Intermediate phase coefficients (case 105: 5 terms)
        #    Source: lines 2130-2307 of LALSimIMRPhenomX_internals.c
        #    Default IMRPhenomXIntermediatePhaseVersion = 105
        #    Effective spin: STotR
        # ---------------------------------------------------------------
        S   = STotR
        S2  = S * S
        eta6 = eta3 * eta3

        # Four intermediate phase collocation fits (STotR spin, version 105)
        v2mRDv4 = (   # v2_IM - v4_RD (conditioned fit, version 105)
            (eta*(0.9951733419499662 + 101.21991715215253*eta + 632.4731389009143*eta2))
            / (0.00016803066316882238 + 0.11412314719189287*eta + 1.8413983770369362*eta2 + eta3)
            + (S*(18.694178521101332 + 16.89845522539974*S + 4941.31613710257*eta2*S
                  + eta*(-697.6773920613674 - 147.53381808989846*S2) + 0.3612417066833153*S2
                  + eta3*(3531.552143264721 - 14302.70838220423*S + 178.85850322465944*S2)))
            / (2.965640445745779 - 2.7706595614504725*S + S2)
            + dchi*delta*eta2*(356.74395864902294 + 1693.326644293169*eta2*S)
        )

        v3mRDv4 = (   # v3_IM - v4_RD (conditioned fit, version 105)
            (eta*(-5.126358906504587 - 227.46830225846668*eta + 688.3609087244353*eta2
                  - 751.4184178636324*eta3))
            / (-0.004551938711031158 - 0.7811680872741462*eta + eta2)
            + (S*(0.1549280856660919 - 0.9539250460041732*S - 539.4071941841604*eta2*S
                  + eta*(73.79645135116367 - 8.13494176717772*S2) - 2.84311102369862*S2
                  + eta3*(-936.3740515136005 + 1862.9097047992134*S + 224.77581754671272*S2)))
            / (-1.5308507364054487 + S)
            + 2993.3598520496153*dchi*delta*eta6
        )

        v2IM = (   # direct fit to v2_IM (version 105)
            (-82.54500000000004 - 5.58197349185435e6*eta - 3.5225742421184325e8*eta2
             + 1.4667258334378073e9*eta3)
            / (1.0 + 66757.12830903867*eta + 5.385164380400193e6*eta2
               + 2.5176585751772933e6*eta3)
            + (S*(19.416719811164853 - 36.066611959079935*S - 0.8612656616290079*S2
                  + eta2*(170.97203068800542 - 107.41099349364234*S - 647.8103976942541*S3)
                  + 5.95010003393006*S3
                  + eta3*(-1365.1499998427248 + 1152.425940764218*S + 415.7134909564443*S2
                          + 1897.5444343138167*S3 - 866.283566780576*S4)
                  + 4.984750041013893*S4
                  + eta*(207.69898051583655 - 132.88417400679026*S - 17.671713040498304*S2
                         + 29.071788188638315*S3 + 37.462217031512786*S4)))
            / (-1.1492259468169692 + S)
            + dchi*delta*eta3*(7343.130973149263 - 20486.813161100774*eta + 515.9898508588834*S)
        )

        d43_int = (   # d43 = v4_IM - v3_IM (version 105, no equivalent in 104)
            (0.4248820426833804 - 906.746595921514*eta - 282820.39946006844*eta2
             - 967049.2793750163*eta3 + 670077.5414916876*eta4)
            / (1.0 + 1670.9440812294847*eta + 19783.077247023448*eta2)
            + (S*(0.22814271667259703 + 1.1366593671801855*S
                  + eta3*(3499.432393555856 - 877.8811492839261*S - 4974.189172654984*S2)
                  + eta*(12.840649528989287 - 61.17248283184154*S2) + 0.4818323187946999*S2
                  + eta2*(-711.8532052499075 + 269.9234918621958*S + 941.6974723887743*S2)
                  + eta4*(-4939.642457025497 - 227.7672020783411*S + 8745.201037897836*S2)))
            / (-1.2442293719740283 + S)
            + dchi*delta*(-514.8494071830514 + 1493.3851099678195*eta)*eta3
        )

        # Collocation values (case 105):
        #   v1_int = phaseIN  (inspiral derivative at fPhaseMatchIN)
        #   v2_int = weighted average of conditioned and direct fit
        #   v3_int = v3mRDv4 + v4_rd
        #   v4_int = d43_int + v3_int  (new fourth point)
        #   v5_int = phaseRD = v1_rd  (first RD collocation = d12 + d24 + v4_rd)
        v1_int = phaseIN
        v2_int = 0.75 * (v2mRDv4 + v4_rd) + 0.25 * v2IM
        v3_int = v3mRDv4 + v4_rd
        v4_int = d43_int + v3_int
        v5_int = phaseRD     # = v1_rd = d12 + d24 + v4_rd

        # GC collocation points for intermediate in [fPhaseMatchIN, fPhaseMatchIM]
        # gpoints5 = [0, 0.5-1/(2√2), 0.5, 0.5+1/(2√2), 1.0]
        dx_int = fPhaseMatchIM - fPhaseMatchIN
        f1_int = fPhaseMatchIN
        f2_int = fPhaseMatchIN + (0.5 - half_inv_sqrt2) * dx_int
        f3_int = fPhaseMatchIN + 0.5 * dx_int
        f4_int = fPhaseMatchIN + (0.5 + half_inv_sqrt2) * dx_int
        f5_int = fPhaseMatchIM

        # For case 105, the matrix ansatz is:
        #   dphase_int(f) = x[0] + x[1]*rt + x[2]*rt^2 + x[3]*rt^3 + x[4]*rt^4
        #   where rt = fRING/f
        # The Lorentzian (4*cL)/(4*fDAMP^2+(f-fRING)^2) is subtracted from b.
        # LALSim stores:  b1=x[1]*fRING, b2=x[2]*fRING^2, b3=x[3]*fRING^3, b4=x[4]*fRING^4
        # Source: internals.c lines 2130-2307 (case 105)
        def _int_row(fi):
            rt = fRING / fi       # = fRING / f
            rt2 = rt * rt
            rt3 = rt * rt2        # (fRING/f)^3
            rt4 = rt2 * rt2       # (fRING/f)^4
            return torch.cat([torch.ones_like(fi), rt, rt2, rt3, rt4], dim=-1)

        def _int_lorentz(fi):
            return 4.0 * cL / (4.0 * fDAMP*fDAMP + (fi - fRING)*(fi - fRING))

        A_int = torch.stack([
            _int_row(f1_int), _int_row(f2_int), _int_row(f3_int),
            _int_row(f4_int), _int_row(f5_int),
        ], dim=-2)  # (B, 5, 5)

        b_int = torch.stack([
            v1_int - _int_lorentz(f1_int),
            v2_int - _int_lorentz(f2_int),
            v3_int - _int_lorentz(f3_int),
            v4_int - _int_lorentz(f4_int),
            v5_int - _int_lorentz(f5_int),
        ], dim=-2)  # (B, 5, 1)

        x_int = torch.linalg.solve(A_int, b_int)  # (B, 5, 1)
        b0_raw = x_int[:, 0, :]
        b1_raw = x_int[:, 1, :]
        b2_raw = x_int[:, 2, :]
        b3_raw = x_int[:, 3, :]
        b4_raw = x_int[:, 4, :]

        # Rescale back to physical coefficients
        # b1 = x[1] * fRING  (so that b1/f = x[1]*(fRING/f))
        b0 = b0_raw
        b1 = b1_raw * fRING
        b2 = b2_raw * fRING * fRING
        b3 = b3_raw * fRING * fRING * fRING
        b4 = b4_raw * fRING * fRING * fRING * fRING

        # ---------------------------------------------------------------
        # 9. Phase-continuity constants (C1Int, C2Int, C1MRD, C2MRD)
        #    Source: IMRPhenomX_Phase_22_ConnectionCoefficients in internals.c
        # ---------------------------------------------------------------
        fIns = fPhaseMatchIN
        fInt = fPhaseMatchIM

        # Helper: evaluate inspiral derivative at f
        def _dphi_ins(f):
            logf = torch.log(f)
            return (
                dphi0
                + dphi1  * f.pow(1.0/3.0)
                + dphi2  * f.pow(2.0/3.0)
                + dphi3  * f
                + dphi4  * f.pow(4.0/3.0)
                + dphi5  * f.pow(5.0/3.0)
                + dphi6  * f * f
                + dphi6L * f * f * logf
                + dphi7  * f.pow(7.0/3.0)
                + dphi8  * f.pow(8.0/3.0)
                + dphi8L * f.pow(8.0/3.0) * logf
                + a0 * f.pow(8.0/3.0)
                + a1 * f.pow(3.0)
                + a2 * f.pow(10.0/3.0)
                + a3 * f.pow(11.0/3.0)
            ) * f.pow(-8.0/3.0) * dphase0

        # Helper: evaluate inspiral phase integral at f
        def _phi_ins(f):
            logf = torch.log(f)
            phasing = (
                phi0
                + phi1  * f.pow(1.0/3.0)
                + phi2  * f.pow(2.0/3.0)
                + phi3  * f
                + phi4  * f.pow(4.0/3.0)
                + phi5  * f.pow(5.0/3.0)   # phi5 = 0
                + phi5L * f.pow(5.0/3.0) * logf
                + phi6  * f.pow(2.0)
                + phi6L * f.pow(2.0) * logf
                + phi7  * f.pow(7.0/3.0)
                + phi8  * f.pow(8.0/3.0)
                + phi8L * f.pow(8.0/3.0) * logf
                + phi9  * f.pow(3.0)       # phi9 = 0
                + phi9L * f.pow(3.0) * logf  # phi9L = 0
                # pseudo-PN sigma terms
                + sigma1 * f.pow(8.0/3.0)   # σ₁ → Mf term after * Mf^{-5/3}
                + sigma2 * f.pow(3.0)
                + sigma3 * f.pow(10.0/3.0)
                + sigma4 * f.pow(11.0/3.0)
                + sigma5 * f.pow(4.0)       # sigma5 = 0
            )
            return phiNorm * f.pow(-5.0/3.0) * phasing

        # Helper: evaluate intermediate derivative at f (case 105: includes b3/f^3)
        def _dphi_int(f):
            inv1 = 1.0 / f
            inv2 = inv1 * inv1
            inv3 = inv2 * inv1
            inv4 = inv2 * inv2
            lorentz = 4.0 * cL / (4.0*fDAMP*fDAMP + (f - fRING)*(f - fRING))
            return b0 + b1*inv1 + b2*inv2 + b3*inv3 + b4*inv4 + lorentz

        # Helper: evaluate intermediate phase integral at f (case 105: includes -b3/(2*f^2))
        def _phi_int(f):
            inv1 = 1.0 / f
            inv2 = inv1 * inv1
            inv3 = inv2 * inv1
            logf = torch.log(f)
            atan_term = (2.0 * cL / fDAMP) * torch.atan((f - fRING) / (2.0 * fDAMP))
            return b0*f + b1*logf - b2*inv1 - (b3/2.0)*inv2 - (b4/3.0)*inv3 + atan_term

        # Helper: evaluate ringdown derivative at f
        def _dphi_rd(f):
            inv1 = 1.0 / f
            fi_m1o3 = f.pow(-1.0/3.0)
            fi_m2   = inv1 * inv1
            fi_m4   = fi_m2 * fi_m2
            lorentz = cL / (fDAMP*fDAMP + (f - fRING)*(f - fRING))
            return c0 + c1*fi_m1o3 + c2*fi_m2 + c4*fi_m4 + lorentz

        # Helper: evaluate ringdown phase integral at f
        def _phi_rd(f):
            inv1 = 1.0 / f
            inv3 = inv1 * inv1 * inv1
            f2o3 = f.pow(2.0/3.0)
            atan_term = cLovfda * torch.atan((f - fRING) / fDAMP)
            return c0*f + 1.5*c1*f2o3 - c2*inv1 - c4ov3*inv3 + atan_term

        # C2Int = DPhiIns(fIns) - DPhiInt(fIns)
        C2Int = _dphi_ins(fIns) - _dphi_int(fIns)

        # C1Int = PhiIns(fIns) - PhiInt(fIns) - C2Int * fIns
        C1Int = _phi_ins(fIns) - _phi_int(fIns) - C2Int * fIns

        # C2MRD = [DPhiInt(fInt) + C2Int] - DPhiRD(fInt)
        C2MRD = _dphi_int(fInt) + C2Int - _dphi_rd(fInt)

        # C1MRD = [PhiInt(fInt) + C1Int + C2Int*fInt] - PhiRD(fInt) - C2MRD*fInt
        C1MRD = (_phi_int(fInt) + C1Int + C2Int*fInt) - _phi_rd(fInt) - C2MRD*fInt

        return {
            # Normalisations
            'phiNorm':  phiNorm,
            'dphase0':  dphase0,
            # PN integral coefficients
            'phi0': phi0, 'phi1': phi1, 'phi2': phi2, 'phi3': phi3,
            'phi4': phi4, 'phi5': phi5, 'phi5L': phi5L,
            'phi6': phi6, 'phi6L': phi6L,
            'phi7': phi7,
            'phi8': phi8, 'phi8L': phi8L,
            'phi9': phi9, 'phi9L': phi9L,
            # Pseudo-PN integral coefficients
            'sigma1': sigma1, 'sigma2': sigma2, 'sigma3': sigma3,
            'sigma4': sigma4, 'sigma5': sigma5,
            # Pseudo-PN derivative coefficients
            'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3,
            # PN derivative coefficients
            'dphi0': dphi0, 'dphi1': dphi1, 'dphi2': dphi2, 'dphi3': dphi3,
            'dphi4': dphi4, 'dphi5': dphi5,
            'dphi6': dphi6, 'dphi6L': dphi6L,
            'dphi7': dphi7,
            'dphi8': dphi8, 'dphi8L': dphi8L,
            # Intermediate coefficients (case 105: b3 ≠ 0)
            'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4,
            # Ringdown coefficients
            'c0': c0, 'c1': c1, 'c2': c2, 'c4': c4,
            'cL': cL, 'cLovfda': cLovfda, 'c4ov3': c4ov3,
            # QNM frequencies
            'fRING': fRING, 'fDAMP': fDAMP,
            # Boundary frequencies
            'fPhaseMatchIN': fPhaseMatchIN,
            'fPhaseMatchIM': fPhaseMatchIM,
            # Continuity constants
            'C1Int': C1Int, 'C2Int': C2Int,
            'C1MRD': C1MRD, 'C2MRD': C2MRD,
        }

    def phase(self, f_Ms, phase_coeffs, derived):
        """
        Evaluate the raw IMRPhenomXAS phase Ψ₂₂(Mf) over the frequency grid.

        Does NOT apply the 1/η re-scaling or the tc/phic offsets — those are
        handled in ``get_hphc``.  The full FD phase used to build h(f) is:

            phi_total = (1/η) * Ψ₂₂(Mf) + linb*Mf + phifRef

        where linb is the time-alignment shift and phifRef is the reference
        phase (both computed in get_hphc).

        Region definitions:
          Inspiral     :  Mf < fPhaseMatchIN
          Intermediate :  fPhaseMatchIN ≤ Mf < fPhaseMatchIM
          Ringdown     :  Mf ≥ fPhaseMatchIM

        Inspiral ansatz integral (Eq. 7.4 of arXiv:2001.11412):
            phiNorm * Mf^{-5/3} * [phi0 + phi1*Mf^{1/3} + … + sigma1*Mf^{8/3} + …]

        Intermediate ansatz integral (Eq. 7.7, case 105):
            b0*Mf + b1*log(Mf) - b2/Mf - b3/(2*Mf²) - b4/(3*Mf³)
            + (2*cL/fDAMP)*atan((Mf-fRING)/(2*fDAMP)) + C1Int + C2Int*Mf

        Ringdown ansatz integral (Eq. 7.12):
            c0*Mf + 1.5*c1*Mf^{2/3} - c2/Mf - (c4/3)*Mf^{-3}
            + (cL/fDAMP)*atan((Mf-fRING)/fDAMP) + C1MRD + C2MRD*Mf

        Parameters
        ----------
        f_Ms        : (B, F) — dimensionless frequency grid  Mf = f·M_s
        phase_coeffs: dict   — output of get_phase_coeffs
        derived     : (B, 13) — (unused here; kept for API symmetry)

        Returns
        -------
        psi : (B, F) — raw phase Ψ₂₂(Mf) in radians
        """
        pc = phase_coeffs
        Mf = f_Ms

        fIN  = pc['fPhaseMatchIN']   # (B, 1) inspiral→intermediate boundary
        fIM  = pc['fPhaseMatchIM']   # (B, 1) intermediate→ringdown boundary
        fRING = pc['fRING']
        fDAMP = pc['fDAMP']
        cL    = pc['cL']
        cLovfda = pc['cLovfda']
        c4ov3   = pc['c4ov3']

        # ---- Inspiral ----
        logMf = torch.log(Mf)
        phi_ins = (
            pc['phi0']
            + pc['phi1']  * Mf.pow(1.0/3.0)
            + pc['phi2']  * Mf.pow(2.0/3.0)
            + pc['phi3']  * Mf
            + pc['phi4']  * Mf.pow(4.0/3.0)
            + pc['phi5']  * Mf.pow(5.0/3.0)
            + pc['phi5L'] * Mf.pow(5.0/3.0) * logMf
            + pc['phi6']  * Mf * Mf
            + pc['phi6L'] * Mf * Mf * logMf
            + pc['phi7']  * Mf.pow(7.0/3.0)
            + pc['phi8']  * Mf.pow(8.0/3.0)
            + pc['phi8L'] * Mf.pow(8.0/3.0) * logMf
            + pc['phi9']  * Mf.pow(3.0)
            + pc['phi9L'] * Mf.pow(3.0) * logMf
            # pseudo-PN sigma terms
            + pc['sigma1'] * Mf.pow(8.0/3.0)
            + pc['sigma2'] * Mf.pow(3.0)
            + pc['sigma3'] * Mf.pow(10.0/3.0)
            + pc['sigma4'] * Mf.pow(11.0/3.0)
            + pc['sigma5'] * Mf.pow(4.0)
        )
        phi_ins = pc['phiNorm'] * Mf.pow(-5.0/3.0) * phi_ins

        # ---- Intermediate (case 105: includes -b3/(2*f^2)) ----
        inv1_Mf = 1.0 / Mf
        inv2_Mf = inv1_Mf * inv1_Mf
        inv3_Mf = inv2_Mf * inv1_Mf
        atan_int = (2.0 * cL / fDAMP) * torch.atan((Mf - fRING) / (2.0 * fDAMP))
        phi_int  = (
            pc['b0'] * Mf
            + pc['b1'] * logMf
            - pc['b2'] * inv1_Mf
            - (pc['b3'] / 2.0) * inv2_Mf
            - (pc['b4'] / 3.0) * inv3_Mf
            + atan_int
            + pc['C1Int'] + pc['C2Int'] * Mf
        )

        # ---- Ringdown ----
        Mf_2o3  = Mf.pow(2.0/3.0)
        atan_rd = cLovfda * torch.atan((Mf - fRING) / fDAMP)
        phi_rd  = (
            pc['c0'] * Mf
            + 1.5 * pc['c1'] * Mf_2o3
            - pc['c2'] * inv1_Mf
            - c4ov3 * inv3_Mf
            + atan_rd
            + pc['C1MRD'] + pc['C2MRD'] * Mf
        )

        # ---- Stitch regions ----
        return torch.where(Mf < fIN, phi_ins,
               torch.where(Mf < fIM, phi_int, phi_rd))

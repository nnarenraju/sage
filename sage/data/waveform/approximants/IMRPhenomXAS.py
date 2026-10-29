#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch port of LALSimulation's IMRPhenomXAS (2,2) mode
(LALSimIMRPhenomX.c, LALSimIMRPhenomX_internals.c), the aligned-spin
base of IMRPhenomXAS_NRTidalv3, IMRPhenomXHM and IMRPhenomXPHM.
"""

import math
import warnings

import torch

from sage.core.constants import MSUN
from sage.data.waveform.approximants.phenom import PhenomConstants
from sage.data.waveform import taper as _taper_mod
from sage.data.waveform import waveform_utils


class IMRPhenomXAS(PhenomConstants):
    """
    Batched IMRPhenomXAS (2,2)-mode waveform on a fixed frequency grid.

    Parameters
    ----------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.
    padding_length_in_s : float
        Leading padding; tc is measured from its end.
    **kwargs
        Forwarded to PhenomConstants.

    Raises
    ------
    ValueError
        If a reference frequency is negative.
    """

    BODY_SWAP = [1, 0, 3, 2, 4, 5, 6, 7]

    def __init__(self, f, f_ref, padding_length_in_s, **kwargs):
        f = f.to(torch.float64)
        f_ref = f_ref.to(torch.float64)
        # Perform initial sanity checks (XLALSimIMRPhenomXASGenerateFD,
        # LALSimIMRPhenomX.c)
        if (f_ref < 0.0).any():
            raise ValueError("fRef_In must be positive or set to 0 to ignore.")
        # If no reference frequency is given, set it to the starting
        # gravitational wave frequency
        f_ref = torch.where(f_ref == 0.0, f[:, 0:1], f_ref)
        super().__init__(
            device=f.device,
            batch_size=f.shape[0],
            dtype=f.dtype,
            **kwargs,
        )

        self.f      = f
        _n = f.shape[1]
        self.df     = (f[0, -1] - f[0, 0]) / (_n - 1)
        self.sample_length_in_s = 1.0 / self.df
        self.f_numel = _n
        self.f_ref  = f_ref
        self.padding_length_in_s = padding_length_in_s
        self.B      = f.shape[0]

        # Index shift between freqs and the frequency series
        self.n_pad = int(round(self.f[0, 0].item() / self.df.item()))
        self.hp_buffer = torch.empty(
            (self.B, self.n_pad + self.f_numel),
            dtype=torch.complex128,
            device=f.device,
        )
        self.hc_buffer = torch.empty_like(self.hp_buffer)

        # Define the default end of the waveform as: 0.3 Mf. This value
        # is chosen such that the 44 mode also shows the ringdown part.
        self.fM_CUT_XAS = torch.tensor(0.3, device=f.device, dtype=f.dtype)

        self.f_min = self.f[0, 0].item()
        self.f_max = self.f[0, -1].item()
        self.delta_f = None

    @classmethod
    def from_band(cls, delta_f, low_frequency_cutoff_in_hz, f_max, reference_frequency_in_hz, batch_size, device):
        """
        The waveform alone, for LAL's deltaF, f_min, f_max and fRef.

        The output has LAL's NextPow2(f_max/deltaF) + 1 bins, zeros
        above the grid (XLALSimIMRPhenomXASGenerateFD,
        LALSimIMRPhenomX.c). With f_max = 0 each call ends at the
        highest fCut of its batch, LAL's length for that signal.

        Parameters
        ----------
        delta_f : float
            LAL's deltaF, in Hz.
        low_frequency_cutoff_in_hz, f_max : float
            LAL's f_min and f_max, in Hz.
        reference_frequency_in_hz : float
            LAL's fRef, in Hz.
        batch_size : int
            Signals per call.
        device : str or torch.device
            Device of the batches.

        Returns
        -------
        IMRPhenomXAS
            Model on LAL's grid.
        """
        model = cls.__new__(cls)
        if isinstance(model, torch.nn.Module):
            torch.nn.Module.__init__(model)
        model.set_band(low_frequency_cutoff_in_hz, reference_frequency_in_hz, f_max, delta_f,
                       cls.next_pow2(f_max / delta_f) + 1, 0.0, batch_size, device, torch.float64)
        return model

    def set_band(self, f_min, f_ref, f_max, delta_f, n_bins, padding_length_in_s, batch_size, device, dtype):
        """
        Set LAL's frequency grid, reference frequency and buffers.

        With f_max = 0 each call sets the grid, in checked_band.

        Parameters
        ----------
        f_min, f_ref, f_max, delta_f : float
            LAL's f_min, fRef, f_max and deltaF, in Hz.
        n_bins : int
            Bins of the output, from DC.
        padding_length_in_s : float
            Window padding, for the training-path tc placement.
        batch_size : int
            Signals per call.
        device : str or torch.device
            Device of the batches.
        dtype : torch.dtype
            Real dtype of the training-path output.

        Raises
        ------
        ValueError
            On LAL's frequency checks.
        """
        # Perform initial sanity checks (XLALSimIMRPhenomXASGenerateFD,
        # LALSimIMRPhenomX.c)
        if f_ref < 0.0:
            raise ValueError("fRef_In must be positive or set to 0 to ignore.")
        if delta_f <= 0.0:
            raise ValueError("deltaF must be positive.")
        if f_min <= 0.0:
            raise ValueError("f_min must be positive.")
        if f_max < 0.0:
            raise ValueError("f_max must be non-negative.")
        # If no reference frequency is given, set it to the starting
        # gravitational wave frequency
        if f_ref == 0.0:
            f_ref = f_min

        self.band = (f_min, f_ref, f_max, delta_f, padding_length_in_s, batch_size, device)
        self._out_cdtype = torch.promote_types(dtype, torch.complex64)
        if f_max:
            self.set_grid(f_max, n_bins)
        else:
            PhenomConstants.__init__(self, device=device, batch_size=batch_size, dtype=torch.float64)
            self.f_min, self.f_max, self.delta_f = f_min, f_max, delta_f

    def set_grid(self, f_top, n_bins):
        """
        LAL's grid from f_min to f_top, and buffers of n_bins from DC.

        Parameters
        ----------
        f_top : float
            Highest frequency of the grid, in Hz.
        n_bins : int
            Bins of the output, from DC.
        """
        f_min, f_ref, f_max, delta_f, padding_length_in_s, batch_size, device = self.band
        # Populate frequency array (IMRPhenomXASGenerateFD,
        # LALSimIMRPhenomX.c)
        i_start = int(f_min / delta_f)
        i_stop = int(f_top / delta_f) + 1
        f = torch.arange(i_start, i_stop, device=device, dtype=torch.float64) * delta_f
        f = f.unsqueeze(0).expand(batch_size, -1).clone()
        f_ref = waveform_utils.reference_frequencies(f_ref, batch_size, device)

        IMRPhenomXAS.__init__(self, f, f_ref, padding_length_in_s=padding_length_in_s)
        self.f_min = f_min
        self.f_max = f_max
        self.delta_f = delta_f
        # Index shift between freqs and the frequency series
        self.n_pad = i_start
        self.hp_buffer = torch.empty((self.B, n_bins), dtype=torch.complex128, device=device)
        self.hc_buffer = torch.empty_like(self.hp_buffer)

    # Source: NextPow2 (LALSimIMRPhenomXUtilities.c).
    @staticmethod
    def next_pow2(n):
        """
        Closest power of 2 at or above int(n); 0 for int(n) = 0.

        Parameters
        ----------
        n : float
            Truncated to an integer, as LAL's size_t argument.

        Returns
        -------
        int
        """
        n = int(n)
        return int(2 ** math.ceil(math.log2(n))) if n else 0

    @staticmethod
    def raise_first(failures):
        """
        Raise the first failing check of the first failing row.

        Parameters
        ----------
        failures : list of (torch.Tensor, callable)
            Per-row failure masks, shape (B,), in LAL's order, each
            with a function of the row index that returns LAL's
            message.

        Raises
        ------
        ValueError
            When any row fails a check.
        """
        failed = torch.stack([mask for mask, _ in failures])
        if not failed.any():
            return
        row = int(failed.any(dim=0).nonzero()[0])
        check = int(failed[:, row].nonzero()[0])
        raise ValueError(f"signal {row}: {failures[check][1](row)}")

    # Source: XLALSimIMRPhenomXASGenerateFD (LALSimIMRPhenomX.c),
    # IMRPhenomXSetWaveformVariables (LALSimIMRPhenomX_internals.c).
    @classmethod
    def check_parameters(cls, theta):
        """
        Raise LAL's error for parameters it rejects.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 8)
            Aligned layout.

        Raises
        ------
        ValueError
            With LAL's message, for the first failing row.
        """
        m1, m2, chi1, chi2, distance = (theta[:, k] for k in range(5))
        m1_SI, m2_SI = m1 * MSUN, m2 * MSUN
        mass_ratio = torch.where(m1_SI > m2_SI, m1_SI / m2_SI, m2_SI / m1_SI)
        cls.raise_first([
            (m1_SI <= 0.0, lambda i: "m1 must be positive."),
            (m2_SI <= 0.0, lambda i: "m2 must be positive."),
            (distance < 0.0, lambda i: "Distance must be positive and greater than 0."),
            ((mass_ratio > 1000.0) & (torch.abs(mass_ratio - 1000.0) > 1e-12),
             lambda i: "ERROR: Model not valid at mass ratios beyond 1000."),
            ((chi1 > 1.0) | (chi1 < -1.0) | (chi2 > 1.0) | (chi2 < -1.0),
             lambda i: "Unphysical spins requested: must obey the Kerr bound [-1,1]."),
        ])

    @classmethod
    def order_bodies(cls, theta):
        """
        Order the bodies so that mass1 >= mass2, as LAL does.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, len(BODY_SWAP))
            Aligned layout.

        Returns
        -------
        torch.Tensor, same shape as theta
        """
        swap = ~(theta[:, 0:1] >= theta[:, 1:2])
        if swap.any():
            warnings.warn("Warning: m1 < m2, swapping the masses, spins, tidal deformabilities, "
                          "and spin-induced quadrupole parameters.")
        return torch.where(swap, theta[:, cls.BODY_SWAP], theta)

    def checked_band(self, derived, failures=()):
        """
        LAL's frequency band of each signal, after its checks.

        With f_max = 0 the grid is then set to the highest fCut of the
        batch. Source: IMRPhenomXSetWaveformVariables
        (LALSimIMRPhenomX_internals.c), IMRPhenomXASGenerateFD
        (LALSimIMRPhenomX.c).

        Parameters
        ----------
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        failures : list of (torch.Tensor, callable)
            Further checks, as for raise_first, after LAL's band checks.

        Returns
        -------
        f_max_prime : torch.Tensor, shape (B, 1)
            Upper frequency of each signal, in Hz.
        f_last : torch.Tensor, shape (B, 1), or None
            Last frequency of LAL's grid, in Hz; None on a given grid.

        Raises
        ------
        ValueError
            With LAL's message, for the first failing row.
        """
        M_s, eta = derived[:, 2:3], derived[:, 3:4]
        STotR, dchi = derived[:, 9:10], derived[:, 10:11]
        f_cut = torch.where(derived[:, 7:8] > 0.99, torch.full_like(M_s, 0.33), torch.full_like(M_s, 0.3)) / M_s
        f_max_prime = f_cut if self.f_max == 0.0 else torch.clamp(f_cut, max=self.f_max)
        for fails, message in ((f_cut <= self.f_min, "(fCut = {:g} Hz) <= f_min = {:g}"),
                               (f_max_prime <= self.f_min, "f_max <= f_min")):
            if fails.any():
                i = int(fails[:, 0].nonzero()[0])
                warnings.warn(message.format(float(f_cut[i]), self.f_min))
        afinal = self.final_spin_2017(eta, STotR, dchi)
        remnant = [
            (self.final_mass_2017(eta, STotR, dchi)[:, 0] > 1.0,
             lambda i: "IMRPhenomX_FinalMass2018: Final mass > 1.0 not physical."),
            (torch.abs(afinal[:, 0]) > 1.0, lambda i: "IMRPhenomX_FinalSpin2018: Final spin > 1.0 is not physical."),
            (self.fMECO(eta, derived[:, 8:9], dchi)[:, 0] > self.fISCO(afinal)[:, 0],
             lambda i: "Error: f_MECO cannot be greater than f_ISCO."),
        ]
        if self.delta_f is None:
            self.raise_first(remnant + list(failures))
            return f_max_prime, None
        i_start = int(self.f_min / self.delta_f)
        i_stop = (f_max_prime / self.delta_f).floor() + 1
        self.raise_first(remnant + [
            (i_start > i_stop[:, 0],
             lambda i: f"minimum freq index {i_start} and maximum freq index {int(i_stop[i])} do not fulfill "
                       f"0<=ind_min<=ind_max<=htilde->data>length={int(i_stop[i])}."),
        ] + list(failures))
        if self.f_max == 0.0:
            f_top = float(f_cut.max())
            self.set_grid(f_top, self.next_pow2(f_top / self.delta_f) + 1)
        return f_max_prime, (i_stop - 1) * self.delta_f

    @staticmethod
    def _linb_fit(eta, STotR, dchi, delta):
        """
        Parameter-space fit of dphi22(fring22 - fdamp22).

        Source: XLALSimIMRPhenomXLinb (LALSimIMRPhenomXUtilities.c).

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
        Fit of the time difference between the strain and psi4 peaks.

        Source: XLALSimIMRPhenomXPsi4ToStrain
        (LALSimIMRPhenomXUtilities.c).

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
        uneqSpin = 105.37711654943146 * dchi * torch.sqrt(1.0 - 4.0*eta) * eta2

        return noSpin + eqSpin + uneqSpin

    def dphase(self, Mf, phase_coeffs):
        """
        Phase derivative of the (2,2) mode, without the 1/eta scaling.

        Source: IMRPhenomX_dPhase_22 (LALSimIMRPhenomX_internals.c).

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F) or (B, 1)
            Dimensionless frequency.
        phase_coeffs : dict
            Output of get_phase_coeffs.

        Returns
        -------
        dphi : torch.Tensor, same shape as Mf
        """
        pc   = phase_coeffs
        fIN  = pc['fPhaseMatchIN']
        fIM  = pc['fPhaseMatchIM']

        dphase0 = pc['dphase0']
        fRING = pc['fRING']
        fDAMP = pc['fDAMP']
        C2Int = pc['C2Int']
        C2MRD = pc['C2MRD']

        # Phase Derivative: TaylorF2 + pseudo-PN coefficients
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
            # Add pseudo-PN Coefficient
            + pc['a0'] * Mf.pow(8.0/3.0)
            + pc['a1'] * Mf.pow(3.0)
            + pc['a2'] * Mf.pow(10.0/3.0)
            + pc['a3'] * Mf.pow(11.0/3.0)
        ) * Mf.pow(-8.0/3.0) * dphase0

        # Phase derivative ansatz for intermediate region
        inv1 = 1.0 / Mf
        inv2 = inv1 * inv1
        inv3 = inv2 * inv1
        inv4 = inv2 * inv2
        lorentz_int = 4.0 * pc['cL'] / (
            4.0*fDAMP*fDAMP + (Mf - fRING)*(Mf - fRING)
        )
        dphi_int = pc['b0'] + pc['b1']*inv1 + pc['b2']*inv2 + pc['b3']*inv3 + pc['b4']*inv4 + lorentz_int + C2Int

        # Phenomenological ringdown phase derivative ansatz
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

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip the FD taper, the tc shift and the df
            normalisation.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex
            Plus and cross polarisations.  The first ``n_pad`` bins
            (DC to f_min) are always zero.

        Raises
        ------
        ValueError
            If any transverse spin or tidal parameter is non-zero, or
            LAL rejects the parameters.
        """
        theta = waveform_utils.aligned_layout(
            theta, waveform_utils.ALIGNED_COLUMNS,
            waveform_utils.TRANSVERSE_SPIN_PARAMETERS + waveform_utils.TIDAL_PARAMETERS,
        )
        self.check_parameters(theta)
        theta = self.order_bodies(theta)
        dist_Mpc = theta[:, 4:5]
        tc       = theta[:, 5:6]
        phic     = theta[:, 6:7]   # Orbital phase at fRef (rad)
        iota     = theta[:, 7:8]   # inclination of source (rad)

        derived  = self.compute_derived_parameters(theta)
        M_s  = derived[:, 2:3]     # total mass in seconds
        eta  = derived[:, 3:4]
        delta= derived[:, 4:5]
        STotR= derived[:, 9:10]
        dchi = derived[:, 10:11]
        f_max_prime, f_last = self.checked_band(derived)

        amp_coeffs   = self.get_amp_coeffs(derived)
        phase_coeffs = self.get_phase_coeffs(derived)

        fRING = phase_coeffs['fRING']
        fDAMP = phase_coeffs['fDAMP']

        # Amplitude normalization
        M_m    = M_s * self.C
        dist_m = dist_Mpc * self.Mpc
        amp0   = M_m * M_s / dist_m
        ampNorm = torch.sqrt(2.0*eta / 3.0) * (self.PI ** (-1.0/6.0))
        Amp0   = amp0 * ampNorm

        # Source: IMRPhenomX_TimeShift_22 (LALSimIMRPhenomX_internals.c)
        # here we align the model to the hybrids, for which psi4 peaks
        # 500M before the end of the waveform
        # linb is a parameter-space fit of dphi22(fring22-fdamp22),
        # evaluated on the calibration dataset
        # here we correct the time-alignment of the waveform by first
        # aligning the peak of psi4, and then adding a correction to
        # align the peak of strain instead
        linb_fit_val = IMRPhenomXAS._linb_fit(eta, STotR, dchi, delta)
        psi4val      = IMRPhenomXAS._psi4tostrain_fit(eta, STotR, dchi)

        frefFit   = fRING - fDAMP
        dphi22Ref = (1.0 / eta) * self.dphase(frefFit, phase_coeffs)

        linb = linb_fit_val - dphi22Ref - 2.0*math.pi*(500.0 + psi4val)

        # Calculate phase at reference frequency:
        # phifRef = 2.0*phi0 + LAL_PI_4 + PhenomXPhase(fRef)
        MfRef   = self.f_ref * M_s
        phi22ref = self.phase(MfRef, phase_coeffs)
        phifRef  = -((1.0/eta) * phi22ref + linb * MfRef) + 2.0*phic + 0.25*math.pi

        Mf = self.f * M_s

        # Scale phase by 1/eta
        phi22 = self.phase(Mf, phase_coeffs, loop=True)
        phi_total = phi22 * (1.0/eta) + (linb * Mf + phifRef)

        # Reconstruct waveform: h(f) = A(f) * Exp[I phi(f)]
        A_mf = self.amp(Mf, amp_coeffs, loop=True)
        h22  = Amp0 * Mf.pow(-7.0/6.0) * A_mf * torch.polar(torch.ones_like(phi_total), phi_total)
        # We do not want to generate the waveform at frequencies > f_max
        # (default = 0.3 Mf)
        within = Mf <= f_max_prime * M_s
        if f_last is not None:
            within = within & (self.f <= f_last)
        h22 = torch.where(within, h22, torch.zeros_like(h22))

        # Source: XLALSimInspiralChooseFDWaveform_legacy, case
        # IMRPhenomXAS (LALSimInspiralGeneratorLegacy.c)
        # This is the factor that comes from Y_22star + (-1)^l * Y_2-2
        # without the dependence in inclination, that is included in
        # pfac and cfac
        Ylmfactor = 2.0 * math.sqrt(5.0 / (64.0 * math.pi)) * complex(math.cos(-math.pi), math.sin(-math.pi))
        cfac = torch.cos(iota)
        pfac = 0.5 * (1.0 + cfac * cfac)
        # The factor for hc is the same but opposite sign
        hc = -1j * cfac * h22 * Ylmfactor
        hp = h22 * (pfac * Ylmfactor)

        if not reproduce_lal:
            fcut = self.fM_CUT_XAS / M_s
            _win = _taper_mod.fd_taper(
                f=self.f,
                f_min=self.f[0, 0].item(),
                f_cut=fcut,
                df=self.df,
            )
            hp = hp * _win
            hc = hc * _win

            hp, hc = self.apply_tc(hp, hc, tc)

            hp = hp * self.df
            hc = hc * self.df

        hp, hc = self.pad_missing_frequencies(hp, hc)
        return hp, hc

    def apply_tc(self, hp, hc, tc):
        """
        Shift the waveform so it merges tc seconds into the window.

        tc is measured from the end of the leading padding, so the
        merger lands padding_length_in_s + tc into the padded segment.

        Parameters
        ----------
        hp, hc : torch.Tensor, shape (B, F), complex
            Plus and cross polarisations.
        tc : torch.Tensor, shape (B, 1)
            Time of coalescence in s from the end of the padding.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, F), complex
            Time-shifted polarisations.
        """
        _tc = (tc + self.padding_length_in_s) - self.sample_length_in_s
        hp = torch.polar(torch.abs(hp), torch.angle(hp) - 2 * self.PI * self.f * _tc)
        hc = torch.polar(torch.abs(hc), torch.angle(hc) - 2 * self.PI * self.f * _tc)
        return hp, hc

    def pad_missing_frequencies(self, hp, hc):
        """
        Zero-pad hp and hc from DC to f_min.

        Parameters
        ----------
        hp, hc : torch.Tensor, shape (B, F), complex
            Polarisations on the frequency grid.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex
            Polarisations with n_pad leading zero bins.
        """
        hp_pad = torch.zeros_like(self.hp_buffer)
        hc_pad = torch.zeros_like(self.hc_buffer)
        hp_pad[:, self.n_pad:self.n_pad + hp.shape[1]] = hp
        hc_pad[:, self.n_pad:self.n_pad + hc.shape[1]] = hc
        return hp_pad, hc_pad

    def get_fcut(self, M_s):
        """
        Physical frequency cutoff fM_CUT_XAS / M_s.

        Parameters
        ----------
        M_s : torch.Tensor, shape (B, 1)
            Total mass in seconds.

        Returns
        -------
        torch.Tensor, shape (B, 1)
            Cutoff frequency in Hz.
        """
        return self.fM_CUT_XAS / M_s

    def compute_derived_parameters(self, theta):
        """
        Mass and spin quantities derived from the parameter batch.

        Requires m1 >= m2. Source: IMRPhenomXSetWaveformVariables
        (LALSimIMRPhenomX_internals.c).

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 4+)
            Columns 0-3: m1, m2, chi1z, chi2z (masses in solar masses).

        Returns
        -------
        derived : torch.Tensor, shape (B, 13)
            Columns:
            0-2    m1, m2 and m1 + m2 times GM (seconds)
            3      eta = min(|0.25*(1 - delta**2)|, 0.25)
            4      delta = |m1 - m2| / (m1 + m2)
            5-6    Xa, Xb = 0.5*(1 +- sqrt(1 - 4*eta))
            7      chiEff
            8      chiPNHat
            9      STotR
            10     dchi = chi1 - chi2
            11-12  chi1, chi2 (aligned spins)
        """
        m1   = theta[:, 0:1]
        m2   = theta[:, 1:2]
        chi1 = theta[:, 2:3]
        chi2 = theta[:, 3:4]

        m1_s = m1 * self.GM
        m2_s = m2 * self.GM
        M_s  = (m1 + m2) * self.GM

        # Symmetric mass ratio
        delta = torch.abs((m1 - m2) / (m1 + m2))
        # use fabs to prevent negative sign due to roundoff
        eta = torch.abs(0.25 * (1.0 - delta * delta))
        # If eta > 0.25, e.g. roundoff, then set to 0.25
        eta.clamp_(max=0.25)

        # Convention m1 >= m2 and chi1 is the spin on m1
        delta_eta = torch.sqrt(1.0 - 4.0 * eta)
        Xa    = 0.5 * (1.0 + delta_eta)
        Xb    = 0.5 * (1.0 - delta_eta)

        # Spin parameterisations
        chiEff   = IMRPhenomXAS._chiEff(Xa, Xb, chi1, chi2)
        chiPNHat = IMRPhenomXAS._chiPNHat(eta, chiEff, chi1, chi2)
        STotR    = IMRPhenomXAS._STotR(Xa, Xb, chi1, chi2)
        dchi     = chi1 - chi2

        return torch.cat(
            [m1_s, m2_s, M_s, eta, delta, Xa, Xb,
             chiEff, chiPNHat, STotR, dchi,
             chi1, chi2],
            dim=1,
        )

    # ------------------------------------------------------------------
    # SPIN PARAMETERIZATIONS
    # ------------------------------------------------------------------

    @staticmethod
    def _chiEff(Xa, Xb, chi1, chi2):
        """
        Effective aligned spin parameter, Xa*chi1 + Xb*chi2.

        Source: XLALSimIMRPhenomXchiEff (LALSimIMRPhenomXUtilities.c).

        Parameters
        ----------
        Xa, Xb, chi1, chi2 : torch.Tensor, shape (B, 1)

        Returns
        -------
        chiEff : torch.Tensor, shape (B, 1)
        """
        return Xa * chi1 + Xb * chi2

    @staticmethod
    def _chiPNHat(eta, chiEff, chi1, chi2):
        """
        Normalised PN reduced spin parameter.

        Source: XLALSimIMRPhenomXchiPNHat (LALSimIMRPhenomXUtilities.c).

        Parameters
        ----------
        eta, chiEff, chi1, chi2 : torch.Tensor, shape (B, 1)

        Returns
        -------
        chiPNHat : torch.Tensor, shape (B, 1)
        """
        num = chiEff - (38.0 / 113.0) * eta * (chi1 + chi2)
        den = 1.0 - (76.0 / 113.0) * eta
        return num / den

    @staticmethod
    def _STotR(Xa, Xb, chi1, chi2):
        """
        Total spin normalised to [-1,1].

        Source: XLALSimIMRPhenomXSTotR (LALSimIMRPhenomXUtilities.c).

        Parameters
        ----------
        Xa, Xb, chi1, chi2 : torch.Tensor, shape (B, 1)

        Returns
        -------
        STotR : torch.Tensor, shape (B, 1)
        """
        Xa2 = Xa * Xa
        Xb2 = Xb * Xb
        return (Xa2 * chi1 + Xb2 * chi2) / (Xa2 + Xb2)

    # ------------------------------------------------------------------
    # FINAL STATE
    # ------------------------------------------------------------------

    @staticmethod
    def final_mass_2017(eta, S, dchi):
        """
        Final Mass = 1 - Energy Radiated.

        Source: XLALSimIMRPhenomXFinalMass2017
        (LALSimIMRPhenomXUtilities.c).

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        S : torch.Tensor, shape (B, 1)
            STotR.
        dchi : torch.Tensor, shape (B, 1)
            chi1 - chi2.

        Returns
        -------
        Mfinal : torch.Tensor, shape (B, 1)
            Remnant mass as a fraction of m1 + m2.
        """
        delta = torch.sqrt(1.0 - 4.0 * eta)
        eta2  = eta  * eta
        eta3  = eta2 * eta
        eta4  = eta3 * eta
        S2    = S    * S
        S3    = S2   * S
        dchi2 = dchi * dchi

        noSpin = (
            0.057190958417936644*eta
            + 0.5609904135313374*eta2
            - 0.84667563764404*eta3
            + 3.145145224278187*eta4
        )

        # Because of the way this is written, we need to subtract the
        # noSpin term
        eqSpin = (
            noSpin
            * (1.0
               + (-0.13084389181783257 - 1.1387311580238488*eta + 5.49074464410971*eta2)*S
               + (-0.17762802148331427 + 2.176667900182948*eta2)*S2
               + (-0.6320191645391563 + 4.952698546796005*eta - 10.023747993978121*eta2)*S3)
            / (1.0 + (-0.9919475346968611 + 0.367620218664352*eta + 4.274567337924067*eta2)*S)
            - noSpin
        )

        uneqSpin = (
            - 0.09803730445895877  * dchi * delta * (1.0 - 3.2283713377939134*eta)  * eta2
            + 0.01118530335431078  * dchi2 * eta3
            - 0.01978238971523653  * dchi * delta * (1.0 - 4.91667749015812*eta) * eta * S
        )

        # Mfinal = 1 - Erad, assuming that M = m1 + m2 = 1
        return 1.0 - (noSpin + eqSpin + uneqSpin)

    @staticmethod
    def final_spin_2017(eta, S, dchi):
        """
        Final Dimensionless Spin.

        Source: XLALSimIMRPhenomXFinalSpin2017
        (LALSimIMRPhenomXUtilities.c).

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        S : torch.Tensor, shape (B, 1)
            STotR.
        dchi : torch.Tensor, shape (B, 1)
            chi1 - chi2.

        Returns
        -------
        afinal : torch.Tensor, shape (B, 1)
            Dimensionless remnant spin.
        """
        delta = torch.sqrt(1.0 - 4.0 * eta)
        m1    = 0.5 * (1.0 + delta)
        m2    = 0.5 * (1.0 - delta)
        m1Sq  = m1 * m1
        m2Sq  = m2 * m2
        eta2  = eta  * eta
        eta3  = eta2 * eta
        S2    = S    * S
        S3    = S2   * S
        dchi2 = dchi * dchi

        noSpin = (
            3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta3
        ) / (1.0 + 7.2388440419467335*eta)

        eqSpin = (m1Sq + m2Sq) * S + (
            ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta3)*S
             + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta3)*S2
             + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta3)*S3)
            / (1.0 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta3)*S)
        )

        uneqSpin = (
            0.3223660562764661   * dchi * delta * (1.0 + 9.332575956437443*eta)  * eta2
            - 0.059808322561702126 * dchi2 * eta3
            + 2.3170397514509933  * dchi * delta * (1.0 - 3.2624649875884852*eta) * eta3 * S
        )

        return noSpin + eqSpin + uneqSpin

    @staticmethod
    def get_fRD_fdamp(af, Mfinal):
        """
        Ringdown and damping frequency of final BH.

        Source: evaluate_QNMfit_fring22, evaluate_QNMfit_fdamp22
        (LALSimIMRPhenomX_qnm.c, QNMfits == 1), divided by Mfinal as in
        IMRPhenomXSetWaveformVariables (LALSimIMRPhenomX_internals.c).

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Final dimensionless spin in [-1, 1].
        Mfinal : torch.Tensor, shape (B, 1)
            Remnant mass fraction from final_mass_2017.

        Returns
        -------
        fRING, fDAMP : torch.Tensor, shape (B, 1)
            Dimensionless frequencies; divide by M_s to get Hz.
        """
        a  = af
        a2 = a  * a
        a3 = a2 * a
        a4 = a2 * a2
        a5 = a3 * a2
        a6 = a3 * a3
        a7 = a4 * a3

        # evaluate fit QNMData_fring_22
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

        # evaluate fit QNMData_fdamp_22
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

        # Ringdown and damping frequency of final BH
        return fring22 / Mfinal, fdamp22 / Mfinal

    # ------------------------------------------------------------------
    # MECO, ISCO, etc
    # ------------------------------------------------------------------

    @staticmethod
    def fMECO(eta, chiPNHat, dchi):
        """
        Fit to hybrid minimum energy circular orbit (MECO) function.

        Uses 3.5PN hybridised with test-particle limit.
        Source: XLALSimIMRPhenomXfMECO (LALSimIMRPhenomXUtilities.c).

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        chiPNHat : torch.Tensor, shape (B, 1)
            Normalised PN reduced spin parameter.
        dchi : torch.Tensor, shape (B, 1)
            chi1 - chi2.

        Returns
        -------
        fMECO : torch.Tensor, shape (B, 1)
            Dimensionless MECO frequency.
        """
        delta = torch.sqrt(1.0 - 4.0 * eta)
        eta2  = eta  * eta
        eta3  = eta2 * eta
        eta4  = eta3 * eta

        S  = chiPNHat
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
        Innermost stable circular orbit (ISCO) frequency.

        Source: XLALSimIMRPhenomXfISCO (LALSimIMRPhenomXUtilities.c).

        Parameters
        ----------
        afinal : torch.Tensor, shape (B, 1)
            Final dimensionless spin.

        Returns
        -------
        fISCO : torch.Tensor, shape (B, 1)
            OmegaISCO / pi.
        """
        a  = afinal
        a2 = a * a
        Z1 = 1.0 + (1.0 - a2).pow(1.0 / 3.0) * ((1.0 + a).pow(1.0 / 3.0) + (1.0 - a).pow(1.0 / 3.0))
        # Finite precission may give Z1>3, but this can not happen.
        Z1 = torch.clamp(Z1, max=3.0)
        Z2 = torch.sqrt(3.0*a2 + Z1*Z1)
        rISCO = 3.0 + Z2 - torch.sign(a) * torch.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2))

        OmegaISCO = 1.0 / (rISCO.pow(1.5) + a)

        return OmegaISCO / math.pi

    def get_amp_coeffs(self, derived, afinal=None):
        """
        Amplitude coefficients for amplitude versions 103, 104 and 103.

        Versions are inspiral, intermediate and ringdown in that order.
        Source: IMRPhenomXGetAmplitudeCoefficients
        (LALSimIMRPhenomX_internals.c).

        Parameters
        ----------
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        afinal : torch.Tensor, shape (B, 1), optional
            Final spin override. The precessing model passes the
            precessing final spin while the final mass stays
            aligned-spin, as in IMRPhenomX_SetPrecessingRemnantParams
            (LALSimIMRPhenomX_precession.c).

        Returns
        -------
        ac : dict of torch.Tensor, shape (B, 1)
            pnTwoThirds, pnThreeThirds, pnFourThirds, pnFiveThirds,
            pnSixThirds, rho1, rho2, rho3, fAmpMatchIN, delta0 to
            delta4, fAmpRDMin, fRING, gammaR, gammaD2, gammaD13.
        """
        eta      = derived[:, 3:4]
        delta    = derived[:, 4:5]
        chi1     = derived[:, 11:12]
        chi2     = derived[:, 12:13]
        STotR    = derived[:, 9:10]
        chiPNHat = derived[:, 8:9]
        dchi     = derived[:, 10:11]

        eta2  = eta  * eta
        eta3  = eta2 * eta
        chi12 = chi1 * chi1
        chi13 = chi12 * chi1
        chi22 = chi2 * chi2

        pi    = math.pi

        # Final Mass and Spin
        Mfinal = IMRPhenomXAS.final_mass_2017(eta, STotR, dchi)
        afinal_aligned = IMRPhenomXAS.final_spin_2017(eta, STotR, dchi)
        if afinal is None:
            afinal = afinal_aligned
        # Ringdown and damping frequency of final BH
        fRING, fDAMP = IMRPhenomXAS.get_fRD_fdamp(afinal, Mfinal)

        # Fit to the hybrid minimum energy circular orbit (MECO)
        fmeco = IMRPhenomXAS.fMECO(eta, chiPNHat, dchi)
        # Innermost stable circular orbit (ISCO)
        # Source: IMRPhenomX_SetPrecessingRemnantParams
        # (LALSimIMRPhenomX_precession.c)
        fisco = IMRPhenomXAS.fISCO(afinal_aligned)

        # Phenomenological ringdown coefficients
        S  = STotR;  S2 = S*S

        gamma2 = (
            (0.8312293675316895 + 7.480371544268765*eta - 18.256121237800397*eta2)
            / (1.0 + 10.915453595496611*eta - 30.578409433912874*eta2)
            + (S*(0.5869408584532747 + eta*(-0.1467158405070222 - 2.8489481072076472*S)
               + 0.031852563636196894*S + eta2*(0.25295441250444334 + 4.6849496672664594*S)))
            / (3.8775263105069953 - 3.41755361841226*S + S2)
            - 0.00548054788508203*dchi*delta*eta
        )

        gamma3 = (
            (1.3666000000000007 - 4.091333144596439*eta + 2.109081209912545*eta2 - 4.222259944408823*eta3)
            / (1.0 - 2.7440263888207594*eta)
            + (0.07179105336478316 + eta2*(2.331724812782498 - 0.6330998412809531*S)
               + eta*(-0.8752427297525086 + 0.4168560229353532*S) - 0.05633734476062242*S)*S
        )

        # Get peak ringdown frequency:
        # Abs[fring + fdamp * gamma3 * (Sqrt[1 - gamma2^2] - 1)/gamma2 ]
        # If gamma2 > 1, then the square root term can become imaginary.
        # Set this term to zero.
        fAmpRDMin_normal = torch.abs(fRING + fDAMP * gamma3 * (torch.sqrt(1.0 - gamma2*gamma2) - 1.0) / gamma2)
        fAmpRDMin_large  = torch.abs(fRING - fDAMP * gamma3 / gamma2)
        fAmpRDMin = torch.where(gamma2 <= 1.0, fAmpRDMin_normal, fAmpRDMin_large)

        # Value of v1RD at fAmpRDMin is used to calcualte gamma1 \propto
        # the amplitude of the deformed Lorentzian
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

        # Solving linear equation: ansatzRD(F1) == v1
        # Pre-cache these constants for the ringdown amplitude
        fDMg3  = fDAMP * gamma3
        gammaR  = gamma2 / fDMg3
        gammaD2 = fDMg3 * fDMg3
        dfr_rd  = fAmpRDMin - fRING
        gamma1  = (v1RD / fDMg3) * (dfr_rd*dfr_rd + gammaD2) * torch.exp(dfr_rd * gammaR)
        gammaD13 = fDMg3 * gamma1

        # Inspiral-Intermediate matching frequency
        fAmpMatchIN = fmeco + 0.25 * (fisco - fmeco)

        # TaylorF2 PN Amplitude Coefficients from usual PN re-expansion
        # of Hlm's
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

        # Generate Pseudo-PN Coefficients for Amplitude.
        S_ins = chiPNHat;  S2_ins = S_ins*S_ins;  S3_ins = S2_ins*S_ins
        eta4  = eta3 * eta

        V2 = (
            (-0.015178276424448592 - 0.06098548699809163*eta + 0.4845148547154606*eta2)
            / (1.0 + 0.09799277215675059*eta)
            + ((0.02300153747158323 + 0.10495263104245876*eta2)*S_ins
               + (0.04834642258922544 - 0.14189350657140673*eta)*eta*S3_ins
               + (0.01761591799745109 - 0.14404522791467844*eta2)*S2_ins)
            / (1.0 - 0.7340448493183307*S_ins)
            + dchi*delta*eta4*(0.0018724905795891192 + 34.90874132485147*eta)
        )

        V3 = (
            (-0.058572000924124644 - 1.1970535595488723*eta + 8.4630293045015*eta2)
            / (1.0 + 15.430818840453686*eta)
            + ((-0.08746408292050666 + eta*(-0.20646621646484237 - 0.21291764491897636*S_ins)
                + eta2*(0.788717372588848 + 0.8282888482429105*S_ins)
                - 0.018924013869130434*S_ins)*S_ins)
            / (-1.332123330797879 + S_ins)
            + dchi*delta*eta4*(0.004389995099201855 + 105.84553997647659*eta)
        )

        V4 = (
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

        F1 = 0.50 * fAmpMatchIN
        F2 = 0.75 * fAmpMatchIN
        F3 = 1.00 * fAmpMatchIN

        # Using the collocation points and their values, we solve a
        # linear system of equations to recover the pseudo-PN
        # coefficients
        # Source: IMRPhenomX_Inspiral_Amp_22_rho1, _rho2, _rho3
        # (LALSimIMRPhenomX_inspiral.c)
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

        # Now we can reconstruct the intermediate region, which depends
        # on the derivative of the inspiral and ringdown fits at the
        # boundaries
        Fma = fAmpMatchIN
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

        # Calculate derivative of amplitude on boundary points
        # Source: IMRPhenomX_Inspiral_Amp_22_DAnsatz
        # (LALSimIMRPhenomX_inspiral.c)
        Fma_p1o3 = Fma.pow(1.0/3.0)
        Fma_p2o3 = Fma.pow(2.0/3.0)
        Fma_p2   = Fma * Fma

        d_inspF1 = (
            ((chi2*(81.0 - 81.0*delta - 44.0*eta) + chi1*(81.0*(1.0 + delta) - 44.0*eta))*pi) / 48.0
            + ((-969.0 + 1804.0*eta)*p2o3) / (1008.0 * Fma_p1o3)
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

        # D[ f^{7/6} Ah^{-1}[f] , f] = ( (7/6) * f^{1/6} / Ah[f] )
        #     - ( f^{7/6} Ah'[f] ) / (Ah[f]^2)
        # where d1 = Ah'[F1], inspF1 = Ah[F1] etc.
        d1 =(7.0/6.0) * Fma.pow(1.0/6.0) / inspF1 - Fma.pow(7.0/6.0) * d_inspF1 / (inspF1*inspF1)

        # Phenomenological Ringdown Amplitude Ansatz.
        F4 = fAmpRDMin
        dfr4 = F4 - fRING
        rdF4  = torch.exp(-dfr4 * gammaR) * gammaD13 / (dfr4*dfr4 + gammaD2)

        # Derivative (with respect to f) of Phenomenological Ringdown
        # Amplitude Ansatz.
        d_rdF4 = (
            -torch.exp(-gamma2 * dfr4 / fDMg3) * gamma1
            * (dfr4*dfr4*gamma2 + 2.0*fDAMP*dfr4*gamma3 + fDAMP*fDAMP*gamma2*gamma3*gamma3)
            / ((dfr4*dfr4 + gammaD2) * (dfr4*dfr4 + gammaD2))
        )

        d4 = (7.0/6.0) * F4.pow(1.0/6.0) / rdF4 - F4.pow(7.0/6.0) * d_rdF4 / (rdF4*rdF4)

        # Pass inverse of points to reconstruction function...
        # Use a 4th order polynomial in intermediate - good
        # extrapolation, recommended default fit
        F1_int = fAmpMatchIN
        F2_int = F1_int + 0.5*(F4 - F1_int)

        V1_int = F1_int.pow(7.0/6.0) / inspF1
        V4_int = F4.pow(7.0/6.0) / rdF4

        # Value of intermediate amplitude collocation vA point at f_2
        S3   = S2*S
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
        V2_int = 1.0 / vA

        # Reconstruct the phenomenological coefficients for the
        # intermediate ansatz:
        # Source: IMRPhenomX_Intermediate_Amp_22_delta0 to _delta4
        # (LALSimIMRPhenomX_intermediate.c)
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
            'pnTwoThirds':   pnTwoThirds,
            'pnThreeThirds': pnThreeThirds,
            'pnFourThirds':  pnFourThirds,
            'pnFiveThirds':  pnFiveThirds,
            'pnSixThirds':   pnSixThirds,
            'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
            'fAmpMatchIN': fAmpMatchIN,
            'delta0': delta0, 'delta1': delta1, 'delta2': delta2,
            'delta3': delta3, 'delta4': delta4,
            'fAmpRDMin': fAmpRDMin,
            'fRING':   fRING,
            'gammaR':  gammaR,   # gamma2 / (fDAMP * gamma3)
            'gammaD2': gammaD2,  # (fDAMP * gamma3)^2
            'gammaD13': gammaD13, # fDAMP * gamma1 * gamma3
        }

    def amp(self, f_Ms, amp_coeffs, loop=False):
        """
        Normalised (2,2) amplitude A(Mf) over the grid.

        Inspiral for Mf <= fAmpMatchIN, intermediate up to fAmpRDMin,
        ringdown above. Source: IMRPhenomX_Amplitude_22
        (LALSimIMRPhenomX_internals.c).

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, F)
            Dimensionless frequency Mf.
        amp_coeffs : dict
            Output of get_amp_coeffs.
        loop : bool
            Region boundaries of the waveform loop instead. Source:
            IMRPhenomXAS_GenerateFD (LALSimIMRPhenomX.c).

        Returns
        -------
        amp : torch.Tensor, shape (B, F)
            Normalised amplitude A(Mf).
        """
        ac = amp_coeffs

        # TaylorF2 PN Amplitude + pseudo-PN coefficients
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

        # Return a polynomial embedded in the background from the merger
        rho_int = (
            ac['delta0']
            + ac['delta1'] * Mf
            + ac['delta2'] * Mf * Mf
            + ac['delta3'] * Mf * Mf * Mf
            + ac['delta4'] * Mf * Mf * Mf * Mf
        )
        A_int = Mf.pow(7.0/6.0) / rho_int

        # Phenomenological Ringdown Amplitude Ansatz.
        dfr   = Mf - ac['fRING']
        A_rd  = (
            torch.exp(-dfr * ac['gammaR'])
            * ac['gammaD13']
            / (dfr*dfr + ac['gammaD2'])
        )

        fIN = ac['fAmpMatchIN']
        fRD = ac['fAmpRDMin']
        if loop:
            return torch.where(Mf < fIN, A_ins,
                   torch.where(Mf > fRD, A_rd, A_int))
        return torch.where(Mf <= fIN, A_ins,
               torch.where(Mf <= fRD, A_int, A_rd))

    def get_phase_coeffs(self, derived, afinal=None):
        """
        Phase coefficients for phase versions 104 (inspiral) and 105.

        Source: IMRPhenomXGetPhaseCoefficients and
        IMRPhenomX_Phase_22_ConnectionCoefficients
        (LALSimIMRPhenomX_internals.c).

        Parameters
        ----------
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        afinal : torch.Tensor, shape (B, 1), optional
            Final spin override. The precessing model passes the
            precessing final spin while the final mass stays
            aligned-spin, as in IMRPhenomX_SetPrecessingRemnantParams
            (LALSimIMRPhenomX_precession.c).

        Returns
        -------
        pc : dict
            Tensors of shape (B, 1), except the floats phiNorm and
            dphase0: phi0 to phi9L, sigma1 to sigma5, a0 to a3, dphi0 to
            dphi8L, b0 to b4, c0, c1, c2, c4, cL, cLovfda, c4ov3, fRING,
            fDAMP, fPhaseMatchIN, fPhaseMatchIM, C1Int, C2Int, C1MRD,
            C2MRD.
        """
        eta      = derived[:, 3:4]
        delta    = derived[:, 4:5]
        chi1     = derived[:, 11:12]
        chi2     = derived[:, 12:13]
        STotR    = derived[:, 9:10]
        chiPNHat = derived[:, 8:9]
        dchi     = derived[:, 10:11]

        eta2  = eta  * eta
        eta3  = eta2 * eta
        eta4  = eta3 * eta

        # For convenience, define some variables here
        chi1L  = chi1
        chi2L  = chi2
        chi1L2 = chi1L * chi1L
        chi1L3 = chi1L2 * chi1L
        chi2L2 = chi2L * chi2L
        chi2L3 = chi2L2 * chi2L
        chi1L2L = chi1L * chi2L

        pi    = math.pi
        GAMMA = 0.5772156649015328606   # LAL_GAMMA
        log2  = math.log(2.0)
        logpi = math.log(pi)

        pi2o3 = pi ** (2.0/3.0)
        pi4o3 = pi ** (4.0/3.0)
        pi5o3 = pi ** (5.0/3.0)
        pi7o3 = pi ** (7.0/3.0)
        pi2   = pi * pi
        pi8o3 = pi ** (8.0/3.0)

        # Phase normalization
        phiNorm  = -(3.0 / (128.0 * pi5o3))
        dphase0  = 5.0 / (128.0 * pi5o3)

        # Final Mass and Spin
        Mfinal = IMRPhenomXAS.final_mass_2017(eta, STotR, dchi)
        afinal_aligned = IMRPhenomXAS.final_spin_2017(eta, STotR, dchi)
        if afinal is None:
            afinal = afinal_aligned
        # Ringdown and damping frequency of final BH
        fRING, fDAMP = IMRPhenomXAS.get_fRD_fdamp(afinal, Mfinal)
        # Fit to the hybrid minimum energy circular orbit (MECO)
        fmeco  = IMRPhenomXAS.fMECO(eta, chiPNHat, dchi)
        # Innermost stable circular orbit (ISCO)
        # Source: IMRPhenomX_SetPrecessingRemnantParams
        # (LALSimIMRPhenomX_precession.c)
        fisco  = IMRPhenomXAS.fISCO(afinal_aligned)

        # Matching regions
        fIMmatch = 0.6 * (0.5 * fRING + fisco)
        fINmatch = fmeco  # This is the MECO frequency
        deltaf   = (fIMmatch - fINmatch) * 0.03

        # Transition frequency is just below the MECO frequency and just
        # above the RD fitting region
        fPhaseMatchIN  = fINmatch - deltaf
        fPhaseMatchIM  = fIMmatch + 0.5 * deltaf
        fPhaseInsMin   = torch.full_like(fRING, 0.0026)
        fPhaseInsMax   = 1.020 * fmeco
        fPhaseRDMin    = fIMmatch
        fPhaseRDMax    = fRING + 1.25 * fDAMP

        # **** TaylorF2 PN Coefficients: Phase ****
        # - These are the PN coefficients normalised by:
        #   3 / (128 * eta * [pi M f]^{5/3} ).
        # - We add in powers of (M f)^{N/3} later but add powers of
        #   pi^{N/3} here

        # Newtonian
        phi0 = torch.ones_like(eta)

        # ~~ 0.5 PN ~~
        phi1 = torch.zeros_like(eta)

        # ~~ 1.0 PN ~~
        phi2 = (3715.0/756.0 + (55.0 * eta) / 9.0) * pi2o3

        # ~~ 1.5 PN ~~
        phi3NS = -16.0 * pi2
        phi3S  = (
            (113.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
             - 76.0*(chi1L + chi2L)*eta) / 6.0
        ) * pi
        phi3 = phi3NS + phi3S

        # ~~ 2.0 PN ~~
        phi4NS = (
            15293365.0/508032.0 + (27145.0 * eta)/504.0 + (3085.0 * eta2)/72.0
        ) * pi4o3
        phi4S  = (
            -5.0*(81.0*chi1L2*(1.0 + delta - 2.0*eta)
                  + 316.0*chi1L2L*eta
                  - 81.0*chi2L2*(-1.0 + delta + 2.0*eta)) / 16.0
        ) * pi4o3
        phi4 = phi4NS + phi4S

        # ~~ 2.5 PN ~~
        phi5 = torch.zeros_like(eta)

        # ~~ 2.5 PN, Log Term ~~
        phi5LNS = (5.0*(46374.0 - 6552.0*eta)*pi / 4536.0) * pi5o3
        phi5LS  = (
            (-732985.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
             - 560.0*(-1213.0*(chi1L + chi2L) + 63.0*(chi1L - chi2L)*delta)*eta
             + 85680.0*(chi1L + chi2L)*eta2) / 4536.0
        ) * pi5o3
        phi5L = phi5LNS + phi5LS

        # ~~ 3.0 PN ~~
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

        # ~~ 3.0 PN, Log Term ~~
        phi6LNS = (-6848.0/63.0) * pi2
        phi6L   = phi6LNS

        # ~~ 3.5 PN ~~
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

        # ~~ 4.0 PN ~~
        spin_combo_8 = (
            1263141.0*(chi1L + chi2L + chi1L*delta - chi2L*delta)
            - 2.0*(794075.0*(chi1L + chi2L) + 178533.0*(chi1L - chi2L)*delta)*eta
            + 94344.0*(chi1L + chi2L)*eta2
        )
        # 4.0 PN, Spin-Orbit
        phi8S  = (
            -5.0 * spin_combo_8 * pi * (logpi - 1.0) / 9072.0
        ) * pi8o3
        # 4.0 PN, log term, Spin-Orbit
        phi8L_S = (
            -5.0 * spin_combo_8 * pi / 9072.0
        ) * pi8o3
        phi8   = phi8S
        phi8L  = phi8L_S

        # ~~ 4.5 PN ~~
        phi9  = torch.zeros_like(eta)
        phi9L = torch.zeros_like(eta)

        # TaylorF2 PN Coefficients: Normalized Phase Derivative
        dphi0  = phi0
        dphi1  = (4.0/5.0) * phi1
        dphi2  = (3.0/5.0) * phi2
        dphi3  = (2.0/5.0) * phi3
        dphi4  = (1.0/5.0) * phi4
        dphi5  = -(3.0/5.0) * phi5L
        dphi6  = -(1.0/5.0) * phi6 - (3.0/5.0) * phi6L
        dphi6L = -(1.0/5.0) * phi6L
        dphi7  = -(2.0/5.0) * phi7
        dphi8  = -(3.0/5.0) * phi8 - (3.0/5.0) * phi8L
        dphi8L = -(3.0/5.0) * phi8L

        # Source: IMRPhenomX_Inspiral_Phase_22_v3, _d13, _d23, _d43
        # (LALSimIMRPhenomX_inspiral.c)
        # Effective spin parameterization used = chiPNHat
        S    = chiPNHat
        S2   = S  * S
        S3   = S2 * S
        S4   = S3 * S
        dchi2 = dchi * dchi

        # Value of phase collocation point at v_3.
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

        # Value of phase collocation point for d13 = v1 - v3.
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

        # Value of phase collocation point for d23 = v2 - v3.
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

        # Value of phase collocation point for d43 = v4 - v3.
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

        # Calculate the value of the collocation points at GC nodes via:
        # v_i = d_i3 + v3
        v1_ins = d13_ins + v3_ins
        v2_ins = d23_ins + v3_ins
        v4_ins = d43_ins + v3_ins

        # By default all models implemented use the following GC points.
        dx_ins = fPhaseInsMax - fPhaseInsMin
        f1_ins = fPhaseInsMin
        f2_ins = fPhaseInsMin + 0.25 * dx_ins
        f3_ins = fPhaseInsMin + 0.75 * dx_ins
        f4_ins = fPhaseInsMax

        def _ins_row(fi):
            fi13 = fi.pow(1.0/3.0)
            fi23 = fi13 * fi13
            return torch.cat([torch.ones_like(fi), fi13, fi23, fi], dim=-1)

        A_ins = torch.stack([
            _ins_row(f1_ins), _ins_row(f2_ins),
            _ins_row(f3_ins), _ins_row(f4_ins),
        ], dim=-2)

        b_ins = torch.stack(
            [v1_ins, v2_ins, v3_ins, v4_ins], dim=-2
        )

        # We now solve the system A x = b via an LU decomposition
        x_ins = torch.linalg.solve(A_ins, b_ins)
        a0 = x_ins[:, 0, :]
        a1 = x_ins[:, 1, :]
        a2 = x_ins[:, 2, :]
        a3 = x_ins[:, 3, :]

        # The pseudo-PN coefficients are normalized such that:
        # (dphase0 / eta) * f^{8/3} * a_j
        # So we must re-scale these terms by an extra factor of f^{-8/3}
        # in the PN phasing
        sigma1 = (-5.0/3.0) * a0
        sigma2 = (-5.0/4.0) * a1
        sigma3 = (-5.0/5.0) * a2
        sigma4 = (-5.0/6.0) * a3
        sigma5 = torch.zeros_like(a0)

        # Source: IMRPhenomX_Ringdown_Phase_22_v4, _d12, _d24, _d34,
        # _d54 (LALSimIMRPhenomX_ringdown.c)
        # Effective Spin Used: STotR.
        S   = STotR
        S2  = S  * S
        S3  = S2 * S
        S4  = S3 * S
        S5  = S4 * S

        eta5 = eta4 * eta

        # Collocation point for ringdown phase evaluated at
        # f_ring = f_4.
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

        # Difference between collocation points 1 and 2 (d12 = v1 - v2).
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

        # Difference between collocation points 2 and 4 (d24 = v2 - v4).
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

        # Difference between collocation points 3 and 4 (d34 = v3 - v4).
        d34_rd = (
            (eta*(-8.506898502692536 + 13.936621412517798*eta)) / (-0.40919671232073945 + eta)
            + (eta*(1.7280582989361533*S + 18.41570325463385*S3 - 13.743271480938104*S4)
               + eta2*(73.8367329022058*S - 95.57802408341716*S3 + 215.78111099820157*S4)
               + 0.046849371468156265*S2
               + eta3*S*(-27.976989112929353 + 6.404060932334562*S - 633.1966645925428*S3 + 109.04824706217418*S2))
            / (1.0 - 0.6862449113932192*S)
            + 641.8965762829259*dchi*delta*eta5
        )

        # Difference between collocation points 5 and 4 (d54 = v5 - v4).
        d54_rd = (
            (eta*(7.05731400277692 + 22.455288821807095*eta + 119.43820622871043*eta2))
            / (0.26026709603623255 + eta)
            + (eta2*(134.88158268621922 - 56.05992404859163*S)*S
               + eta*S*(-7.9407123129681425 + 9.486783128047414*S)
               + eta3*S*(-316.26970506215554 + 90.31815139272628*S))
            / (1.0 - 0.7162058321905909*S)
            + 43.82713604567481*dchi*delta*eta3
        )

        # v_j = d_{j4} + v4
        v2_rd = d24_rd + v4_rd
        v3_rd = d34_rd + v4_rd
        v5_rd = d54_rd + v4_rd
        v1_rd = d12_rd + v2_rd    # v1 = d12  + v2

        # Ringdown phase collocation points
        half_inv_sqrt2 = 1.0 / (2.0 * math.sqrt(2.0))
        gp5 = [0.0, 0.5 - half_inv_sqrt2, 0.5, 0.5 + half_inv_sqrt2, 1.0]
        dx_rd = fPhaseRDMax - fPhaseRDMin
        fi_rd = [fPhaseRDMin + gp * dx_rd for gp in gp5]
        # Collocation point 4 is set to the ringdown frequency ~ dip in
        # Lorentzian
        fi_rd[3] = fRING

        # ansatzRD(f) = a_0 + a_1 f^(-1/3) + a_2 f^(-2) + a_3 f^(-3)
        #     + a_4 f^(-4) + ( aRD ) / ( (f_damp^2 + (f - f_ring)^2 ) )
        # Canonical ansatz sets a_3 to 0.
        def _rd_row(fi):
            inv_fi   = 1.0 / fi
            fi_m1o3  = fi.pow(-1.0/3.0)
            fi_m2    = inv_fi * inv_fi
            fi_m4    = fi_m2 * fi_m2
            lorentz  = -dphase0 / (fDAMP*fDAMP + (fi - fRING)*(fi - fRING))
            return torch.cat([torch.ones_like(fi), fi_m1o3, fi_m2, fi_m4, lorentz], dim=-1)

        A_rd = torch.stack(
            [_rd_row(fi_rd[i]) for i in range(5)], dim=-2
        )

        b_rd = torch.stack(
            [v1_rd, v2_rd, v3_rd, v4_rd, v5_rd], dim=-2
        )

        # We now solve the system A x = b via an LU decomposition
        x_rd = torch.linalg.solve(A_rd, b_rd)
        c0   = x_rd[:, 0, :]
        c1   = x_rd[:, 1, :]
        c2   = x_rd[:, 2, :]
        c4   = x_rd[:, 3, :]
        cRD  = x_rd[:, 4, :]
        cL   = -dphase0 * cRD            # cL = - a_{RD} * dphase0
        cLovfda = cL / fDAMP
        c4ov3   = c4 / 3.0

        # This is used in intermediate phase reconstruction.
        phaseRD = v1_rd

        # Calculate phase at fmatchIN. This will be used as the
        # collocation point for the intermediate fit. In practice, the
        # transition point is just below the MECO frequency.
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
            # Add pseudo-PN Coefficient
            + a0 * fIN.pow(8.0/3.0)
            + a1 * fIN.pow(3.0)
            + a2 * fIN.pow(10.0/3.0)
            + a3 * fIN.pow(11.0/3.0)
        ) * fIN.pow(-8.0/3.0) * dphase0

        # Canonical intermediate model using 5 collocation points
        # Source: IMRPhenomX_Intermediate_Phase_22_v2mRDv4, _v3mRDv4,
        # _v2, _d43 (LALSimIMRPhenomX_intermediate.c)
        # Effective spin parameterization used = StotR
        S   = STotR
        S2  = S * S
        eta6 = eta3 * eta3

        # v2IM - v4RD. Using v4RD helps condition the fits with v4RD
        # being very a robust fit.
        v2mRDv4 = (
            (eta*(0.9951733419499662 + 101.21991715215253*eta + 632.4731389009143*eta2))
            / (0.00016803066316882238 + 0.11412314719189287*eta + 1.8413983770369362*eta2 + eta3)
            + (S*(18.694178521101332 + 16.89845522539974*S + 4941.31613710257*eta2*S
                  + eta*(-697.6773920613674 - 147.53381808989846*S2) + 0.3612417066833153*S2
                  + eta3*(3531.552143264721 - 14302.70838220423*S + 178.85850322465944*S2)))
            / (2.965640445745779 - 2.7706595614504725*S + S2)
            + dchi*delta*eta2*(356.74395864902294 + 1693.326644293169*eta2*S)
        )

        # v3IM - v4RD. Using v4RD helps condition the fits with v4RD
        # being very a robust fit.
        v3mRDv4 = (
            (eta*(-5.126358906504587 - 227.46830225846668*eta + 688.3609087244353*eta2
                  - 751.4184178636324*eta3))
            / (-0.004551938711031158 - 0.7811680872741462*eta + eta2)
            + (S*(0.1549280856660919 - 0.9539250460041732*S - 539.4071941841604*eta2*S
                  + eta*(73.79645135116367 - 8.13494176717772*S2) - 2.84311102369862*S2
                  + eta3*(-936.3740515136005 + 1862.9097047992134*S + 224.77581754671272*S2)))
            / (-1.5308507364054487 + S)
            + 2993.3598520496153*dchi*delta*eta6
        )

        # Direct fit to the collocation point at F2. We will take a
        # weighted average of the direct and conditioned fit.
        v2IM = (
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

        # d43 = v4 - v3.
        d43_int = (
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

        # The first and last collocation points for the intermediate
        # region are set from the inspiral fit and ringdown respectively
        v1_int = phaseIN
        # Take a weighted average for these points. Helps condition the
        # fit.
        v2_int = 0.75 * (v2mRDv4 + v4_rd) + 0.25 * v2IM
        v3_int = v3mRDv4 + v4_rd
        # Collocation points: v4 = d43 + v3
        v4_int = d43_int + v3_int
        v5_int = phaseRD

        # Using 5 collocation points in intermediate region
        dx_int = fPhaseMatchIM - fPhaseMatchIN
        f1_int = fPhaseMatchIN
        f2_int = fPhaseMatchIN + (0.5 - half_inv_sqrt2) * dx_int
        f3_int = fPhaseMatchIN + 0.5 * dx_int
        f4_int = fPhaseMatchIN + (0.5 + half_inv_sqrt2) * dx_int
        f5_int = fPhaseMatchIM

        # This is the canonical intermediate ansatz:
        # a_0 + a_1 ft^(-1) + a_2 ft^(-2) + a_3 ft^(-3) + a4 ft^(-4)
        #     + (4 * a_RD) / ( (2 * f_fdamp)^2 + (f - f_ring)^2 )
        # ft = (f / f_ring)
        def _int_row(fi):
            rt = fRING / fi
            rt2 = rt * rt
            rt3 = rt * rt2
            rt4 = rt2 * rt2
            return torch.cat([torch.ones_like(fi), rt, rt2, rt3, rt4], dim=-1)

        def _int_lorentz(fi):
            return 4.0 * cL / (4.0 * fDAMP*fDAMP + (fi - fRING)*(fi - fRING))

        A_int = torch.stack([
            _int_row(f1_int), _int_row(f2_int), _int_row(f3_int),
            _int_row(f4_int), _int_row(f5_int),
        ], dim=-2)

        b_int = torch.stack([
            v1_int - _int_lorentz(f1_int),
            v2_int - _int_lorentz(f2_int),
            v3_int - _int_lorentz(f3_int),
            v4_int - _int_lorentz(f4_int),
            v5_int - _int_lorentz(f5_int),
        ], dim=-2)

        # We now solve the system A x = b via an LU decomposition
        x_int = torch.linalg.solve(A_int, b_int)
        b0_raw = x_int[:, 0, :]
        b1_raw = x_int[:, 1, :]
        b2_raw = x_int[:, 2, :]
        b3_raw = x_int[:, 3, :]
        b4_raw = x_int[:, 4, :]

        # Set intermediate phenomenological coefficients from solution
        # to A x = b
        b0 = b0_raw
        b1 = b1_raw * fRING
        b2 = b2_raw * fRING * fRING
        b3 = b3_raw * fRING * fRING * fRING
        b4 = b4_raw * fRING * fRING * fRING * fRING

        # Source: IMRPhenomX_Phase_22_ConnectionCoefficients
        # (LALSimIMRPhenomX_internals.c)
        fIns = fPhaseMatchIN
        fInt = fPhaseMatchIM

        # Phase Derivative: TaylorF2 + pseudo-PN coefficients
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

        # Ansatz for the inspiral phase.
        def _phi_ins(f):
            logf = torch.log(f)
            phasing = (
                phi0
                + phi1  * f.pow(1.0/3.0)
                + phi2  * f.pow(2.0/3.0)
                + phi3  * f
                + phi4  * f.pow(4.0/3.0)
                + phi5  * f.pow(5.0/3.0)
                + phi5L * f.pow(5.0/3.0) * logf
                + phi6  * f.pow(2.0)
                + phi6L * f.pow(2.0) * logf
                + phi7  * f.pow(7.0/3.0)
                + phi8  * f.pow(8.0/3.0)
                + phi8L * f.pow(8.0/3.0) * logf
                + phi9  * f.pow(3.0)
                + phi9L * f.pow(3.0) * logf
                # Now add in the pseudo-PN Coefficients
                + sigma1 * f.pow(8.0/3.0)
                + sigma2 * f.pow(3.0)
                + sigma3 * f.pow(10.0/3.0)
                + sigma4 * f.pow(11.0/3.0)
                + sigma5 * f.pow(4.0)
            )
            return phiNorm * f.pow(-5.0/3.0) * phasing

        # Phase derivative ansatz for intermediate region
        def _dphi_int(f):
            inv1 = 1.0 / f
            inv2 = inv1 * inv1
            inv3 = inv2 * inv1
            inv4 = inv2 * inv2
            lorentz = 4.0 * cL / (4.0*fDAMP*fDAMP + (f - fRING)*(f - fRING))
            return b0 + b1*inv1 + b2*inv2 + b3*inv3 + b4*inv4 + lorentz

        # Integrated phase ansatz for intermediate region
        def _phi_int(f):
            inv1 = 1.0 / f
            inv2 = inv1 * inv1
            inv3 = inv2 * inv1
            logf = torch.log(f)
            atan_term = (2.0 * cL / fDAMP) * torch.atan((f - fRING) / (2.0 * fDAMP))
            return b0*f + b1*logf - b2*inv1 - (b3/2.0)*inv2 - (b4/3.0)*inv3 + atan_term

        # Phenomenological ringdown phase derivative ansatz
        def _dphi_rd(f):
            inv1 = 1.0 / f
            fi_m1o3 = f.pow(-1.0/3.0)
            fi_m2   = inv1 * inv1
            fi_m4   = fi_m2 * fi_m2
            lorentz = cL / (fDAMP*fDAMP + (f - fRING)*(f - fRING))
            return c0 + c1*fi_m1o3 + c2*fi_m2 + c4*fi_m4 + lorentz

        # Phenomenological ringdown phase ansatz (i.e. integral of phase
        # derivative ansatz).
        def _phi_rd(f):
            inv1 = 1.0 / f
            inv3 = inv1 * inv1 * inv1
            f2o3 = f.pow(2.0/3.0)
            atan_term = cLovfda * torch.atan((f - fRING) / fDAMP)
            return c0*f + 1.5*c1*f2o3 - c2*inv1 - c4ov3*inv3 + atan_term

        # Assume an ansatz of the form:
        # phi_Inspiral (f) = phi_Intermediate (f) + C1 + C2 * f
        # where transition frequency is fIns
        # C2 = phi_Inspiral'(fIns) - phi_Intermediate'(fIns)
        C2Int = _dphi_ins(fIns) - _dphi_int(fIns)

        # C1 = phi_Inspiral (fIns) - phi_Intermediate (fIns) - C2 * fIns
        C1Int = _phi_ins(fIns) - _phi_int(fIns) - C2Int * fIns

        # Assume an ansatz of the form:
        # phi_Intermediate (f)    = phi_Ringdown (f) + C1 + C2 * f
        # where transition frequency is fIM
        C2MRD = _dphi_int(fInt) + C2Int - _dphi_rd(fInt)

        C1MRD = (_phi_int(fInt) + C1Int + C2Int*fInt) - _phi_rd(fInt) - C2MRD*fInt

        return {
            'phiNorm':  phiNorm,
            'dphase0':  dphase0,
            'phi0': phi0, 'phi1': phi1, 'phi2': phi2, 'phi3': phi3,
            'phi4': phi4, 'phi5': phi5, 'phi5L': phi5L,
            'phi6': phi6, 'phi6L': phi6L,
            'phi7': phi7,
            'phi8': phi8, 'phi8L': phi8L,
            'phi9': phi9, 'phi9L': phi9L,
            'sigma1': sigma1, 'sigma2': sigma2, 'sigma3': sigma3,
            'sigma4': sigma4, 'sigma5': sigma5,
            'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3,
            'dphi0': dphi0, 'dphi1': dphi1, 'dphi2': dphi2, 'dphi3': dphi3,
            'dphi4': dphi4, 'dphi5': dphi5,
            'dphi6': dphi6, 'dphi6L': dphi6L,
            'dphi7': dphi7,
            'dphi8': dphi8, 'dphi8L': dphi8L,
            'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4,
            'c0': c0, 'c1': c1, 'c2': c2, 'c4': c4,
            'cL': cL, 'cLovfda': cLovfda, 'c4ov3': c4ov3,
            'fRING': fRING, 'fDAMP': fDAMP,
            'fPhaseMatchIN': fPhaseMatchIN,
            'fPhaseMatchIM': fPhaseMatchIM,
            'C1Int': C1Int, 'C2Int': C2Int,
            'C1MRD': C1MRD, 'C2MRD': C2MRD,
        }

    def phase(self, f_Ms, phase_coeffs, loop=False):
        """
        Raw (2,2) phase over the grid, without the 1/eta scaling.

        Inspiral below fPhaseMatchIN, intermediate below fPhaseMatchIM,
        ringdown otherwise. Source: IMRPhenomX_Phase_22
        (LALSimIMRPhenomX_internals.c).

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, F)
            Dimensionless frequency Mf.
        phase_coeffs : dict
            Output of get_phase_coeffs.
        loop : bool
            Region boundaries of the waveform loop instead. Source:
            IMRPhenomXAS_GenerateFD (LALSimIMRPhenomX.c).

        Returns
        -------
        psi : torch.Tensor, shape (B, F)
            Raw phase in radians.
        """
        pc = phase_coeffs
        Mf = f_Ms

        fIN  = pc['fPhaseMatchIN']
        fIM  = pc['fPhaseMatchIM']
        fRING = pc['fRING']
        fDAMP = pc['fDAMP']
        cL    = pc['cL']
        cLovfda = pc['cLovfda']
        c4ov3   = pc['c4ov3']

        # Ansatz for the inspiral phase.
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
            # Now add in the pseudo-PN Coefficients
            + pc['sigma1'] * Mf.pow(8.0/3.0)
            + pc['sigma2'] * Mf.pow(3.0)
            + pc['sigma3'] * Mf.pow(10.0/3.0)
            + pc['sigma4'] * Mf.pow(11.0/3.0)
            + pc['sigma5'] * Mf.pow(4.0)
        )
        phi_ins = pc['phiNorm'] * Mf.pow(-5.0/3.0) * phi_ins

        # Integrated phase ansatz for intermediate region
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

        # Phenomenological ringdown phase ansatz (i.e. integral of phase
        # derivative ansatz).
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

        if loop:
            return torch.where(Mf < fIN, phi_ins,
                   torch.where(Mf > fIM, phi_rd, phi_int))
        return torch.where(Mf < fIN, phi_ins,
               torch.where(Mf < fIM, phi_int, phi_rd))

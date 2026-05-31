"""Tests for sage.data.waveform.approximants.IMRPhenomXAS.

Covers:
  debug_b_coeffs.py      — b-coefficient collocation consistency
  debug_c2int.py         — C2Int phase-continuity constant
  debug_linb.py          — C2MRD and linb formula chain
  debug_xas_phase.py     — phase comparison against LALSim
  debug_xas_step_by_step.py — amplitude/phase mismatch against LALSim
"""

import math
import pytest
import numpy as np
import torch

pytest.importorskip("pycbc", reason="pycbc required for LALSim comparison")

from sage.data.waveform.approximants.IMRPhenomXAS import IMRPhenomXAS

# ---------------------------------------------------------------------------
# Shared physical setup
# ---------------------------------------------------------------------------

FSAMP   = 2048.0
SEG_LEN = 16.0
NFREQ   = int(SEG_LEN * FSAMP) // 2 + 1
DELTA_F = 1.0 / SEG_LEN
DTYPE   = torch.float64
F_LOW   = 20.0

_f_arr = torch.linspace(0.0, FSAMP / 2, NFREQ, dtype=DTYPE).unsqueeze(0)
_f_ref = torch.tensor([[F_LOW]], dtype=DTYPE)
_model  = IMRPhenomXAS(_f_arr, _f_ref)

# Reference parameters: equal-mass, non-spinning BBH
_THETA_EQ = torch.tensor([[30.0, 30.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0]], dtype=DTYPE)


def _make_lal_hp(m1, m2, chi1z, chi2z, dist=400.0):
    from pycbc.waveform import get_fd_waveform
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomXAS",
        mass1=m1, mass2=m2,
        spin1z=chi1z, spin2z=chi2z,
        distance=dist,
        delta_f=DELTA_F,
        f_lower=F_LOW, f_ref=F_LOW,
        inclination=0.0, coa_phase=0.0,
    )
    if len(hp) < NFREQ:
        hp.resize(NFREQ)
    return np.array(hp[:NFREQ], dtype=np.complex128)


def _mismatch(h1, h2):
    from pycbc.filter import match
    import pycbc.types
    h1_p = pycbc.types.FrequencySeries(h1, delta_f=DELTA_F)
    h2_p = pycbc.types.FrequencySeries(h2, delta_f=DELTA_F)
    psd   = pycbc.types.FrequencySeries(np.ones(len(h1), dtype=np.float64), delta_f=DELTA_F)
    m, _  = match(h1_p, h2_p, psd=psd, low_frequency_cutoff=F_LOW)
    return 1.0 - m


# ---------------------------------------------------------------------------
# Phase-coefficient internal-consistency tests  (debug_c2int, debug_linb,
#                                               debug_b_coeffs)
# ---------------------------------------------------------------------------

class TestXASPhaseCoefficients:
    """Verify that stored phase-continuity constants equal their defining formulae."""

    @pytest.fixture(scope="class")
    def setup(self):
        derived = _model.compute_derived_parameters(_THETA_EQ)
        pc      = _model.get_phase_coeffs(derived)
        return derived, pc

    # -- C2Int ---------------------------------------------------------------

    def test_c2int_stored_matches_formula(self, setup):
        """C2Int = dphi_ins(fPhaseMatchIN) - dphi_int(fPhaseMatchIN).

        With the 5-term GC collocation (case 105), the first collocation
        value is v1_int = dphi_ins(fPhaseMatchIN), so the solve enforces
        dphi_int(fPhaseMatchIN) = dphi_ins(fPhaseMatchIN) exactly, making
        C2Int ≡ 0 by construction.  The test verifies this self-consistency.
        """
        derived, pc = setup
        fIns  = pc["fPhaseMatchIN"]   # (1, 1)
        fRING = pc["fRING"]
        fDAMP = pc["fDAMP"]
        C2Int = pc["C2Int"]

        # Inspiral derivative at fPhaseMatchIN
        logf = torch.log(fIns)
        poly = (
            pc["dphi0"]  * torch.ones_like(fIns)
            + pc["dphi1"]  * fIns.pow(1.0 / 3.0)
            + pc["dphi2"]  * fIns.pow(2.0 / 3.0)
            + pc["dphi3"]  * fIns
            + pc["dphi4"]  * fIns.pow(4.0 / 3.0)
            + pc["dphi5"]  * fIns.pow(5.0 / 3.0)
            + pc["dphi6"]  * fIns * fIns
            + pc["dphi6L"] * fIns * fIns * logf
            + pc["dphi7"]  * fIns.pow(7.0 / 3.0)
            + pc["dphi8"]  * fIns.pow(8.0 / 3.0)
            + pc["dphi8L"] * fIns.pow(8.0 / 3.0) * logf
            + pc["a0"]     * fIns.pow(8.0 / 3.0)
            + pc["a1"]     * fIns.pow(3.0)
            + pc["a2"]     * fIns.pow(10.0 / 3.0)
            + pc["a3"]     * fIns.pow(11.0 / 3.0)
        )
        dphi_ins = poly * fIns.pow(-8.0 / 3.0) * pc["dphase0"]

        # Intermediate derivative (5-term, case 105) at fPhaseMatchIN
        inv1 = 1.0 / fIns
        inv2 = inv1 * inv1
        inv3 = inv2 * inv1
        inv4 = inv2 * inv2
        lorentz = 4.0 * pc["cL"] / (4.0 * fDAMP * fDAMP + (fIns - fRING) ** 2)
        dphi_int = (
            pc["b0"] + pc["b1"] * inv1 + pc["b2"] * inv2
            + pc["b3"] * inv3 + pc["b4"] * inv4 + lorentz
        )

        C2Int_formula = (dphi_ins - dphi_int).item()
        assert abs(C2Int_formula - C2Int.item()) < 1e-6, (
            f"C2Int mismatch: stored={C2Int.item():.8f}, formula={C2Int_formula:.8f}"
        )

    # -- C2MRD ---------------------------------------------------------------

    def test_c2mrd_stored_matches_formula(self, setup):
        """C2MRD = (dphi_int + C2Int)(fPhaseMatchIM) - dphi_rd(fPhaseMatchIM).

        Uses the 5-term intermediate ansatz (case 105: b0..b4 including b3/f^3).
        """
        derived, pc = setup
        fInt  = pc["fPhaseMatchIM"]   # (1, 1)
        fRING = pc["fRING"]
        fDAMP = pc["fDAMP"]
        C2Int = pc["C2Int"]
        C2MRD = pc["C2MRD"]

        # Intermediate derivative (5-term) + C2Int at fPhaseMatchIM
        inv1 = 1.0 / fInt
        inv2 = inv1 * inv1
        inv3 = inv2 * inv1
        inv4 = inv2 * inv2
        lorentz_int = 4.0 * pc["cL"] / (4.0 * fDAMP * fDAMP + (fInt - fRING) ** 2)
        dphi_int_at_fInt = (
            pc["b0"] + pc["b1"] * inv1 + pc["b2"] * inv2
            + pc["b3"] * inv3 + pc["b4"] * inv4
            + lorentz_int + C2Int
        )

        # Ringdown derivative at fPhaseMatchIM
        lorentz_rd = pc["cL"] / (fDAMP ** 2 + (fInt - fRING) ** 2)
        dphi_rd_at_fInt = (
            pc["c0"]
            + pc["c1"] * fInt.pow(-1.0 / 3.0)
            + pc["c2"] * inv2
            + pc["c4"] * inv4
            + lorentz_rd
        )

        C2MRD_formula = (dphi_int_at_fInt - dphi_rd_at_fInt).item()
        assert abs(C2MRD_formula - C2MRD.item()) < 1e-6, (
            f"C2MRD mismatch: stored={C2MRD.item():.8f}, formula={C2MRD_formula:.8f}"
        )

    # -- b-coefficients collocation (debug_b_coeffs) -------------------------

    def test_b_coeffs_collocation_at_first_point(self, setup):
        """The 5-term GC solve (case 105) enforces dphi_int(fPhaseMatchIN) = dphi_ins(fPhaseMatchIN).

        This is equivalent to C2Int = 0, and verifies the b-coefficient linear
        system is self-consistent with the evaluation formula.
        """
        derived, pc = setup
        fIns  = pc["fPhaseMatchIN"]
        fRING = pc["fRING"]
        fDAMP = pc["fDAMP"]

        # dphi_ins at fPhaseMatchIN (the first collocation value v1_int)
        logf = torch.log(fIns)
        poly = (
            pc["dphi0"]  * torch.ones_like(fIns)
            + pc["dphi1"]  * fIns.pow(1.0 / 3.0)
            + pc["dphi2"]  * fIns.pow(2.0 / 3.0)
            + pc["dphi3"]  * fIns
            + pc["dphi4"]  * fIns.pow(4.0 / 3.0)
            + pc["dphi5"]  * fIns.pow(5.0 / 3.0)
            + pc["dphi6"]  * fIns * fIns
            + pc["dphi6L"] * fIns * fIns * logf
            + pc["dphi7"]  * fIns.pow(7.0 / 3.0)
            + pc["dphi8"]  * fIns.pow(8.0 / 3.0)
            + pc["dphi8L"] * fIns.pow(8.0 / 3.0) * logf
            + pc["a0"]     * fIns.pow(8.0 / 3.0)
            + pc["a1"]     * fIns.pow(3.0)
            + pc["a2"]     * fIns.pow(10.0 / 3.0)
            + pc["a3"]     * fIns.pow(11.0 / 3.0)
        )
        dphi_ins = (poly * fIns.pow(-8.0 / 3.0) * pc["dphase0"]).item()

        # 5-term intermediate evaluation at fPhaseMatchIN (first GC collocation point)
        inv1 = 1.0 / fIns
        inv2 = inv1 * inv1
        inv3 = inv2 * inv1
        inv4 = inv2 * inv2
        lorentz = (4.0 * pc["cL"] / (4.0 * fDAMP * fDAMP + (fIns - fRING) ** 2)).item()
        dphi_int_at_f1 = (
            pc["b0"] + pc["b1"] * inv1 + pc["b2"] * inv2
            + pc["b3"] * inv3 + pc["b4"] * inv4
        ).item() + lorentz

        # Collocation enforces dphi_int(f1) == dphi_ins(f1) to numerical precision
        assert abs(dphi_int_at_f1 - dphi_ins) < 1e-4, (
            f"b-coeff GC collocation at f1: dphi_int={dphi_int_at_f1:.6f}, "
            f"dphi_ins={dphi_ins:.6f}, diff={dphi_int_at_f1 - dphi_ins:.2e}"
        )

    # -- linb components (debug_linb) ----------------------------------------

    def test_linb_formula_components_finite(self, setup):
        """linb, linb_fit, psi4, and dphi22Ref should all be finite scalars."""
        derived, pc = setup
        eta    = derived[0, 3:4]
        STotR  = derived[0, 9:10]
        dchi   = derived[0, 10:11]
        delta  = derived[0, 4:5]

        linb_fit = _model._linb_fit(eta, STotR, dchi, delta)
        psi4     = _model._psi4tostrain_fit(eta, STotR, dchi)

        frefFit = pc["fRING"] - pc["fDAMP"]
        dph_ref = _model.dphase(frefFit, pc, derived)

        dphi22Ref = (1.0 / eta) * dph_ref
        linb      = linb_fit - dphi22Ref - 2.0 * math.pi * (500.0 + psi4)

        for name, val in [("linb_fit", linb_fit), ("psi4", psi4),
                          ("dphi22Ref", dphi22Ref), ("linb", linb)]:
            assert torch.isfinite(val).all(), f"{name} contains non-finite values"

    def test_derived_parameters_equal_mass(self, setup):
        """eta=0.25 and delta=0 for equal-mass system."""
        derived, _ = setup
        assert abs(derived[0, 3].item() - 0.25) < 1e-10
        assert abs(derived[0, 4].item()) < 1e-10


# ---------------------------------------------------------------------------
# Mismatch tests against LALSim  (debug_xas_phase, debug_xas_step_by_step)
# ---------------------------------------------------------------------------

class TestXASMismatch:
    """End-to-end mismatch between Sage IMRPhenomXAS and LALSim."""

    THRESHOLD = 2e-4

    @pytest.mark.parametrize("m1,m2,chi1z,chi2z,label", [
        (30.0, 30.0,  0.0,  0.0,  "30+30 spin0"),
        (20.0, 10.0,  0.3, -0.2,  "20+10 spinning"),
        (36.0, 29.0,  0.31, -0.46, "GW150914-like"),
        (10.0,  5.0,  0.5,  0.1,  "10+5 high spin"),
    ])
    def test_mismatch_vs_lalsim(self, m1, m2, chi1z, chi2z, label):
        hp_lal  = _make_lal_hp(m1, m2, chi1z, chi2z)
        theta   = torch.tensor([[m1, m2, chi1z, chi2z, 400.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
        hp_sage, _ = _model.get_hphc(theta, reproduce_lal=True)
        hp_sage_np = hp_sage[0].detach().cpu().numpy()

        mm = _mismatch(hp_lal, hp_sage_np)
        assert mm < self.THRESHOLD, (
            f"[{label}] mismatch {mm:.2e} exceeds threshold {self.THRESHOLD:.2e}"
        )

    def test_amplitude_ratio_near_unity(self):
        """Amplitude ratio |sage|/|lal| should be within 1 % across the band."""
        hp_lal  = _make_lal_hp(30.0, 30.0, 0.0, 0.0)
        hp_sage, _ = _model.get_hphc(_THETA_EQ, reproduce_lal=True)
        hp_sage_np = hp_sage[0].detach().cpu().numpy()

        for f_hz in [30, 50, 100, 200]:
            idx = int(round(f_hz * SEG_LEN))
            if idx >= NFREQ or abs(hp_lal[idx]) < 1e-40:
                continue
            ratio = abs(hp_sage_np[idx]) / abs(hp_lal[idx])
            assert 0.99 < ratio < 1.01, (
                f"Amplitude ratio at {f_hz} Hz = {ratio:.5f}, expected in [0.99, 1.01]"
            )

"""Tests for sage.data.waveform.approximants.IMRPhenomXAS_NRTidalv3.

Covers:
  debug_nrtidal.py       — tidal-phase comparison against LALSim
  debug_nrtidal2.py      — Padé coefficient comparison against LALSim
  debug_nrtidal_match.py — end-to-end mismatch for BNS parameter grid
"""

import pytest
import numpy as np
import torch

pytest.importorskip("pycbc", reason="pycbc required for LALSim comparison")

from sage.data.waveform.approximants.IMRPhenomXAS_NRTidalv3 import IMRPhenomXAS_NRTidalv3
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.core.config import register_configs, get_cfg

# ---------------------------------------------------------------------------
# Shared physical setup — production BNS grid (T=295 s, 2048 Hz, f_low=20 Hz)
#
# IMRPhenomXAS_NRTidalv3 builds its own frequency grid from the active data
# config; it does NOT take (f, f_ref) the way IMRPhenomD/IMRPhenomXAS do.  So a
# config must be registered before instantiating, and the LAL comparison grid
# below is derived from those same numbers.  These values match
# test_waveform_all_approximants.py and test_waveform_nrt_corner_mismatch.py so
# that the shared global config is consistent whatever order tests collect in.
# ---------------------------------------------------------------------------


def _ensure_bns_config():
    """Register the BNS production config if no suitable config is active yet."""
    try:
        cfg = get_cfg()
        if not (hasattr(cfg, "batch_size") and cfg.batch_size <= 4):
            raise RuntimeError("Wrong config active")
    except RuntimeError:
        class _Cfg:
            batch_size    = 2
            device        = "cpu"
            dtype         = torch.float64
            autocast      = False
            class_balance = 0.5

        class _DataCfg:
            sample_rate                  = 2048.0
            signal_low_frequency_cutoff  = 20.0
            noise_low_frequency_cutoff   = 15.0
            sample_length_in_s           = 287.0
            padding_length_in_s          = 4.0

        register_configs(BaseConfig(_Cfg()), BaseDataConfig(_DataCfg()))


_ensure_bns_config()

FSAMP   = 2048.0
SEG_LEN = 295.0                              # 287 + 2 x 4 padding
NFREQ   = int(FSAMP * SEG_LEN) // 2 + 1      # 302081
DELTA_F = 1.0 / SEG_LEN
DTYPE   = torch.float64
F_LOW   = 20.0
MSUN_S  = 4.925491025873693e-6

_model = IMRPhenomXAS_NRTidalv3()

# Reference: symmetric 1.4+1.4 Msun BNS with Lambda=500
_THETA_SYM = torch.tensor(
    [[1.4, 1.4, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 500.0, 500.0]], dtype=DTYPE
)


def _make_lal_bns_hp(m1, m2, chi1z, chi2z, L1, L2, dist=100.0):
    from pycbc.waveform import get_fd_waveform
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomXAS_NRTidalv3",
        mass1=m1, mass2=m2,
        spin1z=chi1z, spin2z=chi2z,
        lambda1=L1, lambda2=L2,
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
# Structural / internal-consistency tests  (debug_nrtidal2, debug_nrtidal)
# ---------------------------------------------------------------------------

class TestNRTidalStructure:
    """Internal consistency checks that do not require LALSim."""

    @pytest.fixture(scope="class")
    def tidal_setup(self):
        derived = _model.compute_derived_parameters(_THETA_SYM)
        Xa      = derived[:, 5:6]
        Xb      = derived[:, 6:7]
        chi1L   = derived[:, 11:12]
        chi2L   = derived[:, 12:13]
        lambda1 = _THETA_SYM[:, 8:9]
        lambda2 = _THETA_SYM[:, 9:10]

        kappa2T        = _model._kappa2T(Xa, Xb, lambda1, lambda2)
        kappaA, kappaB = _model._kappaAB(Xa, Xb, lambda1, lambda2)
        PN             = _model._PN_coeffs(Xa)
        tc_d           = _model._nrtv3_coeffs(Xa, Xb, kappa2T, kappaA, kappaB, PN)
        Mfmerger       = _model._merger_freq_v3(Xa, Xb, lambda1, lambda2, chi1L, chi2L)
        return kappa2T, kappaA, kappaB, PN, tc_d, Mfmerger

    def test_kappa2t_positive(self, tidal_setup):
        """kappa2T must be positive for finite tidal deformabilities."""
        kappa2T = tidal_setup[0]
        assert kappa2T.item() > 0, f"kappa2T={kappa2T.item():.4f} is not positive"

    def test_kappa2t_symmetric_equal_lambda(self, tidal_setup):
        """For equal mass (Xa=Xb=0.5) and equal lambda=500, kappa2T = 3*(0.5^5+0.5^5)*500
        = 3*2*(1/32)*500 = 93.75."""
        kappa2T = tidal_setup[0]
        expected = 3.0 * 2.0 * (0.5 ** 5) * 500.0  # = 93.75
        assert abs(kappa2T.item() - expected) / expected < 1e-6, (
            f"kappa2T={kappa2T.item():.4f}, expected {expected:.4f}"
        )

    def test_merger_freq_positive(self, tidal_setup):
        """Merger frequency in dimensionless Mf units must be positive."""
        Mfmerger = tidal_setup[5]
        assert Mfmerger.item() > 0, f"Mfmerger={Mfmerger.item():.6f} is not positive"

    def test_merger_freq_in_plausible_range(self, tidal_setup):
        """Mfmerger for a 1.4+1.4 Msun BNS with Lambda=500 is typically ~0.02–0.04 Mf."""
        Mfmerger = tidal_setup[5]
        Mfmerger_val = Mfmerger.item()
        assert 0.01 < Mfmerger_val < 0.06, (
            f"Mfmerger={Mfmerger_val:.5f} is outside plausible range [0.01, 0.06]"
        )

    def test_pade_coefficients_finite(self, tidal_setup):
        """All Padé numerator/denominator coefficients must be finite."""
        tc_d = tidal_setup[4]
        for key in ["n_1A", "n_3o2A", "n_2A", "n_5o2A", "n_3A",
                    "d_1A", "d_3o2A",
                    "n_1B", "n_3o2B", "n_2B", "n_5o2B", "n_3B",
                    "d_1B", "d_3o2B"]:
            val = tc_d[key]
            assert torch.isfinite(val).all(), f"Padé coefficient {key} is not finite"

    def test_pn_coefficients_finite(self, tidal_setup):
        """PN coefficient tensor must have all finite entries."""
        PN = tidal_setup[3]
        assert torch.isfinite(PN).all(), "PN coefficient tensor contains non-finite values"

    def test_tidal_phase_negative_and_growing(self, tidal_setup):
        """Tidal phase phi_tidal should be negative (attractive) and grow in magnitude."""
        _, _, _, PN, tc_d, _ = tidal_setup
        derived  = _model.compute_derived_parameters(_THETA_SYM)
        M_s      = derived[0, 2].item()
        M_total  = (1.4 + 1.4) * MSUN_S

        f_low_Mf  = torch.tensor([[50.0 * M_total]], dtype=DTYPE)
        f_high_Mf = torch.tensor([[800.0 * M_total]], dtype=DTYPE)
        phi_low   = _model._phi_tidal_nrt(f_low_Mf,  PN, tc_d).item()
        phi_high  = _model._phi_tidal_nrt(f_high_Mf, PN, tc_d).item()

        assert phi_low < 0,  f"phi_tidal at 50 Hz = {phi_low:.4f} (expected < 0)"
        assert phi_high < 0, f"phi_tidal at 800 Hz = {phi_high:.4f} (expected < 0)"
        assert phi_high < phi_low, (
            f"phi_tidal magnitude should grow with frequency: "
            f"phi(50 Hz)={phi_low:.4f}, phi(800 Hz)={phi_high:.4f}"
        )


# ---------------------------------------------------------------------------
# Tidal-phase comparison against LALSim  (debug_nrtidal2)
# ---------------------------------------------------------------------------

class TestNRTidalPhaseVsLAL:
    """Compare raw Sage tidal phase against LALSim's NRTidalv3 output."""

    def test_tidal_phase_formula_vs_lalsim(self):
        """Sage _phi_tidal_nrt at low frequencies should agree with LALSim
        within 0.05 rad (before amplitude clamping/tapering dominates)."""
        lalsimulation = pytest.importorskip(
            "lalsimulation", reason="lalsimulation required for direct tidal comparison"
        )
        import lal

        M1_SI = 1.4 * lal.MSUN_SI
        M2_SI = 1.4 * lal.MSUN_SI
        L1, L2 = 500.0, 500.0
        M_total_s = (1.4 + 1.4) * MSUN_S

        freqs_hz = [50.0, 100.0, 200.0]
        N = len(freqs_hz)
        phi_tidal_lal_arr = lal.CreateREAL8Sequence(N)
        amp_tidal_lal_arr = lal.CreateREAL8Sequence(N)
        planck_arr        = lal.CreateREAL8Sequence(N)
        fHz_arr           = lal.CreateREAL8Sequence(N)
        for i, f in enumerate(freqs_hz):
            fHz_arr.data[i] = f

        lalsimulation.SimNRTunedTidesFDTidalPhaseFrequencySeries(
            phi_tidal_lal_arr, amp_tidal_lal_arr, planck_arr, fHz_arr,
            M1_SI, M2_SI, L1, L2, 0.0, 0.0, lalsimulation.NRTidalv3_V,
        )

        derived  = _model.compute_derived_parameters(_THETA_SYM)
        Xa       = derived[:, 5:6]
        Xb       = derived[:, 6:7]
        lambda1  = _THETA_SYM[:, 8:9]
        lambda2  = _THETA_SYM[:, 9:10]
        kappa2T        = _model._kappa2T(Xa, Xb, lambda1, lambda2)
        kappaA, kappaB = _model._kappaAB(Xa, Xb, lambda1, lambda2)
        PN             = _model._PN_coeffs(Xa)
        tc_d           = _model._nrtv3_coeffs(Xa, Xb, kappa2T, kappaA, kappaB, PN)

        for i, f_hz in enumerate(freqs_hz):
            Mf_val   = torch.tensor([[f_hz * M_total_s]], dtype=DTYPE)
            phi_sage = _model._phi_tidal_nrt(Mf_val, PN, tc_d).item()
            phi_lal  = phi_tidal_lal_arr.data[i]
            diff     = abs(phi_sage - phi_lal)
            assert diff < 0.05, (
                f"phi_tidal at {f_hz} Hz: sage={phi_sage:.6f}, "
                f"lal={phi_lal:.6f}, diff={diff:.4f}"
            )


# ---------------------------------------------------------------------------
# End-to-end mismatch tests  (debug_nrtidal_match)
# ---------------------------------------------------------------------------

class TestNRTidalMismatch:
    """End-to-end mismatch between Sage IMRPhenomXAS_NRTidalv3 and LALSim."""

    THRESHOLD = 2e-4

    @pytest.mark.parametrize("m1,m2,chi1z,chi2z,L1,L2,label", [
        (1.4,  1.4,  0.0,   0.0,  500.0, 500.0,  "1.4+1.4 sym spin0 L500"),
        (1.4,  1.2,  0.0,   0.0,  500.0, 300.0,  "1.4+1.2 asym spin0 L500/300"),
        (1.6,  1.2,  0.05, -0.05, 400.0, 200.0,  "1.6+1.2 spin L400/200"),
        (2.0,  1.2,  0.1,   0.0,  200.0, 400.0,  "2.0+1.2 spin L200/400"),
        (1.35, 1.35, 0.0,   0.0, 1000.0, 1000.0, "1.35+1.35 sym L1000"),
        (1.4,  1.4,  0.0,   0.0,   0.0,    0.0,  "1.4+1.4 no tidal (BBH limit)"),
    ])
    def test_mismatch_vs_lalsim(self, m1, m2, chi1z, chi2z, L1, L2, label):
        hp_lal  = _make_lal_bns_hp(m1, m2, chi1z, chi2z, L1, L2)
        theta   = torch.tensor(
            [[m1, m2, chi1z, chi2z, 100.0, 0.0, 0.0, 0.0, L1, L2]], dtype=DTYPE
        )
        with torch.no_grad():
            hp_sage, _ = _model.get_hphc(theta, reproduce_lal=True)
        # get_hphc returns complex64; pycbc.filter.match requires complex128.
        hp_sage_np = hp_sage[0].detach().cpu().to(torch.complex128).numpy()

        mm = _mismatch(hp_lal, hp_sage_np)
        assert mm < self.THRESHOLD, (
            f"[{label}] mismatch {mm:.2e} exceeds threshold {self.THRESHOLD:.2e}"
        )

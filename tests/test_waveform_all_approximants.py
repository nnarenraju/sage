"""
Production parameter-space verification for all four Sage approximants.
Adapted from runs/verify_all_approx.py.

Tests hp and hc mismatch vs LALSim across random draws from each production prior:
  IMRPhenomD           o3b BBH grid (T=16 s, 2048 Hz, f_low=20 Hz)   5 draws
  IMRPhenomXAS         o3b BBH grid                                    5 draws
  IMRPhenomPv2         o3b BBH grid, isotropic precessing spins        5 draws
  NRTidalv3 (no MB)    BNS grid (T=295 s, 2048 Hz, f_low=20 Hz)      5 draws
  NRTidalv3 (MB)       worst-case coarse grid vs full-res              5 draws
"""

import math
import pytest
import numpy as np
import torch

pytest.importorskip("pycbc", reason="pycbc required for LALSim comparison")

from sage.data.waveform.approximants.IMRPhenomD   import IMRPhenomD
from sage.data.waveform.approximants.IMRPhenomXAS import IMRPhenomXAS
from sage.data.waveform.approximants.IMRPhenomPv2 import IMRPhenomPv2
from sage.data.waveform import IMRPhenomXAS_NRTidalv3
from sage.data.waveform import waveform_utils

# ---------------------------------------------------------------------------
# Per-model mismatch thresholds (flat PSD, reproduce_lal=True vs LALSim)
#   IMRPhenomD  : known degradation at |chi|→0.99; ~2e-4 at extreme spin
#   IMRPhenomXAS: verified < 1.5e-5 across parameter space
#   IMRPhenomPv2: worst-case ~1.14e-7; 5e-4 covers edge cases
#   NRTidalv3   : verified ~4e-9 across BNS space
#   MB amp err  : coarse vs full-res amplitude; phase < 1e-3 rad is the key metric.
#                 Measured 5.74e-08 - 5.93e-08 over 20 independent BNS draws
#                 (systematic interpolation floor, ~3% spread), so 1e-6 leaves
#                 ~17x headroom while still catching a real regression.
# ---------------------------------------------------------------------------
THRESH_D   = 5e-4
THRESH_XAS = 2e-5
THRESH_PV2 = 5e-4
THRESH_NRT = 1e-6
THRESH_MB  = 1e-6

DTYPE  = torch.float64
N_PTS  = 5
_RNG   = np.random.default_rng(seed=20260602)

# ---------------------------------------------------------------------------
# BBH models — o3b production grid: T=16 s, 2048 Hz, f_low=20 Hz
# ---------------------------------------------------------------------------
BBH_SR = 2048.0
BBH_T  = 16.0
BBH_FL = 20.0
BBH_DF = 1.0 / BBH_T
BBH_N  = int(BBH_SR * BBH_T) // 2 + 1   # 16385

# IMRPhenomD/XAS/Pv2 expect a grid that STARTS at f_low and pad the
# 0 -> f_low bins themselves (self.n_pad).  Handing them a grid from 0 Hz gives
# n_pad = 0 and evaluates the PN amplitude at f = 0, producing NaN.  get_freqs
# builds the grid the models actually expect: n_pad + len(f) == BBH_N.
_f_bbh, _fr_bbh = waveform_utils.get_freqs(
    BBH_FL, BBH_SR / 2.0, BBH_T, batch_size=1, device="cpu", dtype=DTYPE
)

_model_D   = IMRPhenomD(_f_bbh, _fr_bbh)
_model_XAS = IMRPhenomXAS(_f_bbh, _fr_bbh)


def _make_pv2():
    m = object.__new__(IMRPhenomPv2)
    torch.nn.Module.__init__(m)
    IMRPhenomD.__init__(m, _f_bbh, _fr_bbh)
    m.B            = 1
    m.n_pad        = int(torch.round((m.f[0][0] - m.df) / m.df)) + 1
    m.hp_buffer    = torch.empty((1, m.n_pad + m.f_numel), dtype=torch.complex128)
    m.hc_buffer    = torch.empty_like(m.hp_buffer)
    m.param_sampler = m.waveform_project = m.augment = None
    return m


_model_Pv2 = _make_pv2()

# ---------------------------------------------------------------------------
# BNS config and NRTidal models — production: T=295 s, 2048 Hz, f_low=20 Hz
#   padded_length_in_s = sample_length (287) + 2 × padding (4) = 295 s
# ---------------------------------------------------------------------------
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.core.config import register_configs


class _BNSCfgRaw:
    batch_size    = 2
    device        = "cpu"
    dtype         = torch.float64
    autocast      = False
    class_balance = 0.5


class _BNSDataCfgRaw:
    sample_rate                  = 2048.0
    signal_low_frequency_cutoff  = 20.0
    noise_low_frequency_cutoff   = 15.0
    sample_length_in_s           = 287.0
    padding_length_in_s          = 4.0


register_configs(BaseConfig(_BNSCfgRaw()), BaseDataConfig(_BNSDataCfgRaw()))

BNS_SR = 2048.0
BNS_T  = 295.0      # 287 + 2×4
BNS_FL = 20.0
BNS_DF = 1.0 / BNS_T
BNS_N  = int(BNS_SR * BNS_T) // 2 + 1   # 302081

_model_NRT = IMRPhenomXAS_NRTidalv3()
_model_MB  = IMRPhenomXAS_NRTidalv3(multiband_mode="worst_case",
                                     m1_worst=1.3, m2_worst=1.1)

# ---------------------------------------------------------------------------
# Pre-drawn random parameter sets (BBH and BNS)
# ---------------------------------------------------------------------------
_bbh_m = _RNG.uniform(7, 50, (N_PTS, 2)); _bbh_m.sort(1); _bbh_m = _bbh_m[:, ::-1]
_bbh_m1  = _bbh_m[:, 0].copy()
_bbh_m2  = _bbh_m[:, 1].copy()
_bbh_c1z = _RNG.uniform(-0.99, 0.99, N_PTS)
_bbh_c2z = _RNG.uniform(-0.99, 0.99, N_PTS)
_bbh_inc  = np.arccos(_RNG.uniform(-1, 1, N_PTS))
_bbh_phic = _RNG.uniform(0, 2 * math.pi, N_PTS)
_bbh_dist = _RNG.uniform(200, 1500, N_PTS)


def _rand_spin(n):
    a = _RNG.uniform(0, 0.99, n)
    p = np.arccos(_RNG.uniform(-1, 1, n))
    z = _RNG.uniform(0, 2 * math.pi, n)
    return a * np.sin(p) * np.cos(z), a * np.sin(p) * np.sin(z), a * np.cos(p)


_bbh_c1x, _bbh_c1y, _bbh_c1zp = _rand_spin(N_PTS)
_bbh_c2x, _bbh_c2y, _bbh_c2zp = _rand_spin(N_PTS)

_bns_m = _RNG.uniform(1, 3, (N_PTS, 2)); _bns_m.sort(1); _bns_m = _bns_m[:, ::-1]
_bns_m1   = _bns_m[:, 0].copy()
_bns_m2   = _bns_m[:, 1].copy()
_bns_c1z  = _RNG.uniform(-0.4, 0.4, N_PTS)
_bns_c2z  = _RNG.uniform(-0.4, 0.4, N_PTS)
_bns_lam1 = _RNG.uniform(0, 5000, N_PTS)
_bns_lam2 = _RNG.uniform(0, 5000, N_PTS)
_bns_inc  = np.arccos(_RNG.uniform(-1, 1, N_PTS))
_bns_phic = _RNG.uniform(0, 2 * math.pi, N_PTS)
_bns_dist = _RNG.uniform(10, 500, N_PTS)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flat_match(h_sage, h_lal, df, f_low, N):
    import pycbc.types
    from pycbc.filter import match as pycbc_match

    def prep(h):
        h = np.asarray(h, dtype=np.complex128)
        if len(h) < N:
            h = np.pad(h, (0, N - len(h)))
        return h[:N]

    psd = pycbc.types.FrequencySeries(np.ones(N, dtype=np.float64), delta_f=df)
    m, _ = pycbc_match(
        pycbc.types.FrequencySeries(prep(h_sage), delta_f=df),
        pycbc.types.FrequencySeries(prep(h_lal),  delta_f=df),
        psd=psd, low_frequency_cutoff=f_low,
    )
    return 1.0 - m


def _hf_clean(h, n_pad):
    h = np.asarray(h)[n_pad:]
    return not (np.any(np.isnan(h)) or np.any(np.isinf(h)) or np.all(np.abs(h) == 0.0))


def _lal_hphc(approx, m1, m2, c1z, c2z, dist, inc, phic,
               df, f_low, N, L1=0., L2=0., c1x=0., c1y=0., c2x=0., c2y=0.):
    from pycbc.waveform import get_fd_waveform

    hp, hc = get_fd_waveform(
        approximant=approx,
        mass1=float(m1), mass2=float(m2),
        spin1x=float(c1x), spin1y=float(c1y), spin1z=float(c1z),
        spin2x=float(c2x), spin2y=float(c2y), spin2z=float(c2z),
        lambda1=float(L1), lambda2=float(L2),
        distance=float(dist), delta_f=float(df),
        f_lower=float(f_low), f_ref=float(f_low),
        inclination=float(inc), coa_phase=float(phic),
    )

    def to_np(fs):
        d = np.array(fs.data, dtype=np.complex128)
        if len(d) < N:
            d = np.pad(d, (0, N - len(d)))
        return d[:N]

    return to_np(hp), to_np(hc)


# ---------------------------------------------------------------------------
# IMRPhenomD — aligned-spin BBH, o3b prior: m=[7,50], |chi|<0.99
# ---------------------------------------------------------------------------

class TestIMRPhenomDRandomBBH:

    @pytest.mark.parametrize("i", range(N_PTS))
    def test_hp_hc_mismatch(self, i):
        m1, m2 = _bbh_m1[i], _bbh_m2[i]
        c1, c2 = _bbh_c1z[i], _bbh_c2z[i]
        inc, phic, dist = _bbh_inc[i], _bbh_phic[i], _bbh_dist[i]

        th = torch.tensor([[m1, m2, c1, c2, dist, 0., phic, inc]], dtype=DTYPE)
        with torch.no_grad():
            hp_s, hc_s = _model_D.get_hphc(th, reproduce_lal=True)
        hp_np = hp_s[0].to(torch.complex128).numpy()
        hc_np = hc_s[0].to(torch.complex128).numpy()

        hl, hcl = _lal_hphc("IMRPhenomD", m1, m2, c1, c2, dist, inc, phic,
                              BBH_DF, BBH_FL, BBH_N)
        mhp = _flat_match(hp_np, hl,  BBH_DF, BBH_FL, BBH_N)
        mhc = _flat_match(hc_np, hcl, BBH_DF, BBH_FL, BBH_N)

        assert _hf_clean(hp_np, _model_D.n_pad), f"hp NaN/Inf/zero (sample {i})"
        assert _hf_clean(hc_np, _model_D.n_pad), f"hc NaN/Inf/zero (sample {i})"
        assert mhp < THRESH_D, f"hp mismatch {mhp:.2e} > {THRESH_D:.0e} (sample {i})"
        assert mhc < THRESH_D, f"hc mismatch {mhc:.2e} > {THRESH_D:.0e} (sample {i})"


# ---------------------------------------------------------------------------
# IMRPhenomXAS — aligned-spin BBH, o3b prior
# ---------------------------------------------------------------------------

class TestIMRPhenomXASRandomBBH:

    @pytest.mark.parametrize("i", range(N_PTS))
    def test_hp_hc_mismatch(self, i):
        m1, m2 = _bbh_m1[i], _bbh_m2[i]
        c1, c2 = _bbh_c1z[i], _bbh_c2z[i]
        inc, phic, dist = _bbh_inc[i], _bbh_phic[i], _bbh_dist[i]

        th = torch.tensor([[m1, m2, c1, c2, dist, 0., phic, inc]], dtype=DTYPE)
        with torch.no_grad():
            hp_s, hc_s = _model_XAS.get_hphc(th, reproduce_lal=True)
        hp_np = hp_s[0].to(torch.complex128).numpy()
        hc_np = hc_s[0].to(torch.complex128).numpy()

        hl, hcl = _lal_hphc("IMRPhenomXAS", m1, m2, c1, c2, dist, inc, phic,
                              BBH_DF, BBH_FL, BBH_N)
        mhp = _flat_match(hp_np, hl,  BBH_DF, BBH_FL, BBH_N)
        mhc = _flat_match(hc_np, hcl, BBH_DF, BBH_FL, BBH_N)

        assert _hf_clean(hp_np, _model_XAS.n_pad), f"hp NaN/Inf/zero (sample {i})"
        assert _hf_clean(hc_np, _model_XAS.n_pad), f"hc NaN/Inf/zero (sample {i})"
        assert mhp < THRESH_XAS, f"hp mismatch {mhp:.2e} > {THRESH_XAS:.0e} (sample {i})"
        assert mhc < THRESH_XAS, f"hc mismatch {mhc:.2e} > {THRESH_XAS:.0e} (sample {i})"


# ---------------------------------------------------------------------------
# IMRPhenomPv2 — precessing BBH, isotropic spins, o3b prior
# ---------------------------------------------------------------------------

class TestIMRPhenomPv2RandomBBH:

    @pytest.mark.parametrize("i", range(N_PTS))
    def test_hp_hc_mismatch(self, i):
        m1, m2 = _bbh_m1[i], _bbh_m2[i]
        c1x, c1y, c1z = _bbh_c1x[i], _bbh_c1y[i], _bbh_c1zp[i]
        c2x, c2y, c2z = _bbh_c2x[i], _bbh_c2y[i], _bbh_c2zp[i]
        inc, phic, dist = _bbh_inc[i], _bbh_phic[i], _bbh_dist[i]

        th = torch.tensor(
            [[m1, m2, c1x, c1y, c1z, c2x, c2y, c2z, dist, 0., phic, inc]], dtype=DTYPE
        )
        with torch.no_grad():
            hp_s, hc_s = _model_Pv2.get_hphc(th, reproduce_lal=True)
        hp_np = hp_s[0].to(torch.complex128).numpy()
        hc_np = hc_s[0].to(torch.complex128).numpy()

        hl, hcl = _lal_hphc(
            "IMRPhenomPv2", m1, m2, c1z, c2z, dist, inc, phic,
            BBH_DF, BBH_FL, BBH_N,
            c1x=c1x, c1y=c1y, c2x=c2x, c2y=c2y,
        )
        mhp = _flat_match(hp_np, hl,  BBH_DF, BBH_FL, BBH_N)
        mhc = _flat_match(hc_np, hcl, BBH_DF, BBH_FL, BBH_N)

        assert _hf_clean(hp_np, _model_Pv2.n_pad), f"hp NaN/Inf/zero (sample {i})"
        assert _hf_clean(hc_np, _model_Pv2.n_pad), f"hc NaN/Inf/zero (sample {i})"
        assert mhp < THRESH_PV2, f"hp mismatch {mhp:.2e} > {THRESH_PV2:.0e} (sample {i})"
        assert mhc < THRESH_PV2, f"hc mismatch {mhc:.2e} > {THRESH_PV2:.0e} (sample {i})"


# ---------------------------------------------------------------------------
# NRTidalv3, no multibanding — BNS prior: m=[1,3], |chi|<0.4, Lambda=[0,5000]
# ---------------------------------------------------------------------------

class TestNRTidalNoMBRandomBNS:

    @pytest.mark.parametrize("i", range(N_PTS))
    def test_hp_hc_mismatch(self, i):
        m1, m2 = _bns_m1[i], _bns_m2[i]
        c1, c2 = _bns_c1z[i], _bns_c2z[i]
        L1, L2 = _bns_lam1[i], _bns_lam2[i]
        inc, phic, dist = _bns_inc[i], _bns_phic[i], _bns_dist[i]

        th = torch.tensor([[m1, m2, c1, c2, dist, 0., phic, inc, L1, L2]], dtype=DTYPE)
        with torch.no_grad():
            hp_s, hc_s = _model_NRT.get_hphc(th, reproduce_lal=True)
        hp_np = hp_s[0].to(torch.complex128).numpy()
        hc_np = hc_s[0].to(torch.complex128).numpy()

        hl, hcl = _lal_hphc(
            "IMRPhenomXAS_NRTidalv3", m1, m2, c1, c2, dist, inc, phic,
            BNS_DF, BNS_FL, BNS_N, L1=L1, L2=L2,
        )
        mhp = _flat_match(hp_np, hl,  BNS_DF, BNS_FL, BNS_N)
        mhc = _flat_match(hc_np, hcl, BNS_DF, BNS_FL, BNS_N)

        assert _hf_clean(hp_np, _model_NRT.n_pad), f"hp NaN/Inf/zero (sample {i})"
        assert _hf_clean(hc_np, _model_NRT.n_pad), f"hc NaN/Inf/zero (sample {i})"
        assert mhp < THRESH_NRT, (
            f"hp mismatch {mhp:.2e} > {THRESH_NRT:.0e} "
            f"(sample {i}: m1={m1:.2f} m2={m2:.2f} L1={L1:.0f} L2={L2:.0f})"
        )
        assert mhc < THRESH_NRT, (
            f"hc mismatch {mhc:.2e} > {THRESH_NRT:.0e} "
            f"(sample {i}: m1={m1:.2f} m2={m2:.2f} L1={L1:.0f} L2={L2:.0f})"
        )


# ---------------------------------------------------------------------------
# NRTidalv3, worst-case multibanding — coarse vs full-resolution amplitude
# ---------------------------------------------------------------------------

# Bins inside the low-frequency FD roll-on taper are excluded from comparison.
_TAPER_BINS = 64

_mb_sel       = _model_MB.selector
_mb_coarse_idx = _mb_sel.coarse_indices.numpy()   # full-grid indices (incl. n_pad offset)
_mb_f_taper   = BNS_FL + _TAPER_BINS * BNS_DF
_mb_valid     = (
    (_mb_coarse_idx < BNS_N) &
    (_mb_coarse_idx >= int(round(_mb_f_taper / BNS_DF)))
)


# Amplitude floor for the multiband comparison, as a fraction of peak |h|.
# This must be a physically meaningful fraction, not merely "non-zero": a 1e-40
# floor admits post-merger tail bins sitting at ~1e-19 of peak, where a float64
# relative error is pure numerical noise.  On one BNS draw a single bin at
# |h|=2e-44 (7.8e-19 of peak) produced a spurious 3e-2 "error" while every bin
# carrying actual signal agreed to 5.8e-08.
_AMP_FLOOR = 1e-8


def _coarse_amp_err(a, b):
    """Max relative amplitude error over bins where b carries real signal."""
    amp_b = np.abs(b)
    mask  = amp_b > _AMP_FLOOR * amp_b.max()
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(np.abs(a[mask]) - amp_b[mask]) / amp_b[mask]))


class TestNRTidalWorstCaseMultiband:

    @pytest.mark.parametrize("i", range(N_PTS))
    def test_coarse_amplitude_error(self, i):
        m1, m2 = _bns_m1[i], _bns_m2[i]
        c1, c2 = _bns_c1z[i], _bns_c2z[i]
        L1, L2 = _bns_lam1[i], _bns_lam2[i]
        inc, phic, dist = _bns_inc[i], _bns_phic[i], _bns_dist[i]

        B = _model_MB.signal_batch_size
        th = torch.zeros(B, 10, dtype=DTYPE)
        th[:, 0] = m1;   th[:, 1] = m2
        th[:, 2] = c1;   th[:, 3] = c2
        th[:, 4] = dist; th[:, 5] = 0.
        th[:, 6] = phic; th[:, 7] = inc
        th[:, 8] = L1;   th[:, 9] = L2

        with torch.no_grad():
            hpm, hcm = _model_MB.get_hphc(th, reproduce_lal=False)
            hpf, hcf = _model_NRT.get_hphc(th, reproduce_lal=False)

        hpm_c = hpm[0].to(torch.complex128).numpy()   # (N_coarse,)
        hcm_c = hcm[0].to(torch.complex128).numpy()
        hpf_np = hpf[0].to(torch.complex128).numpy()  # (n_pad + f_numel,)
        hcf_np = hcf[0].to(torch.complex128).numpy()

        # Compare at valid coarse positions in the full-grid coordinates.
        err_hp = _coarse_amp_err(hpm_c[_mb_valid], hpf_np[_mb_coarse_idx[_mb_valid]])
        err_hc = _coarse_amp_err(hcm_c[_mb_valid], hcf_np[_mb_coarse_idx[_mb_valid]])

        assert err_hp < THRESH_MB, (
            f"hp coarse amp error {err_hp:.2e} > {THRESH_MB:.0e} (sample {i})"
        )
        assert err_hc < THRESH_MB, (
            f"hc coarse amp error {err_hc:.2e} > {THRESH_MB:.0e} (sample {i})"
        )

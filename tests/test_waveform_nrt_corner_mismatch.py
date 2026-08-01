"""
Statistical mismatch verification for IMRPhenomXAS_NRTidalv3 across random BNS draws.
Adapted from runs/corner_mismatch_nrt.py (computation portion; corner plot is in
sage/diagnostics/plot_nrt_corner_mismatch.py).

Uses the production BNS grid (T=295 s, 2048 Hz, f_low=20 Hz) and draws 10 random
BNS systems from the same prior as the corner plot (m=[1,3] Msun, |chi|<0.4,
Lambda=[0,5000]) to assert per-sample mismatch and signal health.
"""

import math
import pytest
import numpy as np
import torch

pytest.importorskip("pycbc", reason="pycbc required for LALSim comparison")

from sage.data.waveform import IMRPhenomXAS_NRTidalv3

# ---------------------------------------------------------------------------
# BNS config: production grid  T=295 s, 2048 Hz, f_low=20 Hz
# ---------------------------------------------------------------------------
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.core.config import register_configs, get_cfg


def _ensure_bns_config():
    """Register BNS production config if no config is active yet."""
    try:
        cfg = get_cfg()
        # Already registered — check it looks like a BNS config.
        if not (hasattr(cfg, 'batch_size') and cfg.batch_size <= 4):
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

BNS_SR = 2048.0
BNS_T  = 295.0      # sample_length (287) + 2 × padding (4)
BNS_FL = 20.0
BNS_DF = 1.0 / BNS_T
BNS_N  = int(BNS_SR * BNS_T) // 2 + 1   # 302081

DTYPE = torch.float64

_model = IMRPhenomXAS_NRTidalv3()

# ---------------------------------------------------------------------------
# Random BNS draws — same seed as corner_mismatch_nrt.py for reproducibility
# ---------------------------------------------------------------------------
N_SAMPLES = 10
THRESHOLD = 1e-6
_RNG = np.random.default_rng(seed=20260604)

_m = _RNG.uniform(1, 3, (N_SAMPLES, 2)); _m.sort(1); _m = _m[:, ::-1]
_m1   = _m[:, 0].copy()
_m2   = _m[:, 1].copy()
_c1z  = _RNG.uniform(-0.4, 0.4, N_SAMPLES)
_c2z  = _RNG.uniform(-0.4, 0.4, N_SAMPLES)
_lam1 = _RNG.uniform(0, 5000, N_SAMPLES)
_lam2 = _RNG.uniform(0, 5000, N_SAMPLES)
_inc  = np.arccos(_RNG.uniform(-1, 1, N_SAMPLES))
_phic = _RNG.uniform(0, 2 * math.pi, N_SAMPLES)
_dist = _RNG.uniform(10, 500, N_SAMPLES)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mismatch(hp_sage, hp_lal):
    import pycbc.types
    from pycbc.filter import match as pycbc_match

    def prep(h):
        h = np.asarray(h, dtype=np.complex128)
        if len(h) < BNS_N:
            h = np.pad(h, (0, BNS_N - len(h)))
        return h[:BNS_N]

    psd = pycbc.types.FrequencySeries(np.ones(BNS_N, dtype=np.float64), delta_f=BNS_DF)
    m, _ = pycbc_match(
        pycbc.types.FrequencySeries(prep(hp_sage), delta_f=BNS_DF),
        pycbc.types.FrequencySeries(prep(hp_lal),  delta_f=BNS_DF),
        psd=psd, low_frequency_cutoff=BNS_FL,
    )
    return 1.0 - m


def _lal_hp(m1, m2, chi1z, chi2z, L1, L2, dist, inc, phic):
    from pycbc.waveform import get_fd_waveform
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomXAS_NRTidalv3",
        mass1=float(m1), mass2=float(m2),
        spin1z=float(chi1z), spin2z=float(chi2z),
        lambda1=float(L1), lambda2=float(L2),
        distance=float(dist), delta_f=BNS_DF,
        f_lower=BNS_FL, f_ref=BNS_FL,
        inclination=float(inc), coa_phase=float(phic),
    )
    d = np.array(hp.data, dtype=np.complex128)
    if len(d) < BNS_N:
        d = np.pad(d, (0, BNS_N - len(d)))
    return d[:BNS_N]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNRTidalCornerMismatch:
    """Per-sample mismatch checks over random BNS draws (corner-plot parameter space)."""

    @pytest.mark.parametrize("i", range(N_SAMPLES))
    def test_hp_mismatch_below_threshold(self, i):
        m1, m2 = _m1[i], _m2[i]
        c1, c2 = _c1z[i], _c2z[i]
        L1, L2 = _lam1[i], _lam2[i]
        inc, phic, dist = _inc[i], _phic[i], _dist[i]

        th = torch.tensor([[m1, m2, c1, c2, dist, 0., phic, inc, L1, L2]], dtype=DTYPE)
        with torch.no_grad():
            hp_s, _ = _model.get_hphc(th, reproduce_lal=True)
        hp_np = hp_s[0].to(torch.complex128).numpy()

        assert not np.any(np.isnan(hp_np[_model.n_pad:])), (
            f"hp contains NaN (sample {i}: m1={m1:.2f} m2={m2:.2f})"
        )
        assert not np.any(np.isinf(hp_np[_model.n_pad:])), (
            f"hp contains Inf (sample {i}: m1={m1:.2f} m2={m2:.2f})"
        )

        hp_lal = _lal_hp(m1, m2, c1, c2, L1, L2, dist, inc, phic)
        mm = _mismatch(hp_np, hp_lal)

        assert mm < THRESHOLD, (
            f"hp mismatch {mm:.2e} > {THRESHOLD:.0e} "
            f"(sample {i}: m1={m1:.2f} m2={m2:.2f} L1={L1:.0f} L2={L2:.0f})"
        )

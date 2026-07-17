#!/usr/bin/env python
"""DECISIVE TEST: does Sage's IMRPhenomPv2 place the MERGER at the SAME t=0
reference as PyCBC (LAL)?

Both pipelines declare coalescence at raw-irfft sample 0:
  - PyCBC FD PhenomPv2 has epoch = -16 s = -N*dt  => coalescence(abs t=0) at raw sample 0.
  - Sage get_hphc(reproduce_lal=True) applies t_corr so coalescence is at t=0 (raw sample 0),
    and skips apply_tc (the downstream, mathematically-exact tc placement).

So with identical intrinsic params on the identical 0..1024 Hz / df=1/16 grid, the
difference in the merger location between Sage and PyCBC IS delta_t. ~0 => references
match (peak-vs-tc offsets are physical). tens of ms growing with mass/precession => bug.
"""
import os, sys, json, math
import numpy as np
import torch
from scipy.signal import hilbert

torch.manual_seed(0); np.random.seed(0)
sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/nagarajan/research/sage/sage/diagnostics/plots"

import torch as _t
from sage.core.config import register_configs, get_cfg, get_data_cfg
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()

class O3bCFG:
    export_dir = "/home/nagarajan/research/sage/sage/diagnostics/plots"
    fiducial_dir = "/home/nagarajan/research/sage/runs/o3b/run_export/fiducial_psds"
    batch_size = 128; device = "cpu"; dtype = _t.float32
    detectors = ["H1", "L1"]; train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]; autocast = False
    class_balance = 0.5; clip_norm = 1.0; dropout = 0.0; num_epochs = 10
    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)

class O3bDataCFG:
    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [_SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors]
    validation_noise_files = [_SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3bCFG.detectors]
    sample_rate = 2048.0; noise_low_frequency_cutoff = 15.0
    signal_low_frequency_cutoff = 20.0
    sample_length_in_s = 12.0; padding_length_in_s = 2.0

register_configs(BaseConfig(O3bCFG()), BaseDataConfig(O3bDataCFG()))
cfg, data_cfg = get_cfg(), get_data_cfg()

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler

param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
signal_sampler = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler)
B = signal_sampler.B
f_np = signal_sampler.f[0].cpu().numpy()  # 20..1024 active grid
print("Sage: B=%d  f_ref=%s  df=%s  f_numel=%d" %
      (B, float(signal_sampler.f_ref if signal_sampler.f_ref.ndim == 0 else signal_sampler.f_ref[0]),
       float(signal_sampler.df), signal_sampler.f_numel))

# ------------------------------------------------------------------ grid
DF = 1.0 / 16.0
F_FINAL = 1024.0
F_LOWER = 20.0
F_REF = 20.0
DISTANCE = 1000.0
COA_PHASE = 0.0
NF = int(round(F_FINAL / DF)) + 1      # 16385
N = 2 * (NF - 1)                        # 32768
DT = 1.0 / (N * DF)                     # 1/2048
FULLF = np.arange(NF) * DF             # 0..1024
assert NF == 16385 and N == 32768, (NF, N)

# ------------------------------------------------------------------ cases
# theta layout for get_hphc:
# [m1,m2,s1x,s1y,s1z,s2x,s2y,s2z,dist,tc,coa_phase,incl,pol,ra,dec]
CASES = [
    dict(tag="control_aligned_faceon",  m1=35, m2=30, s1=(0,0,0),        s2=(0,0,0),        incl=0.0),
    dict(tag="aligned_faceon_spin",     m1=35, m2=30, s1=(0,0,0.5),      s2=(0,0,0.3),      incl=0.0),
    dict(tag="precess_q4_incl60",       m1=40, m2=10, s1=(0.6,0,0.0),    s2=(0,0,0),        incl=math.pi/3),
    dict(tag="precess_heavy_incl45",    m1=50, m2=45, s1=(0.5,0.2,0.4),  s2=(-0.3,0,0),     incl=math.pi/4),
    dict(tag="precess_q3.75_incl1.3",   m1=30, m2=8,  s1=(0.7,0,0.2),    s2=(0,0,0),        incl=1.3),
    dict(tag="precess_heavy_edgeon",    m1=45, m2=40, s1=(0.6,0.3,0.3),  s2=(0.4,-0.2,-0.1),incl=math.pi/2),
    dict(tag="mid_precess_incl0.5",     m1=20, m2=18, s1=(0.4,0,0.0),    s2=(0,0,0),        incl=0.5),
    dict(tag="lowmass_precess_incl1.0", m1=12, m2=7,  s1=(0.3,0,0.1),    s2=(0,0,0),        incl=1.0),
    dict(tag="OUTLIER_M90_edgeon_prec", m1=48, m2=42, s1=(0.7,0.2,0.3),  s2=(0.5,0,0),      incl=1.4),
    dict(tag="M90_bigxspin_edgeon",     m1=50, m2=40, s1=(0.8,0,0.0),    s2=(0,0,0),        incl=math.pi/2),
    dict(tag="lowmass_aligned_faceon",  m1=7,  m2=7,  s1=(0,0,0),        s2=(0,0,0),        incl=0.0),
    dict(tag="mc31_highprec_incl1.5",   m1=38, m2=34, s1=(0.6,0.4,0.5),  s2=(0,0,0),        incl=1.5),
]

# ------------------------------------------------------------------ Sage batch
theta = torch.zeros((B, 15), dtype=cfg.dtype)
for i, c in enumerate(CASES):
    theta[i, 0] = c["m1"]; theta[i, 1] = c["m2"]
    theta[i, 2], theta[i, 3], theta[i, 4] = c["s1"]
    theta[i, 5], theta[i, 6], theta[i, 7] = c["s2"]
    theta[i, 8] = DISTANCE
    # tc=6.0 => apply_tc shift _tc = tc + pad - 16 = 6+2-16 = -8.0 s = exactly
    # -16384 samples => Sage declared coalescence lands at sample N/2 = 16384,
    # matching the rolled PyCBC reference. Production path applies fd_taper (like LAL).
    theta[i, 9] = 6.0
    theta[i, 10] = COA_PHASE
    theta[i, 11] = c["incl"]
    theta[i, 12] = 0.0; theta[i, 13] = 0.0; theta[i, 14] = 0.0
# pad remaining rows with case 0 so batch matches self.B
for i in range(len(CASES), B):
    theta[i] = theta[0]

with torch.no_grad():
    # Production path: fd_taper (matches LAL low-f conditioning) + apply_tc(tc=6.0).
    hp_S, hc_S = signal_sampler.get_hphc(theta, reproduce_lal=False)
hp_S = hp_S.cpu().numpy(); hc_S = hc_S.cpu().numpy()
print("Sage hp shape:", hp_S.shape, "(expect (%d,%d))" % (B, NF))
assert hp_S.shape[1] == NF, hp_S.shape

# ------------------------------------------------------------------ helpers
def to_td(hp_fd, hc_fd, roll):
    """irfft FD; optionally roll by +N/2 to bring a sample-0 coalescence to N/2."""
    xp = np.fft.irfft(hp_fd, n=N)
    xc = np.fft.irfft(hc_fd, n=N)
    if roll:
        xp = np.roll(xp, N // 2)
        xc = np.roll(xc, N // 2)
    return xp, xc

# Both pipelines' declared coalescence sits at sample N/2 = 16384:
#   Sage: apply_tc(tc=6.0) places it there directly (no roll).
#   PyCBC: raw coalescence at sample 0, brought to N/2 by the +N/2 roll.
COAL_SAMPLE = N // 2

def env_hp(xp):
    return np.abs(hilbert(xp))          # analytic-signal envelope of plus

WIN = int(0.5 * 2048)                   # +/-0.5 s merger-centred window
WLO, WHI = COAL_SAMPLE - WIN, COAL_SAMPLE + WIN

def env_peak(xp):
    """Estimator (i): argmax of the analytic-signal envelope within +/-0.5 s."""
    e = env_hp(xp)
    p = WLO + int(e[WLO:WHI].argmax())
    return p, e

def xcorr_lag(eS, eP):
    """Estimator (ii): integer-sample lag maximising cross-correlation of the two
    analytic envelopes over the merger window. Positive lag => Sage later than PyCBC.
    Convention-independent whole-shape alignment of the merger."""
    a = eS[WLO:WHI].astype(np.float64); b = eP[WLO:WHI].astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    n = len(a)
    fa = np.fft.rfft(a, 2 * n); fb = np.fft.rfft(b, 2 * n)
    cc = np.fft.irfft(fa * np.conj(fb), 2 * n)
    cc = np.concatenate([cc[-(n - 1):], cc[:n]])   # lags -(n-1)..(n-1)
    lag = int(cc.argmax()) - (n - 1)
    return lag

# ------------------------------------------------------------------ per-case
from pycbc.waveform import get_fd_waveform

rows = []
overlays = {}
for i, c in enumerate(CASES):
    # PyCBC
    hpP, hcP = get_fd_waveform(
        approximant="IMRPhenomPv2",
        mass1=float(c["m1"]), mass2=float(c["m2"]),
        spin1x=float(c["s1"][0]), spin1y=float(c["s1"][1]), spin1z=float(c["s1"][2]),
        spin2x=float(c["s2"][0]), spin2y=float(c["s2"][1]), spin2z=float(c["s2"][2]),
        distance=DISTANCE, inclination=float(c["incl"]), coa_phase=COA_PHASE,
        f_ref=F_REF, delta_f=DF, f_lower=F_LOWER, f_final=F_FINAL)
    hpP = np.asarray(hpP.data); hcP = np.asarray(hcP.data)
    assert hpP.shape[0] == NF, hpP.shape

    xpP, _ = to_td(hpP, hcP, roll=True)             # pycbc: coalescence 0 -> N/2
    xpS, _ = to_td(hp_S[i], hc_S[i], roll=False)     # sage: apply_tc already at N/2

    p1P, e1P = env_peak(xpP)
    p1S, e1S = env_peak(xpS)
    lag = xcorr_lag(e1S, e1P)            # samples, +ve => Sage later

    dt_env = (p1S - p1P) * DT * 1e3       # ms  (estimator i: envelope argmax diff)
    dt_rob = lag * DT * 1e3               # ms  (estimator ii: envelope xcorr lag)
    off_env_S = (p1S - COAL_SAMPLE) * DT * 1e3   # Sage:  env peak - coalescence
    off_env_P = (p1P - COAL_SAMPLE) * DT * 1e3   # PyCBC: env peak - coalescence
    Mtot = c["m1"] + c["m2"]
    chip = math.hypot(c["s1"][0], c["s1"][1])
    row = dict(tag=c["tag"], m1=c["m1"], m2=c["m2"], Mtot=Mtot,
               incl=round(c["incl"], 3), chi1_inplane=round(chip, 3),
               s1z=c["s1"][2],
               delta_t_env_ms=round(dt_env, 4), delta_t_robust_ms=round(dt_rob, 4),
               off_env_S_ms=round(off_env_S, 3), off_env_P_ms=round(off_env_P, 3))
    rows.append(row)
    print(json.dumps(row))
    if c["tag"] in ("control_aligned_faceon", "OUTLIER_M90_edgeon_prec", "mc31_highprec_incl1.5"):
        overlays[c["tag"]] = (e1S, e1P, p1S, p1P)

# ------------------------------------------------------------------ summary
d_env = np.array([r["delta_t_env_ms"] for r in rows])
d_rob = np.array([r["delta_t_robust_ms"] for r in rows])
Mtot = np.array([r["Mtot"] for r in rows], float)
incl = np.array([r["incl"] for r in rows], float)
chip = np.array([r["chi1_inplane"] for r in rows], float)
max_abs = float(max(np.abs(d_env).max(), np.abs(d_rob).max()))

def corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])
# Use the per-case spread of BOTH estimators for the trend test.
absd = np.maximum(np.abs(d_env), np.abs(d_rob))
c_mass = corr(Mtot, absd); c_incl = corr(incl, absd); c_prec = corr(chip, absd)
# MISPLACES requires a real, systematic shift: tens of ms AND growing with a
# physical driver. A few-ms scatter with no trend is waveform-model reimplementation
# noise between Sage's and LAL's PhenomPv2, not a coalescence-reference bug.
grows = bool(max_abs > 10.0 and (c_mass > 0.5 or c_incl > 0.5 or c_prec > 0.5))
verdict = "SAGE_MISPLACES" if (max_abs > 10.0 and grows) else "SAGE_MATCHES_PYCBC"

print("\n=== SUMMARY ===")
print("max_abs_delta_t_ms:", max_abs)
print("delta_t_env  [ms]:", np.round(d_env, 3).tolist())
print("delta_t_rob  [ms]:", np.round(d_rob, 3).tolist())
print("corr(|d_rob|, Mtot)=%.2f  incl=%.2f  chi_inplane=%.2f" % (c_mass, c_incl, c_prec))
print("grows_with_mass_or_precession:", grows, " VERDICT:", verdict)

# ------------------------------------------------------------------ overlay plot
fig, axes = plt.subplots(len(overlays), 1, figsize=(12, 4 * len(overlays)))
if len(overlays) == 1: axes = [axes]
tms = (np.arange(N) - COAL_SAMPLE) * DT * 1e3
for ax, (tag, (eS, eP, pS, pP)) in zip(axes, overlays.items()):
    w = int(0.15 * 2048)  # +/-150 ms window
    lo, hi = COAL_SAMPLE - w, COAL_SAMPLE + w
    eSn = eS / eS.max(); ePn = eP / eP.max()
    ax.plot(tms[lo:hi], eSn[lo:hi], "C0", lw=1.6, label="Sage |h|")
    ax.plot(tms[lo:hi], ePn[lo:hi], "C3--", lw=1.6, label="PyCBC |h|")
    ax.axvline(0, color="k", ls=":", lw=1, label="coalescence (t=0)")
    ax.axvline((pS - COAL_SAMPLE) * DT * 1e3, color="C0", ls="-", lw=0.8)
    ax.axvline((pP - COAL_SAMPLE) * DT * 1e3, color="C3", ls="-", lw=0.8)
    ax.set_title("%s   delta_t_robust=%.3f ms" %
                 (tag, (pS - pP) * DT * 1e3))
    ax.set_xlabel("time - coalescence [ms]"); ax.set_ylabel("|h| (norm)")
    ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/sage_vs_pycbc_tc.png", dpi=120); plt.close()
print("saved", f"{OUT}/sage_vs_pycbc_tc.png")

result = dict(
    verdict=verdict,
    max_abs_delta_t_ms=round(max_abs, 4),
    grows_with_mass_or_precession=grows,
    corr_absdrob_Mtot=round(c_mass, 3), corr_absdrob_incl=round(c_incl, 3),
    corr_absdrob_chi_inplane=round(c_prec, 3),
    pycbc_fd_epoch_convention=(
        "get_fd_waveform->_lalsim_fd_waveform (pycbc/waveform/waveform.py:239-257) passes through "
        "lalsimulation.SimInspiralChooseFDWaveform's hp1.epoch. Empirically epoch=-16.0 s = -N*dt "
        "(N=32768,dt=1/2048), i.e. coalescence(abs t=0) sits at raw-irfft sample 0; to_timeseries "
        "sets start_time=epoch=-16.0 so sample_times run [-16,0). Sage reproduce_lal likewise "
        "declares coalescence at raw sample 0 => same reference, delta_t is the merger-location diff."),
    delta_t_by_case=rows,
)
with open(f"{OUT}/sage_vs_pycbc_tc.json", "w") as f:
    json.dump(result, f, indent=2)
print("=== DONE ===")
print(json.dumps({k: result[k] for k in
      ("verdict", "max_abs_delta_t_ms", "grows_with_mass_or_precession")}, indent=2))

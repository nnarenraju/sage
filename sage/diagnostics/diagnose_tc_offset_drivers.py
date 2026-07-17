#!/usr/bin/env python
"""Characterise what DRIVES the large RAW |h|-peak-vs-tc offsets.

Reuses run_diag.py / raw_vs_whitened_tc.py setup. Draws N>=512 signals, computes
per-signal RAW irfft |h|-peak-minus-tc (same FD->TD framing), correlates the
offset against: inclination, in-plane spin magnitude per body (precession),
total mass, mass ratio, chi_eff.
"""
import os, sys, json, math
import numpy as np
import torch

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
    detectors = ["H1", "L1"]; train_runs = ["O3b"]; do_point_estimate = ["tc", "mchirp"]
    autocast = False; class_balance = 0.5; clip_norm = 1.0; dropout = 0.0; num_epochs = 10
    training_iterations = 1; validation_iterations = 1

class O3bDataCFG:
    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [_SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors]
    validation_noise_files = [_SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3bCFG.detectors]
    sample_rate = 2048.0; noise_low_frequency_cutoff = 15.0; signal_low_frequency_cutoff = 20.0
    sample_length_in_s = 12.0; padding_length_in_s = 2.0

register_configs(BaseConfig(O3bCFG()), BaseDataConfig(O3bDataCFG()))
cfg, data_cfg = get_cfg(), get_data_cfg()
SR = data_cfg.sample_rate

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.dsp.whiten import FiducialWhitening

param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
signal_sampler = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler)
bounds = param_sampler.bounds
tc_col = signal_sampler.tc_col
pidx = param_sampler.param_index
print("param_index:", pidx)

whitener = FiducialWhitening()
S = signal_sampler.B

def fd_to_td_raw(hf):
    x_td = torch.fft.irfft(hf, dim=-1, norm="forward") * whitener.delta_f
    return whitener.remove_corrupted(x_td)  # (B,D,VALID)

def peak_time_of(td_np):
    combined = np.abs(td_np).sum(axis=1)   # (N,VALID)
    pk = combined.argmax(axis=1)
    return pk / SR, pk

N_TARGET = 640
nb = math.ceil(N_TARGET / S)

cols = {k: pidx[k] for k in ["inclination", "mass1", "mass2", "q", "mchirp",
                             "spin1x", "spin1y", "spin1z",
                             "spin2x", "spin2y", "spin2z",
                             "spin1_a", "spin2_a"]}

store = {k: [] for k in cols}
inj_tc_l, raw_pk_l = [], []
for i in range(nb):
    hf, targets, all_theta = signal_sampler(return_theta=True)  # (S,D,F) rescaled
    th = all_theta.cpu().numpy()
    for k, c in cols.items():
        store[k].append(th[:, c])
    inj_tc_l.append(th[:, tc_col])
    rtd = fd_to_td_raw(hf).cpu().numpy()
    rpt, rpk = peak_time_of(rtd)
    raw_pk_l.append(rpt)

for k in store:
    store[k] = np.concatenate(store[k])[:N_TARGET]
inj_tc = np.concatenate(inj_tc_l)[:N_TARGET]
raw_pk = np.concatenate(raw_pk_l)[:N_TARGET]
N = len(inj_tc)

raw_err = (raw_pk - inj_tc) * 1e3      # ms  (negative => peak precedes tc)

# derived predictors
inclination = store["inclination"]
m1, m2 = store["mass1"], store["mass2"]
mtot = m1 + m2
q = store["q"]
chi1z, chi2z = store["spin1z"], store["spin2z"]
chi_eff = (m1 * chi1z + m2 * chi2z) / mtot
s1_inplane = np.sqrt(store["spin1x"]**2 + store["spin1y"]**2)
s2_inplane = np.sqrt(store["spin2x"]**2 + store["spin2y"]**2)
# mass-weighted in-plane (precession) proxy chi_p-like: max over bodies of mass-weighted S_perp
Sp1 = m1**2 * s1_inplane
Sp2 = m2**2 * s2_inplane
chi_p_like = np.maximum(Sp1, Sp2) / (m1**2)   # normalised to primary
max_inplane = np.maximum(s1_inplane, s2_inplane)
sin_inc = np.sin(inclination)
abs_cos_inc = np.abs(np.cos(inclination))

def corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

# correlate absolute offset magnitude too (large offsets = big |err|)
abs_err = np.abs(raw_err)

predictors = {
    "inclination": inclination,
    "sin_inclination": sin_inc,
    "abs_cos_inclination": abs_cos_inc,
    "in_plane_spin1": s1_inplane,
    "in_plane_spin2": s2_inplane,
    "max_in_plane_spin": max_inplane,
    "chi_p_like": chi_p_like,
    "total_mass": mtot,
    "mass_ratio_q": q,
    "chi_eff": chi_eff,
    "chirp_mass": store["mchirp"],
}

corr_signed = {k: corr(v, raw_err) for k, v in predictors.items()}
corr_absoff = {k: corr(v, abs_err) for k, v in predictors.items()}
print("corr(predictor, signed raw_err):")
print(json.dumps(corr_signed, indent=2))
print("corr(predictor, |raw_err|):")
print(json.dumps(corr_absoff, indent=2))

# outlier analysis: signals with |offset| > 50 ms
big = np.abs(raw_err) > 50.0
n_big = int(big.sum())
def summ(mask, name):
    if mask.sum() == 0:
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "inclination_deg_median": float(np.degrees(np.median(inclination[mask]))),
        "inclination_deg_mean": float(np.degrees(np.mean(inclination[mask]))),
        "abs_cos_inc_median": float(np.median(abs_cos_inc[mask])),
        "max_in_plane_spin_median": float(np.median(max_inplane[mask])),
        "chi_p_like_median": float(np.median(chi_p_like[mask])),
        "total_mass_median": float(np.median(mtot[mask])),
        "q_median": float(np.median(q[mask])),
        "chi_eff_median": float(np.median(chi_eff[mask])),
        "raw_err_ms_median": float(np.median(raw_err[mask])),
    }
outlier_summary = {
    "threshold_ms": 50.0,
    "n_big_offset": n_big,
    "frac_big": float(big.mean()),
    "big_offset_signals": summ(big, "big"),
    "small_offset_signals": summ(~big, "small"),
    "all_signals": summ(np.ones(N, bool), "all"),
}
print("outlier summary:")
print(json.dumps(outlier_summary, indent=2))

# ------- plot: offset vs inclination and vs in-plane spin, coloured by Mtot -------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
sc0 = ax[0].scatter(np.degrees(inclination), raw_err, c=mtot, s=16, alpha=0.7, cmap="viridis")
ax[0].axhline(0, color="k", lw=1)
ax[0].axhline(-50, color="r", ls=":", lw=1); ax[0].axhline(50, color="r", ls=":", lw=1)
ax[0].set_xlabel("inclination [deg]"); ax[0].set_ylabel("raw |h|-peak - tc [ms]")
ax[0].set_title(f"offset vs inclination (r={corr_signed['inclination']:.2f}, "
                f"|off| vs |cos i| r={corr_absoff['abs_cos_inclination']:.2f})")
cb0 = fig.colorbar(sc0, ax=ax[0]); cb0.set_label("total mass [Msun]")

sc1 = ax[1].scatter(max_inplane, raw_err, c=mtot, s=16, alpha=0.7, cmap="viridis")
ax[1].axhline(0, color="k", lw=1)
ax[1].axhline(-50, color="r", ls=":", lw=1); ax[1].axhline(50, color="r", ls=":", lw=1)
ax[1].set_xlabel("max in-plane spin magnitude sqrt(sx^2+sy^2)")
ax[1].set_ylabel("raw |h|-peak - tc [ms]")
ax[1].set_title(f"offset vs precession (r={corr_signed['max_in_plane_spin']:.2f}, "
                f"|off| r={corr_absoff['max_in_plane_spin']:.2f})")
cb1 = fig.colorbar(sc1, ax=ax[1]); cb1.set_label("total mass [Msun]")
plt.tight_layout(); plt.savefig(f"{OUT}/offset_drivers.png", dpi=120); plt.close()
print("saved", f"{OUT}/offset_drivers.png")

# rank |off| drivers
ranked = sorted(corr_absoff.items(), key=lambda kv: -abs(kv[1]))
print("ranked |off| drivers:", ranked)

# tail-widening test: spread (std) of offset in low vs high precession / inclination terciles
def tercile_spread(x, err):
    lo_thr, hi_thr = np.percentile(x, [33.3, 66.6])
    lo = err[x <= lo_thr]; hi = err[x >= hi_thr]
    return {"low_std_ms": float(np.std(lo)), "high_std_ms": float(np.std(hi)),
            "low_frac_big": float((np.abs(lo) > 50).mean()),
            "high_frac_big": float((np.abs(hi) > 50).mean()),
            "low_absmax_ms": float(np.abs(lo).max()), "high_absmax_ms": float(np.abs(hi).max())}
tail_test = {
    "chi_p_like": tercile_spread(chi_p_like, raw_err),
    "max_in_plane_spin": tercile_spread(max_inplane, raw_err),
    "abs_cos_inclination_EDGEon_is_low": tercile_spread(abs_cos_inc, raw_err),
    "total_mass": tercile_spread(mtot, raw_err),
}
print("tail-widening (std of offset, low vs high tercile):")
print(json.dumps(tail_test, indent=2))

result = {
    "N": N,
    "raw_err_ms_median": float(np.median(raw_err)),
    "raw_err_ms_std": float(np.std(raw_err)),
    "raw_err_ms_absmax": float(np.abs(raw_err).max()),
    "frac_within_20ms": float((np.abs(raw_err) < 20).mean()),
    "corr_signed_raw_err": corr_signed,
    "corr_abs_offset": corr_absoff,
    "ranked_abs_offset_drivers": [[k, v] for k, v in ranked],
    "outlier_summary": outlier_summary,
    "tail_widening_tercile": tail_test,
}
with open(f"{OUT}/offset_drivers.json", "w") as f:
    json.dump(result, f, indent=2)
print("=== RESULT ===")
print(json.dumps(result, indent=2))

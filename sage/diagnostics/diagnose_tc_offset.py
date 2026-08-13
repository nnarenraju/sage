#!/usr/bin/env python
"""Decide whether the -9ms whitened-peak offset is a tc PLACEMENT bug or benign.

Reuses the working run_diag.py setup. For N>=256 signals compares:
  raw irfft-peak time  vs  whitened-peak time  vs  injected tc.
Same FD->TD framing (irfft norm='forward'*delta_f, strip corrupted_len) for both.
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
    fiducial_dir = "/work/nagarajan/sage_runs/fiducial_psds_o3ab"
    batch_size = 128
    device = "cpu"
    dtype = _t.float32
    detectors = ["H1", "L1"]
    train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = False
    class_balance = 0.5
    clip_norm = 1.0
    dropout = 0.0
    num_epochs = 10
    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)

class O3bDataCFG:
    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [_SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors]
    validation_noise_files = [_SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3bCFG.detectors]
    sample_rate = 2048.0
    noise_low_frequency_cutoff = 15.0
    signal_low_frequency_cutoff = 20.0
    sample_length_in_s = 12.0
    padding_length_in_s = 2.0

register_configs(BaseConfig(O3bCFG()), BaseDataConfig(O3bDataCFG()))
cfg, data_cfg = get_cfg(), get_data_cfg()
SR = data_cfg.sample_rate
CORRUPT = data_cfg.padding_nsamples
VALID = data_cfg.padded_length_in_nsamples - 2 * CORRUPT
print(f"padded={data_cfg.padded_length_in_nsamples} corrupt={CORRUPT} valid={VALID} ({VALID/SR}s) SR={SR}")

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning

param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
signal_sampler = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler)
bounds = param_sampler.bounds
tc_col = signal_sampler.tc_col
pidx = param_sampler.param_index
print("param_index:", pidx)
mchirp_col = pidx["mchirp"]
print("tc_col:", tc_col, "mchirp_col:", mchirp_col)

whitener = FiducialWhitening()
dyadic = DyadicPyramidBinning(bounds)
mrsampler = MultirateSampler(binning_method=dyadic)

# full-rate (unchanged) region of the multirate window in the valid frame
full_bins = [b for b in dyadic.detailed_bins if int(b[2]) == int(SR)]
print("full-rate region(s) [s]:", [(b[0]/SR, b[1]/SR) for b in full_bins])
mr_lo_edge = min(b[0] for b in full_bins) / SR
mr_hi_edge = max(b[1] for b in full_bins) / SR
print(f"full-rate window edges: [{mr_lo_edge:.4f}, {mr_hi_edge:.4f}] s")

S = signal_sampler.B
DF = float(whitener.delta_f.item())
WH = whitener.whitening  # (D,F)

def fd_to_td_raw(hf):
    """Replicate whitener framing but WITHOUT the whitening kernel."""
    x_td = torch.fft.irfft(hf, dim=-1, norm="forward") * whitener.delta_f
    return whitener.remove_corrupted(x_td)  # (B,D,VALID)

def peak_time_of(td_np):
    """|strain| summed over detectors, argmax -> seconds in valid frame."""
    combined = np.abs(td_np).sum(axis=1)   # (N,VALID)
    pk = combined.argmax(axis=1)
    return pk / SR, pk

N_TARGET = 320
nb = math.ceil(N_TARGET / S)
inj_tc_l, mchirp_l = [], []
raw_pk_l, wh_pk_l = [], []
raw_env_pk_l = []
example = None
for i in range(nb):
    hf, targets, all_theta = signal_sampler(return_theta=True)  # (S,D,F) rescaled
    inj_tc_l.append(all_theta[:, tc_col].cpu().numpy())
    mchirp_l.append(all_theta[:, mchirp_col].cpu().numpy())
    # whitened
    wtd = whitener(hf).cpu().numpy()
    wpt, wpk = peak_time_of(wtd)
    wh_pk_l.append(wpt)
    # raw (no whitening), same framing
    rtd = fd_to_td_raw(hf).cpu().numpy()
    rpt, rpk = peak_time_of(rtd)
    raw_pk_l.append(rpt)
    # raw analytic-envelope peak (robustness check, per-detector-summed Hilbert env)
    an = np.abs(np.fft.ifft(np.fft.fft(rtd, axis=-1) *
         (np.arange(rtd.shape[-1])[None, None, :] < rtd.shape[-1] / 2) * 2, axis=-1))
    ept, _ = peak_time_of(an)
    raw_env_pk_l.append(ept)
    if example is None:
        example = (rtd[0], wtd[0], float(all_theta[0, tc_col]), rpk[0], wpk[0])

inj_tc = np.concatenate(inj_tc_l)[:N_TARGET]
mchirp = np.concatenate(mchirp_l)[:N_TARGET]
raw_pk = np.concatenate(raw_pk_l)[:N_TARGET]
wh_pk = np.concatenate(wh_pk_l)[:N_TARGET]
raw_env_pk = np.concatenate(raw_env_pk_l)[:N_TARGET]
N = len(inj_tc)

raw_err = (raw_pk - inj_tc) * 1e3      # ms
wh_err = (wh_pk - inj_tc) * 1e3        # ms
raw_env_err = (raw_env_pk - inj_tc) * 1e3

def stats(a):
    return dict(median=float(np.median(a)), std=float(np.std(a)),
                absmax=float(np.abs(a).max()), mean=float(np.mean(a)),
                frac_within_20ms=float((np.abs(a) < 20).mean()))

raw_s = stats(raw_err); wh_s = stats(wh_err); env_s = stats(raw_env_err)
print("RAW  peak-tc [ms]:", json.dumps(raw_s, indent=2))
print("RAWENV peak-tc [ms]:", json.dumps(env_s, indent=2))
print("WHIT peak-tc [ms]:", json.dumps(wh_s, indent=2))

# mass correlation
def corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
raw_mass_corr = corr(mchirp, raw_err)
wh_mass_corr = corr(mchirp, wh_err)
print(f"corr(mchirp, raw_err)={raw_mass_corr:.3f}  corr(mchirp, wh_err)={wh_mass_corr:.3f}")

# MR margin
min_raw_peak = float(raw_pk.min()); min_wh_peak = float(wh_pk.min())
margin_raw_ms = (min_raw_peak - mr_lo_edge) * 1e3
margin_wh_ms = (min_wh_peak - mr_lo_edge) * 1e3
n_raw_outside = int(((raw_pk < mr_lo_edge) | (raw_pk > mr_hi_edge)).sum())
n_wh_outside = int(((wh_pk < mr_lo_edge) | (wh_pk > mr_hi_edge)).sum())
print(f"MR margin: min_raw_peak={min_raw_peak:.4f}s margin={margin_raw_ms:.1f}ms outside={n_raw_outside}")
print(f"MR margin: min_wh_peak ={min_wh_peak:.4f}s margin={margin_wh_ms:.1f}ms outside={n_wh_outside}")

# ---- plots ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(inj_tc, raw_pk, s=9, alpha=0.5, c="C0", label="raw irfft peak")
ax[0].scatter(inj_tc, wh_pk, s=9, alpha=0.5, c="C3", label="whitened peak")
lo, hi = inj_tc.min()-0.01, inj_tc.max()+0.01
ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
ax[0].axhline(mr_lo_edge, color="gray", ls=":", label=f"MR low edge {mr_lo_edge:.3f}s")
ax[0].set_xlabel("injected tc [s]"); ax[0].set_ylabel("found peak time [s]")
ax[0].set_title("raw vs whitened peak vs inj_tc"); ax[0].legend(fontsize=8)
bins = np.linspace(min(raw_err.min(), wh_err.min())-2, max(raw_err.max(), wh_err.max())+2, 60)
ax[1].hist(raw_err, bins=bins, alpha=0.6, color="C0",
           label=f"raw: med={raw_s['median']:.2f} std={raw_s['std']:.2f}ms")
ax[1].hist(wh_err, bins=bins, alpha=0.6, color="C3",
           label=f"whit: med={wh_s['median']:.2f} std={wh_s['std']:.2f}ms")
ax[1].axvline(0, color="k", lw=1)
ax[1].set_xlabel("peak - inj_tc [ms]"); ax[1].set_ylabel("count")
ax[1].set_title("timing offset distribution"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/raw_vs_whitened_tc.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(mchirp, raw_err, s=9, alpha=0.5, c="C0")
ax[0].axhline(0, color="k", lw=1)
ax[0].set_xlabel("chirp mass [Msun]"); ax[0].set_ylabel("raw peak - tc [ms]")
ax[0].set_title(f"RAW offset vs mass (r={raw_mass_corr:.2f})")
ax[1].scatter(mchirp, wh_err, s=9, alpha=0.5, c="C3")
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xlabel("chirp mass [Msun]"); ax[1].set_ylabel("whitened peak - tc [ms]")
ax[1].set_title(f"WHITENED offset vs mass (r={wh_mass_corr:.2f})")
plt.tight_layout(); plt.savefig(f"{OUT}/offset_vs_mass.png", dpi=110); plt.close()

result = {
    "N": N,
    "raw_peak_minus_tc_ms": raw_s,
    "raw_envelope_peak_minus_tc_ms": env_s,
    "whitened_peak_minus_tc_ms": wh_s,
    "corr_mchirp_raw_err": raw_mass_corr,
    "corr_mchirp_wh_err": wh_mass_corr,
    "offset_samples_raw_median": raw_s["median"] / 1e3 * SR,
    "offset_samples_wh_median": wh_s["median"] / 1e3 * SR,
    "mr_full_rate_edges_s": [mr_lo_edge, mr_hi_edge],
    "min_raw_peak_s": min_raw_peak, "margin_raw_ms": margin_raw_ms, "n_raw_outside": n_raw_outside,
    "min_wh_peak_s": min_wh_peak, "margin_wh_ms": margin_wh_ms, "n_wh_outside": n_wh_outside,
}
with open(f"{OUT}/raw_vs_whitened_tc.json", "w") as f:
    json.dump(result, f, indent=2)
print("=== RESULT ===")
print(json.dumps(result, indent=2))

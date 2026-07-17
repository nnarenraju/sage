#!/usr/bin/env python
"""CPU many-sample diagnostic for the o3b dummy training regression.

Mirrors runs/o3b/train.py's make_training_graph / make_processor exactly,
but forces device='cpu' and scales to many samples. Produces:
  1. snr_distribution.png   -- realised optimal SNR vs HalfNorm(4,5) target
  2. tc_placement.png       -- injected tc vs found whitened-strain peak time
  3. mr_alignment.png       -- pre/post multirate signals around tc
  4. amplitude_stats.png    -- raw noise / whitened noise / whitened signal stats
"""
import os, sys, json, math
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/nagarajan/research/sage/sage/diagnostics/plots"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- config (CPU)
import torch as _t
from sage.core.config import register_configs, get_cfg, get_data_cfg
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()

class O3bCFG:
    export_dir = "/home/nagarajan/research/sage/sage/diagnostics/plots"
    fiducial_dir = "/home/nagarajan/research/sage/runs/o3b/run_export/fiducial_psds"
    batch_size = 128
    device = "cpu"                 # FORCED CPU
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
print("device:", cfg.device, "| detectors:", cfg.detectors)
SR = data_cfg.sample_rate
CORRUPT = data_cfg.padding_nsamples          # samples stripped each side by whitener
VALID = data_cfg.padded_length_in_nsamples - 2 * CORRUPT
print(f"padded_nsamples={data_cfg.padded_length_in_nsamples} corrupt={CORRUPT} valid={VALID} ({VALID/SR}s)")

# ---------------------------------------------------------------- build graph
from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler, OptimalSNREstimator
from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess, MemmapSingleNoiseSampler
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning

param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
signal_sampler = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler)
bounds = param_sampler.bounds
tc_col = signal_sampler.tc_col

# NOTE: RecolourPostprocess loads a ~16 GB/detector ASD bank into RAM (~32 GB),
# which OOM-kills the CPU login node. Recolour only affects the NOISE colouring
# (37% of samples) and is irrelevant to the signal-side diagnostics (SNR, tc,
# MR). For the whitened-noise amplitude check we use the plain FD path
# (postprocess_fn=None) -> real O3b noise -> rfft -> fiducial whitening.
noise_sampler = MemmapNoiseSampler(postprocess_fn=None, prefetch=4, seed=150914)
RECOLOUR_SKIPPED = "recolour skipped on login node (32GB ASD bank OOM); plain O3b noise used for whitened-noise check"

whitener = FiducialWhitening()
dyadic = DyadicPyramidBinning(bounds)
mrsampler = MultirateSampler(binning_method=dyadic)
snr_estimator = OptimalSNREstimator()   # standalone, same fiducial PSDs

print("tc prior bounds:", bounds["tc"], "| mass1 bounds:", bounds["mass1"])
print("MR bins (start,end,fs):")
for b in dyadic.detailed_bins:
    print(f"   [{b[0]:6d} {b[1]:6d}] fs={b[2]:5d}  t=[{b[0]/SR:.3f},{b[1]/SR:.3f}]s")

# unchanged (full-rate) bin(s) in the valid whitened frame:
full_bins = [b for b in dyadic.detailed_bins if int(b[2]) == int(SR)]
print("full-rate region(s):", [(b[0]/SR, b[1]/SR) for b in full_bins])

S = signal_sampler.B   # signals per forward (class_balance*batch = 64)

# ================================================================ 1. SNR dist
N_SNR = 640
realised, target_overlay = [], []
per_det = []
theta_store = []
hf_store = []
nb = math.ceil(N_SNR / S)
tgt_sampler = HalfNorm(scale=4.0, loc=5.0, seed=999)
for i in range(nb):
    hf, targets, all_theta = signal_sampler(return_theta=True)   # hf: (S,D,F) rescaled
    rho_net, rho_det = snr_estimator(hf)
    realised.append(rho_net.detach().cpu().numpy().ravel())
    per_det.append(rho_det.detach().cpu().numpy().reshape(hf.shape[0], -1))
    target_overlay.append(tgt_sampler(hf.shape[0]).detach().cpu().numpy().ravel())
    if i < 4:
        theta_store.append(all_theta.detach().cpu())
        hf_store.append(hf.detach().cpu())
realised = np.concatenate(realised)[:N_SNR]
per_det = np.concatenate(per_det, axis=0)[:N_SNR]
target_overlay = np.concatenate(target_overlay)[:N_SNR]

def pcts(a): return {p: float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
snr_summary = {
    "realised_mean": float(realised.mean()), "realised_median": float(np.median(realised)),
    "realised_min": float(realised.min()), "realised_max": float(realised.max()),
    "realised_pcts": pcts(realised),
    "target_mean": float(target_overlay.mean()), "target_median": float(np.median(target_overlay)),
    "target_pcts": pcts(target_overlay),
    "n_snr_below_0.5": int((realised < 0.5).sum()),
    "n_snr_above_50": int((realised > 50).sum()),
    "n_nan": int(np.isnan(realised).sum()),
    "abs_diff_realised_vs_target_median": float(abs(np.median(realised) - np.median(target_overlay))),
    "n": int(N_SNR),
}
print("SNR summary:", json.dumps(snr_summary, indent=2))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
bins = np.linspace(0, max(realised.max(), target_overlay.max()) + 1, 60)
ax[0].hist(realised, bins=bins, alpha=0.6, label=f"realised optimal SNR (n={N_SNR})", color="C0")
ax[0].hist(target_overlay, bins=bins, alpha=0.5, label="target HalfNorm(4,5)", color="C1", histtype="step", lw=2)
ax[0].axvline(np.median(realised), color="C0", ls="--", label=f"realised med={np.median(realised):.2f}")
ax[0].axvline(np.median(target_overlay), color="C1", ls=":", label=f"target med={np.median(target_overlay):.2f}")
ax[0].set_xlabel("network optimal SNR"); ax[0].set_ylabel("count"); ax[0].legend(); ax[0].set_title("SNR: realised vs target")
qs = np.linspace(0, 100, 200)
ax[1].plot(np.percentile(target_overlay, qs), np.percentile(realised, qs), "C0")
lim = [0, max(realised.max(), target_overlay.max())]
ax[1].plot(lim, lim, "k--", lw=1)
ax[1].set_xlabel("target quantile"); ax[1].set_ylabel("realised quantile"); ax[1].set_title("Q-Q realised vs target")
plt.tight_layout(); plt.savefig(f"{OUT}/snr_distribution.png", dpi=110); plt.close()

# ================================================================ 2/3. tc + MR
theta_all = torch.cat(theta_store, 0)
hf_all = torch.cat(hf_store, 0)
N_TC = min(256, hf_all.shape[0])
hf_all = hf_all[:N_TC]; theta_all = theta_all[:N_TC]
inj_tc = theta_all[:, tc_col].numpy()          # seconds in the 12s valid frame

# whiten signal-only -> TD (N,D,VALID)
wsig = whitener(hf_all)                          # raw-tensor path -> TD float32
wsig_np = wsig.detach().cpu().numpy()
D = wsig_np.shape[1]

# peak (merger) location per sample: use detector-summed |strain| for robustness
absw = np.abs(wsig_np)
combined = absw.sum(axis=1)                      # (N, VALID)
peak_idx = combined.argmax(axis=1)
peak_time = peak_idx / SR                        # seconds in valid frame
tc_err = peak_time - inj_tc
tc_match_frac = float((np.abs(tc_err) < 0.02).sum() / N_TC)   # within 20 ms

tc_summary = {
    "n": int(N_TC),
    "inj_tc_range": [float(inj_tc.min()), float(inj_tc.max())],
    "peak_time_range": [float(peak_time.min()), float(peak_time.max())],
    "tc_err_mean_s": float(tc_err.mean()), "tc_err_median_s": float(np.median(tc_err)),
    "tc_err_std_s": float(tc_err.std()), "tc_err_absmax_s": float(np.abs(tc_err).max()),
    "tc_match_frac_within_20ms": tc_match_frac,
}
print("tc summary:", json.dumps(tc_summary, indent=2))

# --- multirate: does the merger peak survive? ---
mr = mrsampler(wsig)                             # (N,D,Lc) raw-tensor path
mr_np = mr.detach().cpu().numpy()
Lc = mr_np.shape[-1]

# peak amplitude before vs after (per detector, per sample)
peak_before = absw.max(axis=2)                   # (N,D)
peak_after = np.abs(mr_np).max(axis=2)           # (N,D)
peak_ratio = peak_after / (peak_before + 1e-30)
# energy in a +/-50 ms window around the found merger, before vs after.
win = int(0.05 * SR)
e_before, e_after = [], []
# locate the full-rate (unchanged) region in the compressed array:
# bins are laid out in order; compressed length of each bin = width/(SR/fs).
bins = dyadic.detailed_bins
comp_offsets = []
acc = 0
for b in bins:
    w = int(b[1] - b[0]); f = int(b[2])
    clen = w // int(SR // f)
    comp_offsets.append((acc, acc + clen, b))
    acc += clen
full_comp = [(s, e) for (s, e, b) in comp_offsets if int(b[2]) == int(SR)]
for n in range(N_TC):
    p = peak_idx[n]
    lo, hi = max(0, p - win), min(combined.shape[1], p + win)
    e_before.append(float((combined[n, lo:hi] ** 2).sum()))
    # after: search within full-rate compressed slice(s)
    seg = []
    for (s, e) in full_comp:
        seg.append(np.abs(mr_np[n, :, s:e]).sum(axis=0))
    seg = np.concatenate(seg) if seg else np.abs(mr_np[n]).sum(axis=0)
    pj = seg.argmax()
    lo2, hi2 = max(0, pj - win), min(seg.shape[0], pj + win)
    e_after.append(float((seg[lo2:hi2] ** 2).sum()))
e_before = np.array(e_before); e_after = np.array(e_after)
energy_ratio = e_after / (e_before + 1e-30)
mr_survival_frac = float((peak_ratio.min(axis=1) > 0.5).sum() / N_TC)

mr_summary = {
    "n": int(N_TC),
    "compressed_len": int(Lc), "valid_len": int(VALID),
    "compression_factor": float(VALID / Lc),
    "peak_ratio_after_over_before_mean": float(peak_ratio.mean()),
    "peak_ratio_median": float(np.median(peak_ratio)),
    "peak_ratio_min": float(peak_ratio.min()),
    "energy_ratio_window_median": float(np.median(energy_ratio)),
    "energy_ratio_window_mean": float(energy_ratio.mean()),
    "mr_survival_frac_peak_ratio_gt_0.5": mr_survival_frac,
    "full_rate_region_s": [[float(b[0]/SR), float(b[1]/SR)] for (_, _, b) in comp_offsets if int(b[2]) == int(SR)],
}
print("MR summary:", json.dumps(mr_summary, indent=2))

# tc scatter plot
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(inj_tc, peak_time, s=8, alpha=0.5)
lo, hi = inj_tc.min() - 0.01, inj_tc.max() + 0.01
ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
ax[0].set_xlabel("injected tc [s]"); ax[0].set_ylabel("found whitened peak time [s]")
ax[0].set_title(f"tc placement (match<20ms={tc_match_frac*100:.1f}%)"); ax[0].legend()
ax[1].hist(tc_err * 1e3, bins=50)
ax[1].set_xlabel("peak_time - inj_tc [ms]"); ax[1].set_ylabel("count")
ax[1].set_title(f"tc error: med={np.median(tc_err)*1e3:.2f}ms std={tc_err.std()*1e3:.2f}ms")
plt.tight_layout(); plt.savefig(f"{OUT}/tc_placement.png", dpi=110); plt.close()

# MR alignment overlays (a few examples, detector 0, around tc)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
for k, n in enumerate([0, 1, 2]):
    p = peak_idx[n]
    lo, hi = max(0, p - 300), min(VALID, p + 300)
    t_before = np.arange(lo, hi) / SR
    axes[k].plot(t_before, wsig_np[n, 0, lo:hi], "C0", lw=1.2, label="pre-MR whitened (2048Hz)")
    # overlay compressed full-rate region on its own sample axis mapped to time
    for (s, e, b) in comp_offsets:
        if int(b[2]) != int(SR):
            continue
        # this full-rate bin maps compressed idx s..e to original samples b[0]..b[1]
        orig = np.arange(int(b[0]), int(b[1]))
        m = (orig >= lo) & (orig < hi)
        if m.sum():
            axes[k].plot(orig[m] / SR, mr_np[n, 0, s:e][m], "C3--", lw=1.0,
                         label="post-MR (full-rate copy)" if k == 0 else None)
    axes[k].axvline(inj_tc[n], color="k", ls=":", label="injected tc")
    axes[k].set_ylabel("whitened strain"); axes[k].legend(loc="upper left", fontsize=8)
    axes[k].set_title(f"sample {n}: inj_tc={inj_tc[n]:.4f}s peak={peak_time[n]:.4f}s "
                      f"peak_ratio(det0)={peak_ratio[n,0]:.3f}")
axes[-1].set_xlabel("time [s] (valid 12s frame)")
plt.tight_layout(); plt.savefig(f"{OUT}/mr_alignment.png", dpi=110); plt.close()

# ================================================================ 4. amplitudes
# raw TD noise (~1e-20) via single sampler
raw_std, raw_max = [], []
single = MemmapSingleNoiseSampler(_SRV.noise_bin("H1", "O3b"), return_tensor=True)
for _ in range(200):
    seg = single(int(data_cfg.padded_length_in_nsamples)).numpy().ravel()
    raw_std.append(seg.std()); raw_max.append(np.abs(seg).max())
raw_std = np.array(raw_std); raw_max = np.array(raw_max)

# whitened noise std (from real noise sampler FD -> whiten -> TD)
wn_std, wn_max = [], []
for _ in range(8):
    fd_noise, _ = noise_sampler()                # (B,D,F) complex FD
    wtd = whitener(fd_noise).detach().cpu().numpy()   # (B,D,VALID)
    wn_std.append(wtd.std(axis=2).ravel())
    wn_max.append(np.abs(wtd).max(axis=2).ravel())
wn_std = np.concatenate(wn_std); wn_max = np.concatenate(wn_max)

# whitened signal amplitude (per-sample max)
ws_max = np.abs(wsig_np).max(axis=2).ravel()
ws_std = wsig_np.std(axis=2).ravel()

amp_summary = {
    "raw_td_noise_std_median": float(np.median(raw_std)),
    "raw_td_noise_absmax_median": float(np.median(raw_max)),
    "whitened_noise_std_median": float(np.median(wn_std)),
    "whitened_noise_std_mean": float(wn_std.mean()),
    "whitened_noise_absmax_median": float(np.median(wn_max)),
    "whitened_signal_absmax_median": float(np.median(ws_max)),
    "whitened_signal_std_median": float(np.median(ws_std)),
}
print("amplitude summary:", json.dumps(amp_summary, indent=2))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hist(raw_std, bins=40, alpha=0.7, label="std")
ax[0].hist(raw_max, bins=40, alpha=0.5, label="|max|")
ax[0].set_title(f"raw TD noise (med std={np.median(raw_std):.2e})"); ax[0].legend(); ax[0].set_xlabel("strain")
ax[1].hist(wn_std, bins=40, alpha=0.7, label="std")
ax[1].hist(wn_max, bins=40, alpha=0.5, label="|max|")
ax[1].axvline(1.0, color="k", ls="--", label="unit std")
ax[1].set_title(f"whitened noise (med std={np.median(wn_std):.3f})"); ax[1].legend(); ax[1].set_xlabel("whitened amp")
ax[2].hist(ws_max, bins=40, alpha=0.7, label="|max|")
ax[2].hist(ws_std, bins=40, alpha=0.5, label="std")
ax[2].set_title(f"whitened signal (med |max|={np.median(ws_max):.2f})"); ax[2].legend(); ax[2].set_xlabel("whitened amp")
plt.tight_layout(); plt.savefig(f"{OUT}/amplitude_stats.png", dpi=110); plt.close()

noise_sampler.shutdown()

result = {"snr": snr_summary, "tc": tc_summary, "mr": mr_summary, "amp": amp_summary,
          "notes": RECOLOUR_SKIPPED,
          "pngs": [f"{OUT}/snr_distribution.png", f"{OUT}/tc_placement.png",
                   f"{OUT}/mr_alignment.png", f"{OUT}/amplitude_stats.png"]}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(result, f, indent=2)
print("=== DONE ===")
print(json.dumps(result, indent=2))

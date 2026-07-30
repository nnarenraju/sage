#!/usr/bin/env python
"""SNR provenance diagnostic: ORIGINAL SNR -> rescaled SNR -> whitened-frame
effective SNR, for the *current* o3b HL pipeline, on CPU.

Answers the user's question directly:
  1. What is the network optimal SNR of the raw generated signals (BEFORE the
     HalfNorm rescaling)?  -> "original network SNR"
  2. What is it AFTER rescaling?  -> should sit exactly on HalfNorm(4,5).
  3. Does that labelled optimal SNR SURVIVE whitening + edge-crop + the real
     noise floor?  i.e. is the SNR the network can actually exploit
     (||whitened signal|| / sigma_whitened_noise) equal to the labelled optimal
     SNR, or is it "artificially low"?  Broken out per detector (H1 vs L1).

Outputs (PNG + JSON) to sage/diagnostics/plots_snr/ (home dir, never /tmp).
"""
import os, sys, json
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/nagarajan/research/sage/sage/diagnostics/plots_snr"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- config (CPU)
from sage.core.config import register_configs, get_cfg, get_data_cfg
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()

class O3bCFG:
    export_dir = OUT
    fiducial_dir = "/home/nagarajan/research/sage/runs/o3b/run_export/fiducial_psds"
    batch_size = 128
    device = "cpu"
    dtype = torch.float32
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
PADDED = data_cfg.padded_length_in_nsamples
VALID = PADDED - 2 * CORRUPT
DETS = cfg.detectors
print(f"device={cfg.device} dets={DETS}")
print(f"padded_nsamples={PADDED} ({PADDED/SR:.3f}s) corrupt={CORRUPT} valid={VALID} ({VALID/SR:.3f}s)")
print(f"padded_delta_f={data_cfg.padded_delta_f:.6f}  unpadded delta_f={data_cfg.delta_f:.6f}")

# ---------------------------------------------------------------- build graph
from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler, OptimalSNREstimator
from sage.data.noise import MemmapNoiseSampler
from sage.dsp.whiten import FiducialWhitening

param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
# Build the signal sampler WITHOUT augment so we can measure the ORIGINAL SNR,
# then apply the exact same rescaler ourselves to get the AFTER SNR + scale.
signal_sampler = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=None)
estimator = OptimalSNREstimator()
rescaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
tgt_sampler = HalfNorm(scale=4.0, loc=5.0, seed=2718)
whitener = FiducialWhitening()
S = signal_sampler.B
print(f"signals per forward = {S}")

# ================================================================ noise sigma
# whitened-noise per-sample std per detector (real O3b noise, fiducial whiten).
# This is the noise floor the labelled optimal SNR competes against.
noise_sampler = MemmapNoiseSampler(postprocess_fn=None, prefetch=4, seed=150914)
nsig = {d: [] for d in DETS}
for _ in range(24):
    fd_noise, _ = noise_sampler()                 # (B,D,F) complex
    wtd = whitener(fd_noise).detach().cpu().numpy()   # (B,D,VALID)
    for di, d in enumerate(DETS):
        nsig[d].append(wtd[:, di, :].std(axis=1))
noise_sampler.shutdown()
sigma = {d: np.concatenate(v) for d, v in nsig.items()}
sigma_med = {d: float(np.median(sigma[d])) for d in DETS}
print("whitened-noise sigma (median):", {d: round(sigma_med[d], 4) for d in DETS})
# heavy tail = glitch-contaminated segments (whitened std blows up). Quantify per det.
sigma_tail = {d: {"max": float(sigma[d].max()),
                  "p99": float(np.percentile(sigma[d], 99)),
                  "frac_gt_2x_ideal": float((sigma[d] > 2 * 0.3535528).mean())}
              for d in DETS}
print("whitened-noise tail:", json.dumps(sigma_tail))

# ================================================================ signal pass
N = 768
rho_orig, rho_post = [], []                        # network SNR before/after rescale
rho_post_det = []                                  # (n, D) optimal per det, after rescale
tdnorm_det = []                                    # (n, D) ||whitened signal|| per det
target_ov = []
nb = (N + S - 1) // S
for i in range(nb):
    hf_raw, _tg = signal_sampler()                       # (S,D,F) complex, NOT rescaled
    r0, _ = estimator(hf_raw)
    hf_sc, scale = rescaler(hf_raw)                      # exact training rescale
    r1, r1d = estimator(hf_sc)
    wsig = whitener(hf_sc).detach().cpu().numpy()        # (S,D,VALID) whitened TD signal
    # matched-filter norm of the signal in the whitened frame the network sees:
    tdn = np.sqrt((wsig ** 2).sum(axis=2))               # (S,D)
    rho_orig.append(r0.detach().cpu().numpy().ravel())
    rho_post.append(r1.detach().cpu().numpy().ravel())
    rho_post_det.append(r1d.detach().cpu().numpy().reshape(hf_sc.shape[0], -1))
    tdnorm_det.append(tdn)
    target_ov.append(tgt_sampler(hf_sc.shape[0]).detach().cpu().numpy().ravel())

rho_orig = np.concatenate(rho_orig)[:N]
rho_post = np.concatenate(rho_post)[:N]
rho_post_det = np.concatenate(rho_post_det, 0)[:N]     # (N,D)
tdnorm_det = np.concatenate(tdnorm_det, 0)[:N]         # (N,D)
target_ov = np.concatenate(target_ov)[:N]

# Whitening constant c: ||wsig|| = c * rho_opt.  Recovered from the SIGNAL side.
# Crucially c is (empirically) detector-INDEPENDENT: with a perfectly-matched PSD
# the whitened-noise std is set only by delta_f + the norm convention, NOT by the
# detector -- so this same c is exactly the whitened-noise std we WOULD get if the
# fiducial PSD perfectly described the real noise (sigma_ideal). The real whitened
# noise sigma_det > c is line/glitch power the fixed fiducial doesn't model.
c_det = {d: float(np.median(tdnorm_det[:, di] / (rho_post_det[:, di] + 1e-12)))
         for di, d in enumerate(DETS)}
c_ideal = float(np.median(list(c_det.values())))   # matched-noise whitened std
# Effective (usable) matched-filter SNR in the whitened frame the network sees:
#   eff = ||whitened signal|| / sigma_real_noise  =  rho_opt * (c / sigma_real).
# Both numerator and denominator are in the same whitened units so c cancels;
# eff is already in optimal-SNR units (eff == rho_opt iff sigma_real == c).
eff_det = np.zeros_like(rho_post_det)
for di, d in enumerate(DETS):
    eff_det[:, di] = tdnorm_det[:, di] / sigma_med[d]
eff_net = np.sqrt((eff_det ** 2).sum(axis=1))
inflation = {d: round(sigma_med[d] / c_ideal, 4) for d in DETS}   # sigma_real / sigma_ideal
print("noise-floor inflation (sigma_real/sigma_ideal):", inflation)

def pcts(a): return {p: round(float(np.percentile(a, p)), 4) for p in (5, 25, 50, 75, 95)}
summary = {
    "config": {"padded_s": PADDED / SR, "valid_s": VALID / SR, "corrupt_s": CORRUPT / SR,
               "padded_delta_f": float(data_cfg.padded_delta_f)},
    "original_net_snr": {"median": float(np.median(rho_orig)), "pcts": pcts(rho_orig),
                         "min": float(rho_orig.min()), "max": float(rho_orig.max())},
    "rescaled_net_snr": {"median": float(np.median(rho_post)), "pcts": pcts(rho_post)},
    "target_halfnorm":  {"median": float(np.median(target_ov)), "pcts": pcts(target_ov)},
    "rescale_median_abs_err": float(abs(np.median(rho_post) - np.median(target_ov))),
    "whitened_noise_sigma_median": sigma_med,
    "whiten_norm_const_c": c_det,
    "sigma_ideal_matched": c_ideal,
    "noise_floor_inflation_real_over_ideal": inflation,
    "whitened_noise_tail": sigma_tail,
    "effective_net_snr": {"median": float(np.median(eff_net)), "pcts": pcts(eff_net)},
    "effective_over_optimal_median": float(np.median(eff_net / (rho_post + 1e-12))),
    "effective_over_optimal_per_det_median": {
        d: float(np.median(eff_det[:, di] / (rho_post_det[:, di] + 1e-12)))
        for di, d in enumerate(DETS)},
    "N": int(N),
}
print(json.dumps(summary, indent=2))
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ================================================================ FIG 1: orig vs rescaled
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
hi = max(rho_orig.max(), rho_post.max(), target_ov.max())
bins = np.linspace(0, hi + 1, 70)
ax[0].hist(rho_orig, bins=bins, alpha=0.55, color="C2",
           label=f"ORIGINAL (pre-rescale)  med={np.median(rho_orig):.1f}")
ax[0].hist(rho_post, bins=bins, alpha=0.55, color="C0",
           label=f"AFTER rescale  med={np.median(rho_post):.2f}")
ax[0].hist(target_ov, bins=bins, histtype="step", lw=2, color="C1",
           label=f"target HalfNorm(4,5)  med={np.median(target_ov):.2f}")
ax[0].set_xlabel("network optimal SNR"); ax[0].set_ylabel("count")
ax[0].set_title("Original vs rescaled network SNR"); ax[0].legend()
qs = np.linspace(0, 100, 200)
ax[1].plot(np.percentile(target_ov, qs), np.percentile(rho_post, qs), "C0", lw=2)
lim = [0, max(rho_post.max(), target_ov.max())]
ax[1].plot(lim, lim, "k--", lw=1)
ax[1].set_xlabel("target HalfNorm quantile"); ax[1].set_ylabel("rescaled SNR quantile")
ax[1].set_title(f"Q-Q rescaled vs target (|med err|={summary['rescale_median_abs_err']:.3f})")
plt.tight_layout(); plt.savefig(f"{OUT}/1_original_vs_rescaled.png", dpi=120); plt.close()

# ================================================================ FIG 2: whitening faithfulness
fig, ax = plt.subplots(1, len(DETS) + 1, figsize=(6 * (len(DETS) + 1), 5))
for di, d in enumerate(DETS):
    x = rho_post_det[:, di]; y = tdnorm_det[:, di] / c_det[d]   # optimal vs whitened-frame norm (opt units)
    ax[di].scatter(x, y, s=8, alpha=0.4)
    lim = [0, max(x.max(), y.max())]
    ax[di].plot(lim, lim, "k--", lw=1, label="y=x (no loss)")
    ax[di].set_xlabel(f"optimal SNR [{d}]"); ax[di].set_ylabel(f"||whitened signal|| [{d}] (opt units)")
    ax[di].set_title(f"{d}: whitened+cropped signal vs optimal"); ax[di].legend()
# network effective vs optimal
ax[-1].scatter(rho_post, eff_net, s=8, alpha=0.4, color="C3")
lim = [0, max(rho_post.max(), eff_net.max())]
ax[-1].plot(lim, lim, "k--", lw=1, label="y=x")
ax[-1].set_xlabel("labelled optimal net SNR"); ax[-1].set_ylabel("effective net SNR (||s||/sigma)")
ax[-1].set_title(f"effective/optimal med={summary['effective_over_optimal_median']:.3f}")
ax[-1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/2_whitening_faithfulness.png", dpi=120); plt.close()

# ================================================================ FIG 3: noise sigma + effective
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
bulk = np.linspace(0.2, 1.2, 60)
for di, d in enumerate(DETS):
    frac_tail = sigma_tail[d]["frac_gt_2x_ideal"]
    ax[0].hist(np.clip(sigma[d], None, 1.2), bins=bulk, alpha=0.55,
               label=f"{d}  med={sigma_med[d]:.3f}  (tail>2x: {100*frac_tail:.1f}%, max={sigma_tail[d]['max']:.0f})")
ax[0].axvline(c_ideal, color="k", ls="--", label=f"matched-ideal std = {c_ideal:.3f}")
ax[0].set_xlabel("whitened-noise per-sample std (clipped at 1.2; glitch tail runs to 100s)")
ax[0].set_ylabel("count")
ax[0].set_title("Whitened noise floor per detector (fiducial whiten, real O3b)")
ax[0].legend(fontsize=8)
ax[1].hist(rho_post, bins=60, alpha=0.55, color="C0", label=f"labelled optimal med={np.median(rho_post):.2f}")
ax[1].hist(eff_net, bins=60, alpha=0.55, color="C3", label=f"effective (usable) med={np.median(eff_net):.2f}")
ax[1].set_xlabel("network SNR"); ax[1].set_ylabel("count")
ax[1].set_title(f"Optimal vs effective SNR  (ratio={summary['effective_over_optimal_median']:.3f})")
ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/3_noise_and_effective.png", dpi=120); plt.close()

print("=== DONE ===")
print("PNGs:", OUT)

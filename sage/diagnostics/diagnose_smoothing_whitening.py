#!/usr/bin/env python
"""Decisive test of the audit finding: the OLD pipeline whitened with an
UNSMOOTHED median PSD (spectral lines present -> lines divided out -> suppressed),
the CURRENT pipeline whitens with a log-spline-SMOOTHED fiducial (line-free ->
lines SURVIVE whitening -> inflated noise floor -> lower effective SNR).

Whitens the SAME real O3b noise three ways and compares the whitened-noise floor
in the signal band [20,1024] Hz:
  (a) CURRENT smoothed fiducial ASD          -> get_fiducial_asds()
  (b) OLD-style UNSMOOTHED median ASD         -> median of per-segment TorchWelch
  (c) EXACT per-segment ASD (ideal reference) -> each segment by its own Welch PSD
Also overlays the two fiducial ASDs (showing smoothing fills the line notches)
and the residual whitened-noise spectra (line spikes under smoothing).

Outputs PNG+JSON to sage/diagnostics/plots_snr/ (home dir, never /tmp).
"""
import os, sys, json
import numpy as np
import torch

torch.manual_seed(0); np.random.seed(0)
sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/home/nagarajan/research/sage/sage/diagnostics/plots_snr"
os.makedirs(OUT, exist_ok=True)

from sage.core.config import register_configs, get_cfg, get_data_cfg
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()

class O3bCFG:
    export_dir = OUT
    fiducial_dir = "/work/nagarajan/sage_runs/fiducial_psds_o3ab"
    batch_size = 64; device = "cpu"; dtype = torch.float32
    detectors = ["H1", "L1"]; train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]; autocast = False
    class_balance = 0.5; clip_norm = 1.0; dropout = 0.0; num_epochs = 10
    training_iterations = 1; validation_iterations = 1

class O3bDataCFG:
    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [_SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors]
    validation_noise_files = [_SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3bCFG.detectors]
    sample_rate = 2048.0; noise_low_frequency_cutoff = 15.0
    signal_low_frequency_cutoff = 20.0; sample_length_in_s = 12.0; padding_length_in_s = 2.0

register_configs(BaseConfig(O3bCFG()), BaseDataConfig(O3bDataCFG()))
cfg, data_cfg = get_cfg(), get_data_cfg()
SR = data_cfg.sample_rate
PADDED = int(data_cfg.padded_length_in_nsamples)     # 32768
CORRUPT = int(data_cfg.padding_nsamples)             # 4096
DELTA_F = SR / PADDED                                 # 0.0625
F = PADDED // 2 + 1                                   # 16385
pad_freqs = np.arange(F) * DELTA_F
band = (pad_freqs >= 20.0) & (pad_freqs <= 1024.0)   # signal band mask (padded grid)
VALID = PADDED - 2 * CORRUPT                          # 24576 (12 s)
VF = VALID // 2 + 1                                   # 12289
valid_freqs = np.arange(VF) * (SR / VALID)
vband = (valid_freqs >= 20.0) & (valid_freqs <= 1024.0)
print(f"padded={PADDED} F={F} delta_f={DELTA_F} band_bins={band.sum()} valid={VALID} VF={VF}")

from sage.data.asd import get_fiducial_asds
from sage.data.noise import MemmapSingleNoiseSampler
from sage.dsp.welch import TorchWelch

fid_asd = get_fiducial_asds().detach().cpu().numpy()  # (D,F) -- SMOOTHED fiducial ASD
welch = TorchWelch(delta_t=1/SR, seg_len=int(SR*4), seg_stride=int(SR*2), avg_method="median")
welch_freqs = welch.freqs.numpy()

M = 300  # noise segments per detector
kern_const = 2 * DELTA_F / np.sqrt(0.5)

def whiten_np(X_fd, asd, mask=None):
    """Replicate FiducialWhitening exactly (norm='forward'), optional band mask."""
    Xw = X_fd * (kern_const / asd)
    if mask is not None:
        Xw = Xw * mask
    xt = np.fft.irfft(Xw, n=PADDED, norm="forward") * DELTA_F
    return xt[CORRUPT:PADDED - CORRUPT]                # crop to 12 s valid

summary = {}
fig_asd, ax_asd = plt.subplots(1, len(cfg.detectors), figsize=(7*len(cfg.detectors), 5))
fig_sig, ax_sig = plt.subplots(1, len(cfg.detectors), figsize=(7*len(cfg.detectors), 5))
fig_wsp, ax_wsp = plt.subplots(1, len(cfg.detectors), figsize=(7*len(cfg.detectors), 5))

for di, det in enumerate(cfg.detectors):
    single = MemmapSingleNoiseSampler(_SRV.noise_bin(det, "O3b"), return_tensor=True)
    seg_psds = np.empty((M, F), dtype=np.float64)
    Xfds = np.empty((M, F), dtype=np.complex128)
    for m in range(M):
        td = single(PADDED).numpy().astype(np.float64).ravel()
        Xfds[m] = np.fft.rfft(td, norm="forward")
        wp = welch(torch.from_numpy(td)).numpy()                 # (welch_bins,)
        seg_psds[m] = np.interp(pad_freqs, welch_freqs, wp)       # -> padded grid
    raw_median_psd = np.median(seg_psds, axis=0)
    raw_median_asd = np.sqrt(raw_median_psd)
    smoothed_asd = fid_asd[di].astype(np.float64)
    # avoid divide-by-zero at DC / taper
    for a in (raw_median_asd,):
        a[a <= 0] = np.median(a[a > 0])

    # ---- whiten M segments three ways, band-limited to [20,1024] ----
    s_smooth, s_raw, s_exact = [], [], []
    wsp_smooth = np.zeros(VF); wsp_raw = np.zeros(VF)
    bmask = band.astype(np.float64)
    for m in range(M):
        seg_asd = np.sqrt(seg_psds[m]); seg_asd[seg_asd <= 0] = np.median(seg_asd[seg_asd > 0])
        w_s = whiten_np(Xfds[m], smoothed_asd, bmask)
        w_r = whiten_np(Xfds[m], raw_median_asd, bmask)
        w_e = whiten_np(Xfds[m], seg_asd, bmask)
        s_smooth.append(w_s.std()); s_raw.append(w_r.std()); s_exact.append(w_e.std())
        # residual whitened-noise spectrum (full band, no mask) to see line spikes
        wsp_smooth += np.abs(np.fft.rfft(whiten_np(Xfds[m], smoothed_asd), norm="forward"))**2
        wsp_raw    += np.abs(np.fft.rfft(whiten_np(Xfds[m], raw_median_asd), norm="forward"))**2
    s_smooth = np.array(s_smooth); s_raw = np.array(s_raw); s_exact = np.array(s_exact)
    wsp_smooth /= M; wsp_raw /= M

    med = lambda a: float(np.median(a))
    summary[det] = {
        "sigma_smoothed_fiducial_median": med(s_smooth),
        "sigma_unsmoothed_median_median": med(s_raw),
        "sigma_exact_persegment_median":  med(s_exact),
        "smoothed_over_exact":  med(s_smooth) / med(s_exact),
        "unsmoothed_over_exact": med(s_raw) / med(s_exact),
        "smoothed_over_unsmoothed": med(s_smooth) / med(s_raw),
        "effective_snr_gain_unsmoothed_vs_smoothed": med(s_smooth) / med(s_raw),
        "sigma_smoothed_p99": float(np.percentile(s_smooth, 99)),
        "sigma_unsmoothed_p99": float(np.percentile(s_raw, 99)),
    }
    print(det, json.dumps(summary[det], indent=2))

    # ---- ASD overlay ----
    fvalid = pad_freqs > 10
    ax_asd[di].loglog(pad_freqs[fvalid], smoothed_asd[fvalid],
                      "C0", lw=1.2, label="CURRENT smoothed fiducial")
    ax_asd[di].loglog(pad_freqs[fvalid], raw_median_asd[fvalid], "C3", lw=0.9, alpha=0.8,
                      label="OLD-style unsmoothed median")
    ax_asd[di].set_xlim(20, 1024); ax_asd[di].set_xlabel("Hz"); ax_asd[di].set_ylabel("ASD")
    ax_asd[di].set_title(f"{det}: fiducial ASD (smoothing PRESERVES line peaks -> nearly identical)"); ax_asd[di].legend(fontsize=8)

    # ---- sigma distributions ----
    bins = np.linspace(min(s_smooth.min(), s_raw.min(), s_exact.min())*0.98,
                       np.percentile(np.concatenate([s_smooth, s_raw]), 98), 60)
    ax_sig[di].hist(s_exact, bins=bins, alpha=0.5, color="C2", label=f"exact per-seg med={med(s_exact):.3f}")
    ax_sig[di].hist(s_raw, bins=bins, alpha=0.5, color="C3", label=f"unsmoothed median med={med(s_raw):.3f}")
    ax_sig[di].hist(s_smooth, bins=bins, alpha=0.5, color="C0", label=f"smoothed fiducial med={med(s_smooth):.3f}")
    ax_sig[di].set_xlabel("whitened-noise std in [20,1024] Hz"); ax_sig[di].set_ylabel("count")
    ax_sig[di].set_title(f"{det}: floor; smoothed/unsmoothed={summary[det]['smoothed_over_unsmoothed']:.3f}")
    ax_sig[di].legend(fontsize=8)

    # ---- residual whitened-noise spectrum ----
    ax_wsp[di].semilogy(valid_freqs[vband], wsp_smooth[vband], "C0", lw=0.8, label="smoothed fiducial")
    ax_wsp[di].semilogy(valid_freqs[vband], wsp_raw[vband], "C3", lw=0.8, alpha=0.8, label="unsmoothed median")
    ax_wsp[di].set_xlabel("Hz"); ax_wsp[di].set_ylabel("whitened-noise power")
    ax_wsp[di].set_title(f"{det}: residual lines survive whitening (non-stationary; smoothing irrelevant)"); ax_wsp[di].legend(fontsize=8)

fig_asd.tight_layout(); fig_asd.savefig(f"{OUT}/4_asd_smoothed_vs_raw.png", dpi=120); plt.close(fig_asd)
fig_sig.tight_layout(); fig_sig.savefig(f"{OUT}/5_floor_three_ways.png", dpi=120); plt.close(fig_sig)
fig_wsp.tight_layout(); fig_wsp.savefig(f"{OUT}/6_whitened_spectrum.png", dpi=120); plt.close(fig_wsp)

with open(f"{OUT}/smoothing_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("=== DONE ===\n" + json.dumps(summary, indent=2))

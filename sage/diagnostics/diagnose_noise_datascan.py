#!/usr/bin/env python
"""Scan the on-disk O3b noise .bin memmaps for weirdness (data-quality check).

Samples random 16 s windows from the flat float32 memmaps, divides by
dyn_range_fac to physical strain, and checks: amplitude sanity, NaN/Inf,
zero/flat runs (gaps), glitch rate (>8 sigma), and the Welch ASD (physical O3
shape, expected lines, 15 Hz highpass, no aliasing). Run on CPU. Prints stats +
saves plots/noise_datascan.png.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import welch

from sage.utils.servers import get_server
from sage.data.noise._pycbc_lazy import dyn_range_fac
_SRV = get_server()
OUT = "/home/nagarajan/research/sage/sage/diagnostics/plots"; os.makedirs(OUT, exist_ok=True)

FS = 2048; SEG = 16 * FS               # 16 s window = 32768 samples
N_WIN = 120                            # windows per detector
DYN = float(dyn_range_fac())
DETS = ["H1", "L1"]
rng = np.random.default_rng(150914)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
summary = {}
for di, det in enumerate(DETS):
    path = _SRV.noise_bin(det, "O3b")
    mm = np.memmap(path, dtype=np.float32, mode="r")
    ntot = mm.shape[0]
    stds, absmaxes, glitch_hits, zero_runs, nans, infs = [], [], 0, 0, 0, 0
    concat = []
    for _ in range(N_WIN):
        off = int(rng.integers(0, ntot - SEG))
        w = np.asarray(mm[off:off+SEG], dtype=np.float64) / DYN     # physical strain
        nans += int(np.isnan(w).sum()); infs += int(np.isinf(w).sum())
        s = w.std(); stds.append(s); absmaxes.append(np.abs(w).max())
        if s > 0 and np.abs(w).max() > 8*s: glitch_hits += 1
        # zero/flat run > 0.1 s (205 samples)
        flat = np.abs(np.diff(w)) < 1e-30
        if flat.any():
            run = 0; mx = 0
            for f in flat:
                run = run+1 if f else 0; mx = max(mx, run)
            if mx > 205: zero_runs += 1
        if len(concat) < 40: concat.append(w)
    stds = np.array(stds); absmaxes = np.array(absmaxes)
    conc = np.concatenate(concat)
    f, pxx = welch(conc, fs=FS, nperseg=16*FS)                     # ASD
    asd = np.sqrt(pxx)
    band = (f >= 20) & (f <= 1024)
    summary[det] = {
        "amp_std_median": float(np.median(stds)),
        "amp_std_range": [float(stds.min()), float(stds.max())],
        "n_nan": nans, "n_inf": infs,
        "frac_glitch_windows_gt8sigma": round(glitch_hits/N_WIN, 4),
        "frac_windows_with_zero_run": round(zero_runs/N_WIN, 4),
        "asd_min_in_band": float(asd[band].min()),
        "asd_at_100Hz": float(asd[np.argmin(np.abs(f-100))]),
        "asd_at_10Hz_below_cutoff": float(asd[np.argmin(np.abs(f-10))]),
    }
    print(f"=== {det} (n={N_WIN} windows, {ntot/FS/3600:.1f} h of data) ===")
    for k, v in summary[det].items(): print(f"  {k:32s} {v}")
    print()
    axes[0].loglog(f, asd, label=det, alpha=0.8)
    axes[1].hist(absmaxes/stds, bins=40, alpha=0.5, label=det)

axes[0].axvline(20, color="k", ls=":", lw=0.7); axes[0].axvline(15, color="r", ls=":", lw=0.7)
axes[0].set_xlim(8, 1100); axes[0].set_xlabel("Hz"); axes[0].set_ylabel("ASD /rtHz")
axes[0].set_title("O3b noise ASD (15 Hz highpass, lines, no aliasing)"); axes[0].legend()
axes[1].set_xlabel("window abs-max / std (>8 = glitch)"); axes[1].set_ylabel("count")
axes[1].set_title("glitch / non-stationarity"); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/noise_datascan.png", dpi=120); plt.close()

weird = any(s["n_nan"] or s["n_inf"] or s["frac_windows_with_zero_run"] > 0.02
            for s in summary.values())
summary["weirdness_found"] = weird
json.dump(summary, open(f"{OUT}/noise_datascan.json", "w"), indent=2)
print(f"weirdness_found = {weird}")
print(f"saved {OUT}/noise_datascan.png")
print("DONE")

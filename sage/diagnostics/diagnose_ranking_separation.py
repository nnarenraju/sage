#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_ranking_separation.py

Distribution of the network ranking statistic (detection logit, network_output
col 0) split by class -- SIGNAL vs NOISE -- for a run's latest validation epoch.
This is the core detection diagnostic: how well the two classes separate, and
(critically) how they OVERLAP in the tails, which is what sets the FAR /
sensitivity trade-off.

Left  : density histograms, linear y (bulk separation).
Right : same, log y (the overlap tails that actually matter), with the FAR=1e-3
        threshold (set by the noise tail) drawn in.
Prints class counts, separation, noise-tail quantiles, and sensitivity at
fixed FAR.

Usage:  python diagnose_ranking_separation.py [run_export_dir] [--epoch N]
Default run: production_run_HL under /work.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT_DIR = os.path.join(_HERE, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
DEFAULT_RUN = "/work/nagarajan/sage_runs/o3b/production_run_HL"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=DEFAULT_RUN)
    ap.add_argument("--epoch", type=int, default=None, help="epoch group (default: latest)")
    args = ap.parse_args()

    tag = "/".join(args.run.rstrip("/").split("/")[-2:]).replace("run_export_", "")
    with h5py.File(os.path.join(args.run, "validation_data.h5"), "r") as f:
        eps = sorted(int(k.split("_")[1]) for k in f.keys())
        ep = args.epoch if args.epoch is not None else eps[-1]
        g = f[f"epoch_{ep:04d}"]
        s = g["network_output"][:, 0].astype(np.float64)   # ranking stat (logit)
        lab = g["network_target"][:, 2].astype(np.float64)  # 1 signal / 0 noise

    noise = s[lab < 0.5]
    sig = s[lab > 0.5]

    # ---- console report ------------------------------------------------------
    print("=" * 74)
    print(f"RANKING-STAT SEPARATION  {tag}  epoch {ep}  (val on cross-run noise)")
    print("=" * 74)
    print(f"n_signal = {sig.size:,}   n_noise = {noise.size:,}")
    print(f"noise:  median {np.median(noise):+.2f}  p99 {np.percentile(noise,99):.2f} "
          f"p99.9 {np.percentile(noise,99.9):.2f}  p99.99 {np.percentile(noise,99.99):.2f}  max {noise.max():.2f}")
    print(f"signal: median {np.median(sig):+.2f}  p1 {np.percentile(sig,1):.2f} "
          f"p10 {np.percentile(sig,10):.2f}  min {sig.min():.2f}")
    print("-" * 74)
    print("Sensitivity at fixed FAR (threshold set by noise tail):")
    fars = [1e-2, 1e-3, 1e-4]
    thr = {}
    for far in fars:
        t = np.quantile(noise, 1 - far)
        thr[far] = t
        print(f"  FAR={far:.0e}  thr(logit)={t:6.2f}  sensitivity={ (sig>t).mean():.4f}")
    print("=" * 74)

    # ---- plot ----------------------------------------------------------------
    lo = min(noise.min(), sig.min())
    hi = max(noise.max(), sig.max())
    bins = np.linspace(np.floor(lo) - 0.5, np.ceil(hi) + 0.5, 120)
    c_noise, c_sig = "#d62728", "#1f77b4"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax in (ax1, ax2):
        ax.hist(noise, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                color=c_noise, label=f"noise (N={noise.size:,})")
        ax.hist(sig, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                color=c_sig, label=f"signal (N={sig.size:,})")
        ax.hist(noise, bins=bins, density=True, histtype="step", lw=1.4, color=c_noise)
        ax.hist(sig, bins=bins, density=True, histtype="step", lw=1.4, color=c_sig)
        ax.axvline(thr[1e-3], color="black", ls="--", lw=1.1,
                   label=f"FAR=1e-3 thr = {thr[1e-3]:.1f}")
        ax.set_xlabel("ranking statistic (detection logit)")
        ax.grid(alpha=0.25)
    ax1.set_ylabel("density")
    ax1.set_title("Linear scale — bulk separation")
    ax1.legend(fontsize=9, loc="upper center")
    ax2.set_yscale("log")
    ax2.set_title("Log scale — the overlap tails (FAR / missed-signal region)")
    ax2.legend(fontsize=9, loc="upper center")

    fig.suptitle(f"{tag}  ep{ep}: ranking-statistic distribution, signal vs noise "
                 f"(sens@1e-3 = {(sig>thr[1e-3]).mean():.3f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, f"diagnose_ranking_separation_{tag.replace('/','_')}.png")
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

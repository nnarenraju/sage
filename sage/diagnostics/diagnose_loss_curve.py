#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_loss_curve.py

Plot a production run's train + validation BCE loss vs epoch, against the
historical target (0.185, reached ~epoch 40 in a prior correct-SNR run with no
surviving loss record on disk -- shown as a line only, not a curve).

Usage:
    python diagnose_loss_curve.py [run_export_dir ...] [--target 0.185]
Default: runs/o3b/run_export_HL and run_export_HV relative to repo root.
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

DEFAULT_RUNS = [
    os.path.join(_REPO, "runs", "o3b", "run_export_HL"),
    os.path.join(_REPO, "runs", "o3b", "run_export_HV"),
]


def load(path):
    with h5py.File(os.path.join(path, "losses.h5"), "r") as f:
        tr = f["training/loss"][:]
        va = f["validation/loss"][:]
    filled = ~np.all(tr == 0, axis=1)
    n = int(filled.sum())
    tr_bce = tr[:n, 1].astype(np.float64)
    va_full = va[:n].astype(np.float64)
    vmask = ~np.all(va_full == 0, axis=1)
    return np.arange(n), tr_bce, vmask, va_full[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--target", type=float, default=0.185)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.tab10.colors

    for i, run in enumerate(args.runs):
        tag = os.path.basename(run.rstrip("/")).replace("run_export_", "")
        ep, tr, vmask, va = load(run)
        c = colors[i % len(colors)]
        ax.plot(ep, tr, "-", lw=1.6, color=c, label=f"{tag} train bce")
        ax.plot(ep[vmask], va[vmask], "--o", ms=4, lw=1.3, color=c, alpha=0.75,
                label=f"{tag} val bce")
        print(f"{tag}: {len(ep)} epochs | train latest={tr[-1]:.4f} best={tr.min():.4f}@ep{int(tr.argmin())} "
              f"| val best={va[vmask].min():.4f}@ep{int(ep[vmask][np.argmin(va[vmask])])}")

    ax.axhline(args.target, color="green", ls=":", lw=1.5,
               label=f"target {args.target:g} (old run, ~ep40, no surviving curve)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Production loss trajectories vs historical target")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "diagnose_loss_curve.png")
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

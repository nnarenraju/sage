#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_hardbank_scores.py

Distribution of the ranking statistic (detection logit) of the hard-noise
samples held in each run's hard-mining bank. Reads every
``hardbank_<runs>_<dets>.h5`` under the mining dir and plots, per run:

  * left  : density histogram of ``start_found_score`` (the model's ranking stat
            for each segment AT THE EPOCH IT WAS MINED), log-y so the high tail
            (the worst false alarms) is visible.
  * right : survival function P(score > x) vs x, log-y -- directly compares how
            heavy each run's hard-noise tail is (the tail that sets the FAR).

If the bank's re-evaluation column (``eval_stats`` latest column = every sample
re-scored by the newest model) is populated, its distribution is overlaid dashed
so you can see the CURRENT hardness vs the at-mine-time hardness.

Prints a percentile table per run. Run on CPU (login node fine).
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

# Banks live PER RUN under <export_dir>/hard_mining/ (so two same-(runs,dets)
# jobs never contend for one HDF5 lock). They therefore share a filename, and a
# single-directory glob cannot tell them apart -- pass run dirs instead:
#   diagnose_hardbank_scores.py <export_dir> [<export_dir> ...]
# With no arguments the defaults below are used. The old shared
# /work/nagarajan/hard_mining now holds only quarantined pre-2026-08-07 banks.
DEFAULT_RUNS = [
    "/work/nagarajan/sage_runs/o3b/production_run_HL",
    "/work/nagarajan/sage_runs/o3b/production_run_HV",
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# sigmoid(logit) reference guides on the logit axis
_GUIDES = [(0.0, "p=.5"), (2.2, ".9"), (4.6, ".99"), (6.9, ".999"), (9.2, ".9999")]


def _latest_eval(f):
    """Return the newest re-eval column of eval_stats, or None if unusable."""
    if "eval_stats" not in f or f["eval_stats"].shape[1] == 0:
        return None
    col = f["eval_stats"][:, -1].astype(np.float64)
    col = col[np.isfinite(col)]
    return col if col.size else None


def main():
    run_dirs = sys.argv[1:] or DEFAULT_RUNS
    banks = []
    for d in run_dirs:
        found = sorted(glob.glob(os.path.join(d, "hard_mining", "hardbank_*.h5")))
        if not found:                       # allow passing a bank dir directly
            found = sorted(glob.glob(os.path.join(d, "hardbank_*.h5")))
        banks += [(d, b) for b in found
                  if ".killed" not in b and ".tmp" not in b]
    if not banks:
        print("No banks found under: " + ", ".join(run_dirs))
        return

    data = {}
    for d, b in banks:
        # Label by RUN, not by the bank filename -- every run's bank is called
        # hardbank_<runs>_<dets>.h5, so filenames collide across runs.
        tag = os.path.basename(os.path.normpath(d))
        with h5py.File(b, "r") as f:
            found = f["start_found_score"][:].astype(np.float64)
            ev = _latest_eval(f)
            sfe = f["start_found_epoch"][:]
        data[tag] = dict(found=found, eval=ev, nsess=len(np.unique(sfe)))

    # ---- console percentile table --------------------------------------------
    print("=" * 84)
    print("HARD-NOISE RANKING-STAT DISTRIBUTION  (start_found_score = logit at mine time)")
    print("=" * 84)
    print(f"{'run':10s} {'N':>9s} {'sess':>5s} {'median':>7s} {'p90':>6s} "
          f"{'p99':>6s} {'p99.9':>6s} {'max':>6s}  {'frac>4.6(p.99)':>14s}")
    for tag, d in data.items():
        s = d["found"]
        print(f"{tag:10s} {s.size:>9,} {d['nsess']:>5d} {np.median(s):>7.2f} "
              f"{np.percentile(s,90):>6.2f} {np.percentile(s,99):>6.2f} "
              f"{np.percentile(s,99.9):>6.2f} {s.max():>6.2f}  {np.mean(s>4.6):>14.4f}")
    print("=" * 84)

    # ---- plot ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.tab10.colors
    bins = np.linspace(-2, 15, 120)
    for i, (tag, d) in enumerate(data.items()):
        c = colors[i % len(colors)]
        s = d["found"]
        ax1.hist(s, bins=bins, density=True, histtype="step", lw=1.8, color=c,
                 label=f"{tag} (N={s.size:,})")
        if d["eval"] is not None:
            ax1.hist(d["eval"], bins=bins, density=True, histtype="step", lw=1.1,
                     ls="--", color=c, alpha=0.7)
        # survival function P(score > x)
        xs = np.sort(s)
        surv = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs, surv, lw=1.8, color=c, label=tag)

    for x, lbl in _GUIDES:
        for ax in (ax1, ax2):
            ax.axvline(x, color="grey", ls=":", lw=0.6, alpha=0.5)
        ax1.text(x, ax1.get_ylim()[1] * 0.92 if False else 0, "", fontsize=7)

    ax1.set_yscale("log")
    ax1.set_xlabel("ranking statistic (detection logit)")
    ax1.set_ylabel("density (log)")
    ax1.set_title("Hard-noise ranking-stat distribution per run\n"
                  "(solid = at mine time; dashed = re-eval by newest model, if present)",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.25)

    ax2.set_yscale("log")
    ax2.set_xlabel("ranking statistic x")
    ax2.set_ylabel("P(score > x)  [fraction of bank]")
    ax2.set_title("Survival function — tail heaviness (the FAR-setting tail)", fontsize=10)
    ax2.set_xlim(-2, 15)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25, which="both")
    # annotate the sigmoid guides on ax2
    for x, lbl in _GUIDES:
        ax2.text(x, 1.3e-4, f"σ={lbl}", rotation=90, fontsize=6,
                 va="bottom", ha="right", color="grey")

    fig.suptitle("Hard-mining bank: distribution of banked hard-noise ranking statistics",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "diagnose_hardbank_scores.png")
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

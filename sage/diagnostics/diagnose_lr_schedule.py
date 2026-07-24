#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_lr_schedule.py

Reconstruct and plot the exact per-batch learning-rate schedule used by the
production trainer (runs/*/train_hard.py): a short LinearLR warmup followed by
a single CosineAnnealingLR decay, wired with SequentialLR and stepped once per
optimiser step. We build the SAME torch scheduler objects with the SAME args as
the trainer, so this is a faithful replay (including any SequentialLR off-by-one)
rather than an analytic approximation.

Run on CPU (login node is fine): builds a 1-param dummy optimiser, steps the
scheduler total_steps times, records the LR, and plots LR vs epoch. Prints the
warmup peak, end-of-warmup LR, a per-epoch-band table, and the final LR.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# ── Production config values (runs/{o3a,o3b}/config_*.py + train_hard.py) ──────
BASE_LR      = 2e-4          # AdamW lr
START_FACTOR = 1e-3          # LinearLR start_factor -> warmup starts at 2e-7
ETA_MIN      = 1e-6          # CosineAnnealingLR floor
NUM_EPOCHS   = 128
BATCH_SIZE   = 64
TRAIN_ITERS  = int(2_000_000 / BATCH_SIZE)   # = 31250 iters/epoch
WARMUP_STEPS = 20_000

TOTAL_STEPS  = NUM_EPOCHS * TRAIN_ITERS       # = 4,000,000
warmup_steps = min(WARMUP_STEPS, max(1, TOTAL_STEPS // 2))

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def build_scheduler():
    """Identical construction to runs/o3b/train_hard.py."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=BASE_LR)
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=START_FACTOR, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=max(1, TOTAL_STEPS - warmup_steps),
                              eta_min=ETA_MIN),
        ],
        milestones=[warmup_steps],
    )
    return opt, sched


def main():
    opt, sched = build_scheduler()

    lrs = np.empty(TOTAL_STEPS, dtype=np.float64)
    for step in range(TOTAL_STEPS):
        lrs[step] = opt.param_groups[0]["lr"]   # LR *applied* at this step
        opt.step()
        sched.step()

    steps  = np.arange(TOTAL_STEPS)
    epochs = steps / TRAIN_ITERS

    # ── Console report ────────────────────────────────────────────────────────
    peak_idx = int(np.argmax(lrs))
    print("=" * 68)
    print("LR SCHEDULE  (warmup -> cosine, stepped per batch)")
    print("=" * 68)
    print(f"base_lr           : {BASE_LR:.3e}")
    print(f"warmup start_lr   : {BASE_LR * START_FACTOR:.3e}  (start_factor={START_FACTOR})")
    print(f"warmup_steps      : {warmup_steps:,}  (~{warmup_steps / TRAIN_ITERS:.3f} epoch)")
    print(f"total_steps       : {TOTAL_STEPS:,}  ({NUM_EPOCHS} epochs x {TRAIN_ITERS:,} iters)")
    print(f"eta_min (floor)   : {ETA_MIN:.3e}")
    print("-" * 68)
    print(f"peak LR           : {lrs[peak_idx]:.6e}  at step {peak_idx:,} (epoch {peak_idx / TRAIN_ITERS:.3f})")
    print(f"LR at end of warmup: {lrs[warmup_steps]:.6e}  (step {warmup_steps:,})")
    print(f"final LR          : {lrs[-1]:.6e}  (step {TOTAL_STEPS - 1:,})")
    print("-" * 68)
    print("LR at start of epoch:")
    for e in [0, 1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 120, 127]:
        s = e * TRAIN_ITERS
        print(f"  epoch {e:3d}  (step {s:>9,}) : {lrs[s]:.6e}")
    print("=" * 68)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Full-run curve, linear y
    ax1.plot(epochs, lrs, lw=1.4, color="#1f77b4")
    ax1.axvline(warmup_steps / TRAIN_ITERS, color="crimson", ls="--", lw=1.0,
                label=f"warmup end ({warmup_steps / TRAIN_ITERS:.2f} ep)")
    ax1.axhline(BASE_LR, color="grey", ls=":", lw=0.8)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("learning rate")
    ax1.set_title("Full schedule: linear warmup -> cosine decay")
    ax1.set_xlim(0, NUM_EPOCHS)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Warmup zoom (first ~1.5 epochs), so the 20k-step ramp is visible
    zoom_ep = 1.5
    zmask = epochs <= zoom_ep
    ax2.plot(epochs[zmask], lrs[zmask], lw=1.6, color="#ff7f0e")
    ax2.axvline(warmup_steps / TRAIN_ITERS, color="crimson", ls="--", lw=1.0,
                label=f"warmup end ({warmup_steps:,} steps)")
    ax2.axhline(BASE_LR, color="grey", ls=":", lw=0.8, label=f"base lr {BASE_LR:.0e}")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("learning rate")
    ax2.set_title(f"Warmup zoom (first {zoom_ep} epochs)")
    ax2.set_xlim(0, zoom_ep)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Sage production LR schedule  |  base={BASE_LR:.0e}  warmup={warmup_steps:,} steps  "
        f"cosine->{ETA_MIN:.0e}  |  {NUM_EPOCHS} epochs x {TRAIN_ITERS:,} iters",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "diagnose_lr_schedule.png")
    fig.savefig(out, dpi=130)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

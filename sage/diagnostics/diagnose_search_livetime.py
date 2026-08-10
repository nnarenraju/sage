#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : diagnose_search_livetime.py
Description   : Audit segment coverage and the analysed-time decomposition.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Analysed time is the denominator of every rate the search reports, so it is audited
rather than assumed. This runs on the sidecars alone and needs no GPU.

What it reports, per detector and observing run:

* how the stored chunks are laid out in index and in time, and how far the two disagree;
* the overlap between consecutive chunks, and how much of each overlap can host a window;
* where coverage is lost, separated into the window not fitting inside a chunk, the band
  at each boundary too narrow to host a window start, the stride phase restarting at each
  chunk, and genuine gaps in the data;
* the coincident time between detectors, and the count of window starts it supports;
* the sub-sample misalignment between detectors, which is a timing systematic.

The decomposition is what matters: a total on its own cannot show whether a shortfall is
expected geometry or a bug.

Usage
-----
    python -m sage.diagnostics.diagnose_search_livetime --run O3a --detectors H1 L1
    python -m sage.diagnostics.diagnose_search_livetime --run O3a --detectors H1 L1 V1
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sage.search.geometry import SearchGeometry  # noqa: E402
from sage.search.grid import AnalysisGrid  # noqa: E402
from sage.search.segments import (  # noqa: E402
    coincident_intervals,
    load_segments,
    merge_intervals,
    sort_by_gps,
    window_hosts,
)

_HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(_HERE, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_RELEASES = {
    "O3a": "/work/nagarajan/data_release_o3a",
    "O3b": "/work/nagarajan/data_release",
    "O4a": "/work/nagarajan/data_release_o4a",
    "O4b": "/work/nagarajan/data_release_o4b",
}

GEOMETRY = SearchGeometry(
    sample_rate=2048.0,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=205,
    tc_lower_s=5.0,
    tc_upper_s=7.0,
)

DAY = 86400.0


def audit_layout(segments) -> dict:
    """Compare the index layout against the time layout for one detector."""
    ordered = sort_by_gps(segments)
    starts = np.array([s.gps_start for s in segments])
    index_gaps = [
        b.sample_start_idx - (a.sample_start_idx + a.nsamples)
        for a, b in zip(segments, segments[1:])
    ]
    durations = np.array([s.duration_s for s in segments])
    spans = merge_intervals((s.gps_start, s.gps_end) for s in segments)
    return {
        "n_segments": len(segments),
        "index_contiguous": all(g == 0 for g in index_gaps),
        "gps_sorted_in_file": bool(np.all(np.diff(starts) > 0)),
        "duration_min_s": float(durations.min()),
        "duration_max_s": float(durations.max()),
        "naive_sum_d": float(durations.sum() / DAY),
        "union_d": float(sum(e - s for s, e in spans) / DAY),
        "inflation": float(durations.sum() / sum(e - s for s, e in spans)),
        "n_union_intervals": len(spans),
        "first_gps": float(ordered[0].gps_start),
        "last_gps": float(ordered[-1].gps_end),
    }


def audit_overlaps(segments, window_s: float) -> dict:
    """Distribution of chunk overlaps and the hole each boundary leaves."""
    ordered = sort_by_gps(segments)
    overlaps = np.array(
        [
            a.gps_end - b.gps_start
            for a, b in zip(ordered, ordered[1:])
            if a.gps_end > b.gps_start
        ]
    )
    gaps = np.array(
        [
            b.gps_start - a.gps_end
            for a, b in zip(ordered, ordered[1:])
            if b.gps_start > a.gps_end
        ]
    )
    return {
        "n_overlaps": int(overlaps.size),
        "overlap_median_s": float(np.median(overlaps)) if overlaps.size else 0.0,
        "overlap_min_s": float(overlaps.min()) if overlaps.size else 0.0,
        "overlap_max_s": float(overlaps.max()) if overlaps.size else 0.0,
        "overlaps": overlaps,
        "n_gaps": int(gaps.size),
        "gap_total_d": float(gaps.sum() / DAY) if gaps.size else 0.0,
        "predicted_hole_s": float(window_s - np.median(overlaps)) if overlaps.size else 0.0,
    }


def audit_coverage(segments, window_s: float, stride_s: float, restrict_to=None) -> dict:
    """Decompose lost coverage into its causes."""
    spans, report = window_hosts(
        segments,
        GEOMETRY.window_samples,
        GEOMETRY.stride_samples,
        restrict_to=restrict_to,
    )
    out = report.as_dict()
    out["closes_to_s"] = abs(
        out["hosted_s"]
        + out["lost_window_fit_s"]
        + out["lost_boundary_holes_s"]
        + out["lost_phase_restart_s"]
        - out["union_s"]
    )
    out["mean_hole_s"] = (
        out["lost_boundary_holes_s"] / out["n_holes"] if out["n_holes"] else 0.0
    )
    return out


def audit_coincidence(segments_by_detector, window_s: float, stride_s: float) -> dict:
    """Coincident time and the window starts it supports."""
    detectors = list(segments_by_detector)
    coincident = coincident_intervals(segments_by_detector)
    coincident_s = sum(e - s for s, e in coincident)
    grid = AnalysisGrid.build(GEOMETRY, segments_by_detector, coincident)
    blocks = grid.blocks(4096.0)
    residuals = {d: 0.0 for d in grid.detectors}
    for block in blocks[: min(len(blocks), 40)]:
        for detector, value in grid.alignment_residuals(block).items():
            residuals[detector] = max(residuals[detector], value)
    return {
        "detectors": detectors,
        "coincident_d": coincident_s / DAY,
        "n_intervals": len(coincident),
        "n_windows": len(grid),
        "analysed_d": grid.livetime_s / DAY,
        "efficiency": grid.livetime_s / coincident_s if coincident_s else 0.0,
        "n_blocks": len(blocks),
        "residual_samples": residuals,
        "residual_ms": {d: v / 2048.0 * 1e3 for d, v in residuals.items()},
        "coverage": grid.coverage.as_dict() if grid.coverage else {},
    }


def plot(report: dict, outdir: str | Path) -> Sequence[Path]:
    """Write the overlap and coverage figures."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run = report["run"]
    written = []

    # --- chunk overlaps, per detector
    fig, axes = plt.subplots(
        1, len(report["per_detector"]), figsize=(5.0 * len(report["per_detector"]), 4.0),
        squeeze=False,
    )
    for ax, (detector, block) in zip(axes[0], report["per_detector"].items()):
        overlaps = block["overlaps"]["overlaps"]
        ax.hist(overlaps, bins=80, color="#3b6ea5")
        ax.axvline(
            GEOMETRY.window_s, color="k", ls="--", lw=1.5,
            label=f"window = {GEOMETRY.window_s:.1f} s",
        )
        ax.set_yscale("log")
        ax.set_xlabel("chunk overlap (s)")
        ax.set_ylabel("boundaries")
        ax.set_title(f"{detector} {run}: {overlaps.size:,} boundaries")
        ax.legend(fontsize=9)
    fig.suptitle(
        "Overlap below the window length is what leaves an unreachable band at "
        "each boundary",
        fontsize=10,
    )
    fig.tight_layout()
    path = outdir / f"diagnose_search_livetime_overlaps_{run}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- coverage decomposition, per detector
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    detectors = list(report["per_detector"])
    keys = [
        ("hosted_s", "analysed", "#2a9d5c"),
        ("lost_window_fit_s", "window does not fit", "#e08a1e"),
        ("lost_boundary_holes_s", "boundary holes", "#c1443c"),
        ("lost_phase_restart_s", "stride phase restart", "#7d5ba6"),
    ]
    bottom = np.zeros(len(detectors))
    for key, label, colour in keys:
        values = np.array(
            [report["per_detector"][d]["coverage"][key] / DAY for d in detectors]
        )
        ax.bar(detectors, values, bottom=bottom, label=label, color=colour)
        bottom += values
    for i, detector in enumerate(detectors):
        ax.text(
            i, bottom[i], f"  {bottom[i]:.2f} d union", va="bottom", ha="center",
            fontsize=9,
        )
    ax.set_ylabel("time (days)")
    ax.set_title(f"{run}: where the observing time goes, per detector")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = outdir / f"diagnose_search_livetime_coverage_{run}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- losses alone, which are invisible next to the analysed bar
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    bottom = np.zeros(len(detectors))
    for key, label, colour in keys[1:]:
        values = np.array(
            [report["per_detector"][d]["coverage"][key] for d in detectors]
        )
        ax.bar(detectors, values, bottom=bottom, label=label, color=colour)
        bottom += values
    ax.set_ylabel("time lost (s)")
    ax.set_title(f"{run}: itemised losses only")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = outdir / f"diagnose_search_livetime_losses_{run}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- network comparison
    if report["networks"]:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        names = list(report["networks"])
        analysed = [report["networks"][n]["analysed_d"] for n in names]
        coincident = [report["networks"][n]["coincident_d"] for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, coincident, width=0.4, label="coincident", color="#9bb7d4")
        ax.bar(x + 0.2, analysed, width=0.4, label="analysed", color="#2a9d5c")
        for i, name in enumerate(names):
            ax.text(
                i + 0.2, analysed[i],
                f"  {report['networks'][name]['n_windows'] / 1e6:.1f} M",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("time (days)")
        ax.set_title(f"{run}: coincident and analysed time by network")
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = outdir / f"diagnose_search_livetime_networks_{run}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def _report(run: str, detectors: Sequence[str], release_dir: Path) -> dict:
    segments_by_detector = {}
    for detector in detectors:
        path = release_dir / f"data_{detector}_{run}_segments.json"
        if not path.is_file():
            raise SystemExit(f"missing sidecar: {path}")
        segments_by_detector[detector] = load_segments(path)

    per_detector = {}
    for detector, segments in segments_by_detector.items():
        per_detector[detector] = {
            "layout": audit_layout(segments),
            "overlaps": audit_overlaps(segments, GEOMETRY.window_s),
            "coverage": audit_coverage(segments, GEOMETRY.window_s, GEOMETRY.stride_s),
        }

    networks = {}
    names = list(detectors)
    pairs = [
        [names[i], names[j]]
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    combos = pairs + ([names] if len(names) > 2 else [])
    for combo in combos:
        key = "".join(d[0] for d in combo)
        networks[key] = audit_coincidence(
            {d: segments_by_detector[d] for d in combo},
            GEOMETRY.window_s,
            GEOMETRY.stride_s,
        )
    return {"run": run, "per_detector": per_detector, "networks": networks}


def main(argv: Optional[list] = None) -> int:
    """Run the audit and print the decomposition."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[5])
    parser.add_argument("--run", default="O3a", choices=sorted(DEFAULT_RELEASES))
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    parser.add_argument("--release-dir", default=None)
    parser.add_argument("--outdir", default=OUT_DIR)
    args = parser.parse_args(argv)

    release_dir = Path(args.release_dir or DEFAULT_RELEASES[args.run])
    report = _report(args.run, args.detectors, release_dir)

    bar = "=" * 78
    print(bar)
    print(f"SEARCH LIVETIME AUDIT   run={args.run}   release={release_dir}")
    print(bar)

    for detector, block in report["per_detector"].items():
        layout, overlaps, coverage = block["layout"], block["overlaps"], block["coverage"]
        print(f"\n-- {detector}")
        print(
            f"   segments {layout['n_segments']:,}   "
            f"index-contiguous={layout['index_contiguous']}   "
            f"gps-sorted-in-file={layout['gps_sorted_in_file']}"
        )
        print(
            f"   durations {layout['duration_min_s']:.1f}-{layout['duration_max_s']:.1f} s"
            f"   naive sum {layout['naive_sum_d']:.2f} d"
            f"   TRUE UNION {layout['union_d']:.2f} d"
            f"   (inflation {layout['inflation']:.3f}x)"
        )
        print(
            f"   overlaps: n={overlaps['n_overlaps']:,}"
            f"  median {overlaps['overlap_median_s']:.4f} s"
            f"  min {overlaps['overlap_min_s']:.4f}"
            f"  max {overlaps['overlap_max_s']:.4f}"
        )
        print(
            f"   predicted hole = window - overlap = {overlaps['predicted_hole_s']:.4f} s"
            f"   |   genuine gaps: {overlaps['n_gaps']:,} totalling "
            f"{overlaps['gap_total_d']:.3f} d"
        )
        print(f"   coverage decomposition (union {coverage['union_s'] / DAY:.4f} d):")
        print(
            f"      analysed              {coverage['hosted_s'] / DAY:9.4f} d"
            f"   ({coverage['n_windows']:,} windows)"
        )
        print(f"      window does not fit   {coverage['lost_window_fit_s'] / DAY:9.4f} d")
        print(
            f"      boundary holes        {coverage['lost_boundary_holes_s'] / DAY:9.4f} d"
            f"   ({coverage['n_holes']:,} holes, mean {coverage['mean_hole_s']:.4f} s)"
        )
        print(f"      stride phase restart  {coverage['lost_phase_restart_s'] / DAY:9.4f} d")
        print(f"      closes to             {coverage['closes_to_s']:.3e} s")

    print(f"\n-- networks")
    for name, net in report["networks"].items():
        print(
            f"   {name:4s} coincident {net['coincident_d']:8.3f} d"
            f"   analysed {net['analysed_d']:8.3f} d"
            f"   ({net['n_windows']:,} windows, {net['efficiency'] * 100:.2f}% of coincident)"
        )
        worst = ", ".join(
            f"{d} {v:.3f} samp ({net['residual_ms'][d]:.4f} ms)"
            for d, v in net["residual_samples"].items()
        )
        print(f"        alignment residual: {worst}")

    written = plot(report, args.outdir)
    print()
    for path in written:
        print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Filename      : split.py
Description   : Partition an observing run's noise into disjoint named parts.

Created on 2026-07-31

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL


Why this exists
---------------
The monolithic-noise loader (:class:`sage.data.noise.real_noise.MemmapNoiseSampler`)
separates training from validation *only* by which file list each sampler reads --
there is no train/val split of a single run's noise stream.  Pointing training and
validation at the same ``data_{det}_{run}.bin`` therefore does NOT hold anything out:
both samplers draw random windows from the identical segment pool and overlap.

This module splits ONE observing run's segments into disjoint named parts (e.g.
``{"train": 0.9, "val": 0.1}``) so each part is usable as its own "run".  Because the
loader (a) builds its eligible-window pool entirely from a bin's sidecar
``*_segments.json`` and (b) confines every sampled window inside a single segment,
**disjoint segment sets imply provably disjoint (non-overlapping) windows** -- a
genuine held-out split, not merely a different RNG seed.

What it writes (zero data duplication)
--------------------------------------
For run ``R``, detector ``D``, part ``P`` it creates, next to the real bin::

    data_{D}_{R}_{P}.bin           -> symlink to data_{D}_{R}.bin   (shared bytes)
    data_{D}_{R}_{P}_segments.json -> subset of the original sidecar records

The records are copied verbatim (their ``sample_start_idx`` / ``byte_offset`` are
absolute indices into the shared bytes, so they stay valid).  Each part is then
addressable via ``get_server().noise_bin(D, f"{R}_{P}", release_dirname)`` exactly
like a native run -- e.g. a config can set ``training_noise_files`` to the
``*_{R}_train.bin`` and ``validation_noise_files`` to the ``*_{R}_val.bin``.

Splitting is chronological by GPS (parts are ordered earliest -> latest) with a
SINGLE global set of cutoffs shared across all detectors, so every part covers the
same wall-clock periods for every detector (coincident multi-detector noise).  A
segment that straddles a cutoff is dropped so the parts stay strictly disjoint.

CLI
---
    python -m sage.data.noise.split --run O3a --detectors H1 L1 \
        --release-dir /work/nagarajan/data_release_o3a \
        --parts train:0.9 val:0.1 [--by volume|span] [--overwrite] [--dry-run]
"""

import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


# ── data structures ──────────────────────────────────────────────────────────
@dataclass
class DetectorPart:
    """One (detector, part) slice of a run."""
    detector: str
    n_segments: int
    duration_days: float
    gps_start: float
    gps_end: float
    bin_path: str
    sidecar_path: str


@dataclass
class RunPart:
    """One named part of a run, across all detectors."""
    name: str                                   # e.g. "train"
    run_part: str                               # e.g. "O3a_train" -> pass to noise_bin()
    fraction: float
    detectors: dict[str, DetectorPart] = field(default_factory=dict)


# ── core ─────────────────────────────────────────────────────────────────────
def _load_sidecar(release_dir: Path, run: str, detector: str) -> tuple[Path, list[dict]]:
    """Return (bin_path, segment records) for one detector of a run; validate presence."""
    bin_path = release_dir / f"data_{detector}_{run}.bin"
    sidecar = release_dir / f"data_{detector}_{run}_segments.json"
    if not bin_path.exists():
        raise FileNotFoundError(f"noise bin not found: {bin_path}")
    if not sidecar.exists():
        raise FileNotFoundError(f"segment sidecar not found: {sidecar}")
    with open(sidecar) as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"sidecar {sidecar} is empty or not a list of segments")
    for k in ("gps_start", "gps_end", "nsamples"):
        if k not in records[0]:
            raise KeyError(f"sidecar records lack required key {k!r}: {sidecar}")
    return bin_path, records


def _cutoffs(records_by_det: dict[str, list[dict]], cum_targets: list[float], by: str) -> list[float]:
    """Compute the N-1 global chronological GPS cutoffs shared across all detectors.

    ``by="volume"`` places a cutoff where the cumulative data volume (sum of
    ``nsamples`` over all detectors, GPS-ordered) crosses each target fraction --
    i.e. "N% of the actual noise data".  ``by="span"`` places it at that fraction
    of the total GPS span (simpler; approximate volume when the run has gaps).
    """
    if by == "span":
        gmin = min(s["gps_start"] for recs in records_by_det.values() for s in recs)
        gmax = max(s["gps_end"]   for recs in records_by_det.values() for s in recs)
        return [gmin + t * (gmax - gmin) for t in cum_targets]

    if by == "volume":
        allsegs = sorted(
            ((s["gps_start"], s["nsamples"])
             for recs in records_by_det.values() for s in recs),
            key=lambda x: x[0],
        )
        total = sum(w for _, w in allsegs)
        cutoffs: list[float] = []
        acc, ti = 0.0, 0
        for gps_start, w in allsegs:
            if ti >= len(cum_targets):
                break
            acc += w
            # first segment whose start pushes cumulative past the target opens the
            # next part -> boundary sits at that segment's start (it lands later part)
            while ti < len(cum_targets) and acc / total >= cum_targets[ti]:
                cutoffs.append(gps_start)
                ti += 1
        return cutoffs

    raise ValueError(f"unknown split method by={by!r} (expected 'volume' or 'span')")


def split_observing_run(
    run: str,
    detectors: Sequence[str],
    parts: Mapping[str, float],
    release_dir: str | os.PathLike,
    *,
    by: str = "volume",
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, RunPart]:
    """Split one observing run's noise into disjoint, chronological named parts.

    Parameters
    ----------
    run
        Observing-run tag as it appears in the bin filename, e.g. ``"O3a"``
        (files ``data_{det}_{run}.bin`` + ``data_{det}_{run}_segments.json``).
    detectors
        Detectors to split, e.g. ``["H1", "L1"]``.  All are split at the SAME global
        GPS cutoffs so parts are temporally aligned across detectors.
    parts
        Ordered mapping ``name -> fraction`` (earliest part first), fractions summing
        to ~1.0, e.g. ``{"train": 0.9, "val": 0.1}``.
    release_dir
        Directory holding the run's bins + sidecars (the ``release_dirname`` folder,
        e.g. ``/work/nagarajan/data_release_o3a``).
    by
        ``"volume"`` (default; fraction of actual noise data) or ``"span"`` (fraction
        of GPS span).  Both are chronological / temporally disjoint.
    overwrite
        If a part's symlink/sidecar already exists, replace it (else raise).
    dry_run
        Compute and report the split but write nothing.
    verbose
        Print a per-detector/part summary.

    Returns
    -------
    dict[str, RunPart]
        Keyed by part name; each ``RunPart.run_part`` is the string to hand to
        ``noise_bin(det, run_part, release_dirname)``.
    """
    part_names = list(parts)
    fracs = [float(parts[p]) for p in part_names]
    if len(part_names) < 2:
        raise ValueError("need at least 2 parts to split")
    if any(f <= 0 for f in fracs):
        raise ValueError(f"all fractions must be > 0, got {dict(parts)}")
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {sum(fracs)}")

    release_dir = Path(release_dir)
    records_by_det = {d: _load_sidecar(release_dir, run, d)[1] for d in detectors}

    cum_targets = []
    running = 0.0
    for f in fracs[:-1]:
        running += f
        cum_targets.append(running)
    cutoffs = _cutoffs(records_by_det, cum_targets, by)
    if len(cutoffs) != len(part_names) - 1:
        raise RuntimeError(
            f"expected {len(part_names)-1} cutoffs, computed {len(cutoffs)} "
            f"(degenerate fractions or too few segments?)"
        )
    bounds = [-math.inf, *cutoffs, math.inf]

    result: dict[str, RunPart] = {}
    for name, frac in zip(part_names, fracs):
        result[name] = RunPart(name=name, run_part=f"{run}_{name}", fraction=frac)

    for det in detectors:
        recs = records_by_det[det]
        buckets: dict[str, list[dict]] = {p: [] for p in part_names}
        dropped = 0
        for s in recs:
            for i, name in enumerate(part_names):
                if bounds[i] <= s["gps_start"] and s["gps_end"] <= bounds[i + 1]:
                    buckets[name].append(s)
                    break
            else:
                dropped += 1  # straddles a cutoff -> dropped to keep parts disjoint

        # strict-disjointness safety check across consecutive non-empty parts
        spans = [(buckets[p][0]["gps_start"], buckets[p][-1]["gps_end"]) if buckets[p] else None
                 for p in part_names]
        for a, b in zip(spans, spans[1:]):
            if a and b and not (a[1] <= b[0]):
                raise RuntimeError(f"{det}: parts overlap in GPS ({a} vs {b}) -- not disjoint")

        for name in part_names:
            segs = buckets[name]
            if not segs:
                raise ValueError(
                    f"{det} part {name!r} is empty -- fractions too small for the "
                    f"segment granularity of {run}"
                )
            run_part = f"{run}_{name}"
            link = release_dir / f"data_{det}_{run_part}.bin"
            side = release_dir / f"data_{det}_{run_part}_segments.json"
            dur = sum(x["nsamples"] for x in segs) / segs[0].get("sample_rate", 2048.0) / 86400.0
            result[name].detectors[det] = DetectorPart(
                detector=det, n_segments=len(segs), duration_days=dur,
                gps_start=segs[0]["gps_start"], gps_end=segs[-1]["gps_end"],
                bin_path=str(link), sidecar_path=str(side),
            )
            if not dry_run:
                for path in (link, side):
                    if path.exists() or path.is_symlink():
                        if not overwrite:
                            raise FileExistsError(f"{path} exists (pass overwrite=True)")
                        path.unlink()
                # relative symlink -> survives moving the release dir
                os.symlink(f"data_{det}_{run}.bin", link)
                with open(side, "w") as f:
                    json.dump(segs, f)

        if verbose:
            tag = "[dry-run] " if dry_run else ""
            print(f"{tag}{det} ({run}): "
                  + " | ".join(f"{p}={len(buckets[p])} segs "
                               f"({result[p].detectors[det].duration_days:.1f}d)"
                               for p in part_names)
                  + f" | dropped {dropped} straddling")

    if verbose:
        print(f"parts addressable as: "
              + ", ".join(f'noise_bin(det, "{rp.run_part}", ...)' for rp in result.values()))
    return result


def train_val_split(
    run: str,
    detectors: Sequence[str],
    release_dir: str | os.PathLike,
    *,
    val_fraction: float = 0.1,
    by: str = "volume",
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, RunPart]:
    """Convenience: two-way held-out split into ``{run}_train`` / ``{run}_val``.

    ``val`` is the LATEST ``val_fraction`` of the run (temporally held out); ``train``
    is the earlier remainder.  See :func:`split_observing_run` for the mechanics.
    """
    return split_observing_run(
        run, detectors,
        {"train": 1.0 - val_fraction, "val": val_fraction},
        release_dir, by=by, overwrite=overwrite, dry_run=dry_run, verbose=verbose,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def _main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Split an observing run's noise into disjoint held-out parts.")
    ap.add_argument("--run", required=True, help="run tag in the bin filename, e.g. O3a")
    ap.add_argument("--detectors", nargs="+", required=True, help="e.g. H1 L1")
    ap.add_argument("--release-dir", required=True,
                    help="dir holding data_{det}_{run}.bin + sidecars")
    ap.add_argument("--parts", nargs="+", required=True, metavar="NAME:FRAC",
                    help="ordered earliest->latest, e.g. train:0.9 val:0.1")
    ap.add_argument("--by", choices=("volume", "span"), default="volume",
                    help="cut by fraction of data volume (default) or GPS span")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    parts: dict[str, float] = {}
    for spec in args.parts:
        name, _, frac = spec.partition(":")
        if not frac:
            ap.error(f"bad --parts entry {spec!r}; expected NAME:FRAC")
        parts[name] = float(frac)

    split_observing_run(
        args.run, args.detectors, parts, args.release_dir,
        by=args.by, overwrite=args.overwrite, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _main()

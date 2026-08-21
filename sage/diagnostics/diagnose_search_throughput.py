#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : diagnose_search_throughput.py
Description   : Measure scoring throughput and the cost of the two forward stages.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Background dominates the cost of a campaign, and how it is best computed depends on the
split between the per-detector stage and the shared stage. Caching the per-detector
stage lets each slide re-run only the shared part, which pays off in proportion to how
much of the work is per-detector. That split is measured here rather than assumed, and
the caching path is only usable when the per-detector stage is genuinely separable.

Reports the achieved rate, the split between stages, the number of slides beyond which
caching wins, and the resulting estimate for a campaign.

Usage
-----
    python -m sage.diagnostics.diagnose_search_throughput --config config_o4a_HL
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence


def _sync(device_type: str) -> None:
    """Wait for the device to finish before stopping a clock."""
    import torch

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call, batches, warmup: int, device_type: str) -> tuple:
    """
    Run ``call`` over pre-read batches and return ``(windows, seconds)``.

    The first ``warmup`` batches are timed and thrown away: the first pass through a
    network pays for kernel autotuning, memory-pool growth and, under ``torch.compile``,
    a full graph capture. Counting it reports a rate the campaign will never see again.
    """
    import time

    for batch in batches[:warmup]:
        call(batch)
    _sync(device_type)

    start = time.perf_counter()
    n_windows = 0
    for batch in batches[warmup:]:
        call(batch)
        n_windows += len(batch)
    _sync(device_type)
    return n_windows, time.perf_counter() - start


def measure_full(engine, reader, n_windows: int) -> dict:
    """
    Windows scored per second through the whole network.

    ``f_full`` in the cost model: the rate a slide runs at with no frontend cache, which
    is what the whole campaign costs if the cache turns out to be unusable.
    """
    batches, warmup = _collect(reader, n_windows)
    scored, elapsed = _timed(
        lambda b: engine.forward(b.strain), batches, warmup, engine.device.type
    )
    return {
        "f_full": scored / elapsed,
        "n_windows": scored,
        "seconds": elapsed,
        "n_batches": len(batches) - warmup,
    }


def measure_split(engine, reader, n_windows: int) -> dict:
    """
    Cost of the per-detector stage and the shared stage separately.

    ``f_front`` covers the whole frontend pass over every detector, and ``f_back`` one
    re-paired set, because that is how the cache is spent: the frontend runs once per
    detector per block and the backend once per slide. The per-detector rate is reported
    beside it for reading, but the projection must use the all-detector one -- charging a
    single detector's frontend for a two-detector network halves that term.
    """
    batches, warmup = _collect(reader, n_windows)
    device_type = engine.device.type
    n_detectors = len(reader.detectors)

    front, front_s = _timed(
        lambda b: [
            engine.forward_frontend(b.strain, i) for i in range(n_detectors)
        ],
        batches,
        warmup,
        device_type,
    )

    # The backend is timed on features the frontend actually produced, not on random
    # tensors of the right shape: the shapes are what the split network defines and
    # guessing them is how a benchmark comes to measure a different network.
    features = [
        [engine.forward_frontend(b.strain, i) for i in range(n_detectors)]
        for b in batches[:1]
    ][0]
    probe = batches[:1] * len(batches)
    back, back_s = _timed(
        lambda _: engine.forward_backend(features), probe, warmup, device_type
    )

    return {
        # Windows per second for the *complete* frontend pass, all detectors. The cost
        # model spends it as one term per campaign (`n_windows / f_front`), so a
        # per-detector rate here would charge for one detector and run D.
        "f_front": front / front_s,
        "f_front_per_detector": front / front_s * n_detectors,
        "f_back": back / back_s,
        "n_detectors": n_detectors,
        "front_seconds": front_s,
        "back_seconds": back_s,
    }


def measure_io(reader, n_windows: int) -> dict:
    """
    Read bandwidth, to show whether scoring is limited by data or by compute.

    Measured on the contiguous block the reader actually moves, not on the unfolded
    window view: consecutive windows overlap by all but ``stride`` samples, so the view
    is roughly 160x the bytes behind it and quoting its size as I/O would overstate the
    read by that factor.
    """
    import time

    import numpy as np

    start = time.perf_counter()
    read_windows = 0
    read_bytes = 0
    for batch in reader:
        read_windows += len(batch)
        block = batch.block
        if block is not None:
            read_bytes += int(np.asarray(block).nbytes)
        if read_windows >= n_windows:
            break
    elapsed = time.perf_counter() - start
    return {
        "read_windows_per_s": read_windows / elapsed,
        "mb_per_s": read_bytes / elapsed / 1.0e6,
        "bytes_per_window": read_bytes / max(read_windows, 1),
        "n_windows": read_windows,
    }


def project(measurements: dict, n_slides: int, n_windows: int) -> dict:
    """
    Projected cost with and without caching, and the crossover in slides.

    The uncached ladder costs ``n_slides`` full passes; the cached one costs one frontend
    pass per detector plus ``n_slides + 1`` backend passes, the extra being the zero-lag
    foreground, which goes through the backend like any slide.
    """
    from sage.search.features import crossover_slides

    f_full = float(measurements["f_full"])
    f_front = float(measurements["f_front"])
    f_back = float(measurements["f_back"])

    uncached_h = n_slides * n_windows / f_full / 3600.0
    cached_h = (n_windows / f_front + (n_slides + 1) * n_windows / f_back) / 3600.0
    crossover = crossover_slides(f_full, f_front, f_back)
    return {
        "n_slides": int(n_slides),
        "n_windows": int(n_windows),
        "zerolag_gpu_h": n_windows / f_full / 3600.0,
        "uncached_gpu_h": uncached_h,
        "cached_gpu_h": cached_h,
        "speedup": uncached_h / cached_h if cached_h > 0 else float("inf"),
        "crossover_slides": crossover,
        "cache_pays": bool(n_slides > crossover),
    }


def _collect(reader, n_windows: int) -> tuple:
    """
    Read enough batches to time against, and say how many are warm-up.

    Read first and timed after, so the measurement is of the network rather than of the
    filesystem; :func:`measure_io` answers the I/O question separately. Two batches of
    warm-up, or one when that is all there is.
    """
    batches = []
    collected = 0
    for batch in reader:
        batches.append(batch)
        collected += len(batch)
        if collected >= n_windows:
            break
    if not batches:
        raise RuntimeError(
            "the reader produced no windows; the lattice may be empty for this "
            "configuration"
        )
    return batches, min(2, len(batches) - 1) if len(batches) > 1 else 0


def _concurrency(args, n_slides: int, n_lattice: int) -> int:
    """
    Aggregate throughput against the number of workers sharing one GPU.

    The layout question a campaign has to answer before it is queued. A device that one
    worker already saturates gains nothing from a second, and the right shape is one
    slide per GPU; a device left waiting on the host between batches scales with
    workers, and the same cards then carry several slides each.

    Workers are separate processes rather than threads or forks: each needs its own CUDA
    context and its own reader, and a forked process cannot re-initialise CUDA at all.
    """
    import subprocess
    import sys
    import time

    base = [
        sys.executable,
        "-m",
        "sage.diagnostics.diagnose_search_throughput",
        "--config",
        args.config,
        "--windows",
        str(args.windows),
        "--batch-size",
        str(args.batch_size),
        "--worker",
        "--skip-io",
    ]
    print(f"{'workers':>8s} {'aggregate':>12s} {'per worker':>12s} {'scaling':>9s}")
    single = None
    for n_workers in args.streams:
        start = time.perf_counter()
        procs = [
            subprocess.Popen(base, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for _ in range(n_workers)
        ]
        rates = []
        for proc in procs:
            out, _ = proc.communicate()
            for line in out.decode().splitlines():
                if line.startswith("WORKER_RATE"):
                    rates.append(float(line.split()[1]))
        elapsed = time.perf_counter() - start
        if len(rates) != n_workers:
            print(f"{n_workers:>8d}   {len(rates)} of {n_workers} workers reported; "
                  f"the rest failed")
            continue
        aggregate = sum(rates)
        single = single or aggregate
        print(f"{n_workers:>8d} {aggregate:>12,.0f} {aggregate/n_workers:>12,.0f} "
              f"{aggregate/single:>8.2f}x   ({elapsed:.0f} s wall)")
    return 0


def _sweep(spec, args, n_slides: int, n_lattice: int) -> int:
    """
    Measure the rates at several batch sizes and report the projection for each.

    Throughput on a large card is usually launch-bound at small batches and bandwidth-
    bound at large ones, so the campaign rate is a function of the batch size and quoting
    one number without saying which size produced it is quoting a coincidence. Peak
    device memory is reported beside each, so the largest size that fits is measured
    rather than inferred.
    """
    import torch

    print(f"{'batch':>8s} {'full':>10s} {'front':>10s} {'back':>10s} "
          f"{'peak GB':>9s} {'cached GPU-h':>13s}")
    best = None
    for batch_size in args.sweep:
        engine, reader, _ = _build(spec, batch_size)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            full = measure_full(engine, reader, args.windows)
            reader.seek(0)
            split = measure_split(engine, reader, args.windows)
            peak = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else float("nan")
            )
        except Exception as error:
            # Out of memory is the answer to "does this size fit", so the sweep stops
            # rather than failing. Anything else is a defect that a larger batch has
            # exposed, and truncating it to a one-line summary is how it stays hidden --
            # so the whole traceback is printed and the sweep still stops, because every
            # remaining size is larger.
            import traceback

            oom = "out of memory" in str(error).lower()
            print(f"{batch_size:>8,d}   {'OOM' if oom else type(error).__name__}: "
                  f"{error}")
            if not oom:
                traceback.print_exc()
            break
        finally:
            reader.close()
        plan = project({**full, **split}, n_slides, n_lattice)
        print(f"{batch_size:>8,d} {full['f_full']:>10,.0f} {split['f_front']:>10,.0f} "
              f"{split['f_back']:>10,.0f} {peak:>9.1f} {plan['cached_gpu_h']:>13.1f}")
        if best is None or full["f_full"] > best[1]:
            best = (batch_size, full["f_full"], plan)
    if best is not None:
        size, rate, plan = best
        print(f"\nfastest at batch {size:,}: {rate:,.0f} win/s   "
              f"{n_slides} slides cached {plan['cached_gpu_h']:.1f} GPU-h")
    return 0


def _build(spec, batch_size: int):
    """Engine and reader for this campaign, on the configured device."""
    from pathlib import Path as _Path

    from sage.search.checkpoint import as_config, load_search_model
    from sage.search.engine import SearchEngine, build_param_sampler, build_processor
    from sage.search.grid import AnalysisGrid
    from sage.search.reader import StreamingStrainReader
    from sage.search.segments import coincident_intervals, load_segments

    geometry = spec.geometry_object()
    segments = {
        detector: load_segments(
            _Path(spec.data.release_dir)
            / f"data_{detector}_{spec.data.observing_run}_segments.json"
        )
        for detector in spec.data.detectors
    }
    grid = AnalysisGrid.build(geometry, segments, coincident_intervals(segments))

    model, ckpt = load_search_model(
        spec.engine.checkpoint,
        cfg=None,
        data_cfg=None,
        device=spec.engine.device,
        architecture=spec.engine.architecture,
    )
    cfg, data_cfg = as_config(ckpt.cfg), as_config(ckpt.data_cfg)
    spec.apply_shadow_overrides(cfg, data_cfg)
    sampler = build_param_sampler(
        cfg, data_cfg, spec.engine.gwconfig, seed=int(spec.engine.sampler_seed)
    )
    engine = SearchEngine(
        model,
        build_processor(sampler),
        geometry,
        device=spec.engine.device,
        amp_dtype=spec.engine.amp_dtype,
        autocast=bool(ckpt.cfg.get("autocast", True)),
    )
    reader = StreamingStrainReader(
        spec.data.release_dir, grid, geometry, batch_size=batch_size
    )
    return engine, reader, grid


def main(argv: Optional[list] = None) -> int:
    """Run the measurements and print the projection."""
    parser = argparse.ArgumentParser(
        description="Measure search throughput and project the campaign cost."
    )
    parser.add_argument("--config", required=True, help="Campaign config module or path.")
    parser.add_argument("--windows", type=int, default=200_000, help="Windows to time.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--streams",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Concurrent worker counts to probe on one GPU. Answers how to lay a "
            "campaign out across several cards: if one worker already saturates the "
            "device, the aggregate rate is flat and the right shape is one slide per "
            "GPU; if it scales, the device is waiting on the host and several workers "
            "per GPU is the cheaper campaign."
        ),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Batch sizes to sweep instead of measuring one. Background is the expensive "
            "stage and these cards hold 80-140 GB, so the rate at the batch size the "
            "campaign will actually use is the number that matters -- not the rate at a "
            "default. Reports peak device memory per size, so the largest that fits is "
            "measured rather than guessed."
        ),
    )
    parser.add_argument(
        "--slides",
        type=int,
        default=None,
        help="Slides to project for. Default: the campaign's own.",
    )
    parser.add_argument("--skip-io", action="store_true")
    args = parser.parse_args(argv)

    from sage.search.spec import load_spec

    spec = load_spec(args.config)
    spec.validate()
    engine, reader, grid = _build(spec, args.batch_size)
    n_slides = args.slides
    if n_slides is None:
        # A campaign states a depth rather than a count, so the count comes from the
        # measured foreground -- which the lattice this diagnostic just built carries.
        from sage.search.slides import slides_for_background

        n_slides = (
            spec.slides.n_slides
            if spec.slides.n_slides is not None
            else slides_for_background(
                float(spec.slides.target_background_yr),
                grid.livetime_s,
                1.0 if spec.slides.method == "roll" else 0.9,
            )
        )

    print(f"campaign {spec.tag}   device {spec.engine.device}   "
          f"amp {spec.engine.amp_dtype}")
    print(f"lattice  {len(grid):,} windows   batch {args.batch_size}\n")

    if args.worker:
        # One member of a concurrency probe: measure and print the rate alone.
        full = measure_full(engine, reader, args.windows)
        reader.close()
        print(f"WORKER_RATE {full['f_full']:.3f}")
        return 0

    if args.streams:
        return _concurrency(args, n_slides, len(grid))

    if args.sweep:
        return _sweep(spec, args, n_slides, len(grid))

    try:
        if not args.skip_io:
            io = measure_io(reader, args.windows)
            print(f"I/O      {io['read_windows_per_s']:>12,.0f} win/s   "
                  f"{io['mb_per_s']:.1f} MB/s   "
                  f"{io['bytes_per_window']:.0f} B/window")
            reader.seek(0)

        full = measure_full(engine, reader, args.windows)
        print(f"full     {full['f_full']:>12,.0f} win/s   "
              f"({full['n_windows']:,} windows in {full['seconds']:.2f} s)")
        reader.seek(0)

        split = measure_split(engine, reader, args.windows)
        print(f"frontend {split['f_front']:>12,.0f} win/s   "
              f"(all {split['n_detectors']} detectors; "
              f"{split['f_front_per_detector']:,.0f} per detector)")
        print(f"backend  {split['f_back']:>12,.0f} win/s")
    finally:
        reader.close()

    plan = project({**full, **split}, n_slides, len(grid))
    print(f"\nzero-lag pass            {plan['zerolag_gpu_h']:10.2f} GPU-h")
    print(f"background, {n_slides:3d} slides")
    print(f"  no cache               {plan['uncached_gpu_h']:10.2f} GPU-h")
    print(f"  frontend cached        {plan['cached_gpu_h']:10.2f} GPU-h   "
          f"({plan['speedup']:.2f}x)")
    print(f"  crossover at           {plan['crossover_slides']:10.2f} slides   "
          f"cache pays: {plan['cache_pays']}")
    if not plan["cache_pays"]:
        print("  the backend is not enough cheaper than the whole model for the cache "
              "to pay at this depth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : run_stage.py
Description   : Single entry point for every search stage.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One driver for all stages, so a campaign is described by which stages have run rather
than by a collection of scripts. Stages are resumable and idempotent: re-running one
replaces its own products and leaves the rest alone.

Usage
-----
    python run_stage.py --config config_o4a_HL --stage zerolag
    python run_stage.py --config config_o4a_HL --stage background --slide 7
    python run_stage.py --config config_o4a_HL --stage all
"""

import argparse
import os
import sys
from typing import Optional


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one search stage, or every stage still outstanding."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Campaign config: a path to runs/search/config_*.py, or a dotted module.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        help="Stage name, or 'all' for every stage still pending on the track.",
    )
    parser.add_argument(
        "--track",
        default="core",
        help="Track to resolve 'all' against. Default: core.",
    )
    parser.add_argument(
        "--slide",
        type=int,
        action="append",
        default=None,
        help="Slide id for the background stage; repeatable for an array task.",
    )
    parser.add_argument(
        "--slide-group",
        type=int,
        default=None,
        help=(
            "Array-task index, 1-based. With --slides-per-group this selects a "
            "contiguous group of slides, which is how a background campaign is laid "
            "across several GPUs. Slides within a group share one frontend cache build."
        ),
    )
    parser.add_argument(
        "--slides-per-group",
        type=int,
        default=None,
        help="Slides each array task owns. Required with --slide-group.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Stage to omit, repeatable. Its dependants are omitted with it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run even if the stage is recorded complete. Needed after a code change: "
            "the spec hash covers the configuration and the data, not the source."
        ),
    )
    parser.add_argument(
        "--no-cascade",
        action="store_true",
        help=(
            "Do not invalidate downstream stages on success. A promise that this re-run "
            "is byte-for-byte idempotent; wrong, it leaves later products describing an "
            "input that has been replaced."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stages that would run, and stop.",
    )
    return parser.parse_args(argv)


def load_spec(config_module: str):
    """Import a campaign config module and return its specification."""
    from sage.search.spec import load_spec as _load

    return _load(config_module)


def main(argv: Optional[list] = None) -> int:
    """Resolve the requested stages, run them in dependency order, record the outcome."""
    from sage.search import stages as S

    args = parse_args(argv)
    spec = load_spec(args.config)
    spec.validate()

    if args.slide_group is not None:
        if not args.slides_per_group:
            raise SystemExit("--slide-group needs --slides-per-group")
        if args.slide:
            raise SystemExit(
                "--slide and --slide-group both name which slides to run; give one. "
                "--slide lists them, --slide-group derives them from an array index"
            )
        # 1-based, because SLURM array indices are, and an off-by-one here silently
        # drops the last slide of the ladder rather than failing.
        first = (args.slide_group - 1) * args.slides_per_group + 1
        args.slide = list(range(first, first + args.slides_per_group))

    if args.stage == "all":
        if args.force:
            # pending() answers "what is not yet recorded complete", which is empty for a
            # finished campaign -- so --stage all --force printed nothing and ran nothing,
            # which reads as success. With --force the plan is the whole track in
            # dependency order, minus what was skipped.
            omitted = set(args.skip)
            plan = [
                stage.name
                for stage in S.resolve_order(
                    [s.name for s in S.track(args.track) if s.name not in omitted],
                    include_dependencies=False,
                )
            ]
        else:
            plan = [stage.name for stage in S.pending(spec, args.track, skip=args.skip)]
    else:
        S.stage_by_name(args.stage)
        if args.skip:
            # --skip prunes a plan, and a named stage is not a plan. Accepting it
            # silently meant a submit script could name a stage, skip that same stage,
            # and get it anyway.
            raise SystemExit(
                f"--skip {args.skip} applies to '--stage all', which resolves a plan to "
                f"prune. This invocation names one stage ({args.stage!r}); to omit it, "
                "do not name it"
            )
        plan = [args.stage]

    if args.dry_run:
        # Inputs are checked even here: the point of a dry run is to fail before a
        # scheduler has queued anything, and an absent release is exactly that failure.
        spec.validate_inputs()
        print(f"campaign {spec.tag} at {spec.out_dir}")
        print(f"spec hash {spec.hash()}")
        for name in plan:
            print(f"  would run {name}")
        if not plan:
            print("  nothing pending")
        return 0

    spec.validate_inputs()
    cascade = False if args.no_cascade else "auto"
    for name in plan:
        if args.force or not S.is_complete(spec, name):
            extra = {}
            if args.slide and name == "background":
                extra["slides"] = list(args.slide)
            report = S.run_stage(spec, name, cascade=cascade, **extra)
            print(f"{name}: {_summarise(report)}")
        else:
            print(f"{name}: already complete under this configuration")
    return 0


def _summarise(report) -> str:
    """One line of a stage report, without dumping arrays into a job log."""
    if not isinstance(report, dict):
        return repr(report)
    parts = []
    for key, value in report.items():
        if isinstance(value, (int, float, str, bool)) and len(str(value)) <= 60:
            parts.append(f"{key}={value}")
    return " ".join(parts) if parts else "done"


if __name__ == "__main__":
    raise SystemExit(main())

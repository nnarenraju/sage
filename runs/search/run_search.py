#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : run_search.py
Description   : Search an observing run with a trained network.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One call from trained weights to candidates, sensitivity and figures.

Usage
-----
    # the whole core track, from the campaign config
    python run_search.py --config config_o3a_HL.py

    # see the plan and the campaign identity without running anything
    python run_search.py --config config_o3a_HL.py --dry-run

    # shallow background first, to exercise every step end to end. This is a separate
    # campaign in its own directory, not a cheaper version of the production one
    python run_search.py --config config_o3a_HL.py --n-slides 8

    # stop once the candidate list exists
    python run_search.py --config config_o3a_HL.py --stop-after candidates

The checkpoint, the observing run and the detector network come from the config -- they
are what a campaign *is*, and a flag that overrode one would produce a campaign whose
configuration file does not describe it.

Only the ``core`` track runs here. Per-event characterization and parameter estimation
are the ``followup`` track: they need a second environment and are driven per candidate
by ``characterize.py`` against the candidate list this produces.
"""

import argparse
from typing import Optional


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments mirroring :func:`sage.search.pipeline.run_search`."""
    parser = argparse.ArgumentParser(
        description="Search an observing run with a trained network, end to end."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Campaign config: a path to runs/search/config_*.py, or a dotted module.",
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        help="Last stage to run. Everything it depends on runs first.",
    )
    parser.add_argument(
        "--n-slides",
        type=int,
        default=None,
        help=(
            "Override the ladder depth, for a shallow smoke run. Makes a separate "
            "campaign -- its own tag, its own out_dir and its own spec hash -- so it "
            "cannot overwrite the production one it is a smoke test of."
        ),
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Stage to omit, repeatable. Its dependants are omitted with it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the campaign identity, and stop.",
    )
    return parser.parse_args(argv)


def build_spec(args):
    """The campaign specification this invocation describes."""
    import dataclasses
    from pathlib import Path

    from sage.search.spec import load_spec

    spec = load_spec(args.config)
    if args.n_slides is not None:
        # Through the spec, so it lands in the hash -- and through the tag, so it lands in
        # a different directory. The hash alone was not enough: out_dir is built from the
        # tag, so the smoke run wrote its 8-slide plan and shards over the production
        # campaign's 82-slide ones while recording them under its own hash, leaving the
        # finished campaign reporting every stage incomplete with nothing on disk left to
        # complete it from.
        n_slides = int(args.n_slides)
        spec = dataclasses.replace(
            spec,
            tag=f"{spec.tag}_smoke{n_slides}",
            out_dir=Path(spec.out_dir).with_name(
                f"{Path(spec.out_dir).name}_smoke{n_slides}"
            ),
            slides=dataclasses.replace(spec.slides, n_slides=n_slides),
        )
    spec.validate()
    return spec


def main(argv: Optional[list] = None) -> int:
    """Run the search and print the summary."""
    from sage.search import stages as S

    args = parse_args(argv)
    spec = build_spec(args)

    pending = [stage.name for stage in S.pending(spec, "core", skip=args.skip)]
    if args.stop_after is not None:
        declared = S.stage_by_name(args.stop_after)
        if declared.track != "core":
            # Intersecting a followup stage with the core plan silently empties it, and
            # the run then prints "nothing pending" and exits 0 -- which reads as
            # "already done" rather than "this driver does not run that stage".
            raise SystemExit(
                f"--stop-after {args.stop_after!r} names a {declared.track!r} stage; "
                f"this driver runs the core track only. The followup stages "
                f"({', '.join(s.name for s in S.track('followup'))}) are driven per "
                "candidate by characterize.py, against the candidate list this produces"
            )
        wanted = {
            stage.name
            for stage in S.resolve_order([args.stop_after], include_dependencies=True)
        }
        pending = [name for name in pending if name in wanted]

    print(f"campaign {spec.tag}")
    print(f"  out_dir   {spec.out_dir}")
    print(f"  release   {spec.data.release_dir}")
    print(f"  network   {spec.engine.checkpoint}")
    print(
        f"  slides    {spec.slides.n_slides}"
        if spec.slides.n_slides is not None
        else f"  slides    derived from {spec.slides.target_background_yr} yr of "
        f"background ({spec.slides.method})"
    )
    print(f"  spec hash {spec.hash()}")
    if not pending:
        print("  nothing pending")
        return 0
    print("  plan      " + " -> ".join(pending))

    if args.dry_run:
        spec.validate_inputs()
        return 0

    spec.validate_inputs()
    for name in pending:
        report = S.run_stage(spec, name)
        print(f"{name}: {_summarise(report)}")
    return 0


def _summarise(report) -> str:
    """One line of a stage report, without dumping arrays into a job log."""
    if not isinstance(report, dict):
        return repr(report)
    parts = [
        f"{key}={value}"
        for key, value in report.items()
        if isinstance(value, (int, float, str, bool)) and len(str(value)) <= 60
    ]
    return " ".join(parts) if parts else "done"


if __name__ == "__main__":
    raise SystemExit(main())

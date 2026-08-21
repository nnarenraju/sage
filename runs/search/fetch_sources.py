#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : fetch_sources.py
Description   : Build one external release into its canonical form.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The thin driver over :mod:`sage.search.sources`. Each handler there knows one published
release; this knows only how to name one and where the campaign keeps its inputs.

Reaches the network, so it runs on a login node, once, before the campaign. What it
writes is small and self-describing, and every later stage reads that instead of the
release.

Usage
-----
    python fetch_sources.py --config config_o3a_HL --source gwtc3_powerlawpeak
    python fetch_sources.py --config config_o3a_HL --source gwtc3_powerlawpeak --list
"""

import argparse
import importlib
from typing import Optional

#: Known handlers, and where each one's canonical output belongs in a campaign. The
#: registry is here rather than in the package so that adding a release is one entry
#: beside one new module, and so importing ``sage.search`` never enumerates them.
SOURCES = {
    "gwtc3_powerlawpeak": ("injections", "hyperposterior_gwtc3_pp.json"),
}


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download one external release and reduce it to the canonical form "
        "the search reads, inside the campaign's export directory."
    )
    parser.add_argument("--config", required=True, help="Campaign config module or path.")
    parser.add_argument(
        "--source",
        help=f"Handler to run, one of {sorted(SOURCES)}.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print the known handlers and stop."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run even if the canonical file is already there.",
    )
    parser.add_argument(
        "--drop-archive",
        action="store_true",
        help="Delete the release archive once the wanted file is extracted from it.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "How to pick the representative hyperposterior sample, used only under "
            "population_mode='representative'. Default reads the likelihood where the "
            "release publishes one. The whole posterior is stored regardless, which is "
            "what the default marginalising mode draws from."
        ),
    )
    parser.add_argument(
        "--expect-index",
        type=int,
        default=None,
        help=(
            "Refuse to write unless the selection lands on this posterior sample, for "
            "pinning a campaign to an answer that has been checked."
        ),
    )
    parser.add_argument(
        "--no-population",
        action="store_true",
        help=(
            "Store only the representative sample. Smaller, and it makes "
            "population_mode='marginalise' -- the default -- impossible for this "
            "campaign, so the representative would be conditioned on instead."
        ),
    )
    parser.add_argument(
        "--accept-new-selection",
        action="store_true",
        help=(
            "Accept a hyperposterior sample other than the one the reference analysis "
            "recorded. Needed only when the release itself has changed, and it changes "
            "the population injections are drawn from."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Run one handler and report what it wrote."""
    from sage.search.spec import load_spec

    args = parse_args(argv)
    if args.list:
        for name, parts in sorted(SOURCES.items()):
            print(f"  {name:24s} -> {'/'.join(parts)}")
        return 0
    if not args.source:
        raise SystemExit(f"--source is required; known handlers are {sorted(SOURCES)}")
    if args.source not in SOURCES:
        raise SystemExit(
            f"unknown source {args.source!r}; known handlers are {sorted(SOURCES)}. "
            "A release with no handler needs one written -- see sage/search/sources"
        )

    spec = load_spec(args.config)
    dest = spec.path(*SOURCES[args.source])
    if dest.is_file() and not args.refresh:
        print(f"{dest} already built; --refresh to rebuild")
        return 0

    handler = importlib.import_module(f"sage.search.sources.{args.source}")
    print(f"campaign {spec.tag}")
    print(f"record   {handler.RECORD}  ({handler.DOI})")
    method = args.method or handler.DEFAULT_METHOD
    if method not in handler.METHODS:
        raise SystemExit(
            f"unknown method {method!r}; this handler offers {list(handler.METHODS)}"
        )
    print(f"method   {method}")
    written = handler.build(
        dest,
        method=method,
        keep_archive=not args.drop_archive,
        store_population=not args.no_population,
        expect_index=None if args.accept_new_selection else args.expect_index,
    )
    print(f"wrote    {written}")
    import json

    selection = json.loads(written.read_text())["selection"]
    print(f"selected sample {selection['index']} of {selection['n_samples']}"
          + (f", log_likelihood {selection['log_likelihood']:.4f}"
             if selection.get("log_likelihood") is not None else ""))
    for name, value in sorted(handler.load(written).items()):
        print(f"  {name:12s} {value:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

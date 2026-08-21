#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : fetch_catalogues.py
Description   : Seed and freeze a campaign's catalogue cache.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The one place in the search that is allowed to reach the network. Every other stage runs
against ``CatalogueSpec.offline``, which refuses a URL that is not already cached, so the
comparison a campaign makes is against bytes it holds rather than whatever the service
returns that day.

Run this once on a login node before the catalogue stage. The artefacts land in the
campaign's own export directory and the manifest pins each one by digest, so the campaign
is re-runnable offline, on a compute node with no route out, and years later against the
catalogue it actually used.

Usage
-----
    python fetch_catalogues.py --config config_o3a_HL
    python fetch_catalogues.py --config config_o3a_HL --refresh
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

#: Sub-threshold and marginal lists. A candidate matching one of these is a confirmation
#: rather than a discovery, so leaving them out makes the new-event list longer and wrong.
MARGINAL_RELEASES = ("GWTC-1-marginal", "GWTC-2.1-marginal")


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the catalogues this campaign compares against, into its "
        "export directory, and freeze the manifest."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Campaign config: a path to runs/search/config_*.py, or a dotted module.",
    )
    parser.add_argument(
        "--release",
        action="append",
        default=[],
        help="Extra named GWOSC release to cache, repeatable, e.g. GWTC-2.1-confident.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-fetch entries already cached. This is how a catalogue is deliberately "
            "updated; without it a second run touches the network exactly never."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Fetch every catalogue the campaign needs and write the manifest beside them."""
    from sage.search.catalogue.cache import MANIFEST_NAME, CatalogueCache
    from sage.search.catalogue.gwosc import load_cumulative, load_release
    from sage.search.spec import load_spec

    args = parse_args(argv)
    spec = load_spec(args.config)

    cache_dir = Path(spec.catalogue.cache_dir or spec.path("catalogue", "cache"))
    # Deliberately not offline: this script is the fetch the offline default assumes has
    # already happened.
    cache = CatalogueCache(cache_dir, offline_only=False)
    print(f"campaign {spec.tag}")
    print(f"cache    {cache_dir}")

    releases: List[str] = list(args.release)
    if spec.catalogue.include_marginal:
        releases = list(MARGINAL_RELEASES) + releases

    endpoint = spec.catalogue.gwtc_endpoint
    if args.refresh:
        cache.fetch(endpoint, refresh=True)
    cumulative = load_cumulative(cache, endpoint=endpoint)
    _report("gwosc", cumulative, spec)

    for release in releases:
        url = f"https://gwosc.org/eventapi/json/{release}/"
        if args.refresh:
            cache.fetch(url, refresh=True)
        _report(release, load_release(cache, release), spec)

    manifest = cache.freeze(cache_dir / MANIFEST_NAME)
    print(f"\nfrozen {len(cache._entries)} entries -> {manifest}")
    return 0


def _report(key: str, catalogue, spec) -> None:
    """One line per catalogue: what it holds, and how much of it this campaign searched."""
    total = len(catalogue)
    bbh = len(catalogue.filter_bbh())
    print(f"  {key:22s} {total:4d} events  ({bbh} BBH)")


if __name__ == "__main__":
    raise SystemExit(main())

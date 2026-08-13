#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : fetch.py
Description   : Fetch the reference documents cited by sage/search.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The PDFs are not checked in, so this restores them into ``docs/references``. It also
reads back the first page of each file and reports the title, which is how the registry
entries were confirmed rather than transcribed.

Usage
-----
    python docs/references/fetch.py [--verify-only]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARXIV_IDS = (
    "1302.5341",
    "1508.02357",
    "2005.12761",
    "2108.01045",
    "2111.03606",
    "2305.00071",
    "2401.08709",
    "2412.04638",
    "2508.10638",
    "2508.18079",
    "2508.18081",
    "2508.18082",
    "2605.27224",
)


def fetch(arxiv_id: str, dest: Path, timeout: int = 180) -> bool:
    """Download one document if it is not already present."""
    out = dest / f"arxiv_{arxiv_id}.pdf"
    if out.exists() and out.stat().st_size > 0:
        print(f"have  {arxiv_id}")
        return True
    req = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        out.write_bytes(data)
    except Exception as exc:
        print(f"FAIL  {arxiv_id}: {exc}")
        return False
    print(f"got   {arxiv_id}  ({len(data) / 1e6:.1f} MB)")
    return True


def title_of(path: Path) -> str:
    """First-page heading of a stored document, for confirming its identity."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "(pypdf not installed)"
    try:
        text = PdfReader(str(path)).pages[0].extract_text() or ""
    except Exception as exc:
        return f"(unreadable: {exc})"
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines[:3])[:110]


def main(argv=None) -> int:
    """Fetch every document and report what is present."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--verify-only", action="store_true", help="do not download")
    args = parser.parse_args(argv)

    missing = 0
    for arxiv_id in ARXIV_IDS:
        path = HERE / f"arxiv_{arxiv_id}.pdf"
        if args.verify_only:
            if not path.exists():
                print(f"MISSING {arxiv_id}")
                missing += 1
            continue
        if not fetch(arxiv_id, HERE):
            missing += 1

    print()
    for arxiv_id in ARXIV_IDS:
        path = HERE / f"arxiv_{arxiv_id}.pdf"
        if path.exists():
            print(f"{arxiv_id:14s} :: {title_of(path)}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : query.py
Description   : Command-line access to the campaign store.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Answers questions about a campaign without writing code.

Usage
-----
    # everything known about one candidate
    python query.py --config config_o4a_HL --event SGW230814_230901

    # candidates meeting a condition over any recorded quantity
    python query.py --config config_o4a_HL --where "pastro > 0.9 AND dq_vetoed = 0"

    # candidates near a time reported elsewhere
    python query.py --config config_o4a_HL --gps 1368268505.8 --tolerance 1.0

    # arbitrary SQL, exported
    python query.py --config config_o4a_HL --sql "SELECT ..." --export out.csv

    # what can be queried
    python query.py --config config_o4a_HL --describe
"""

import argparse
from typing import Optional


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    raise NotImplementedError


def main(argv: Optional[list] = None) -> int:
    """Run the requested query and print or export the result."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

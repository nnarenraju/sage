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
    raise NotImplementedError


def load_spec(config_module: str):
    """Import a campaign config module and return its specification."""
    raise NotImplementedError


def main(argv: Optional[list] = None) -> int:
    """Resolve the requested stages, run them in dependency order, record the outcome."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

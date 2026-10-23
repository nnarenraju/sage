#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : conftest.py
Description   : Keeps pytest's temporary files off /tmp.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

import os
from pathlib import Path


# Nothing this project runs may write to /tmp, and pytest's tmp_path lands there by
# default. Redirected before any fixture asks for a temporary directory, since the temp
# root is resolved lazily on first use. Set here rather than in addopts so it can still be
# overridden per machine, and so a stale basetemp is not wiped from under a parallel run.
_TEMPROOT = Path(
    os.environ.get("SAGE_PYTEST_TEMPROOT", Path.home() / ".cache" / "sage-pytest")
)
_TEMPROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMPROOT))


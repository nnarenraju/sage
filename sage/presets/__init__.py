#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : __init__.py
Description     : Reusable, named configuration presets for Sage runs.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan


Status
------
Intentionally empty. The previous contents (``configs.py``, ``data_configs.py``
and ``legacy.py`` — 5,314 lines) were written against a package layout that no
longer exists: they imported ``sage.data.generation``, ``sage.data.datasets``
and a top-level ``data`` package, none of which are present, so none of the
three modules had been importable for some time. They were removed rather than
repaired; the full history is preserved in git.

Planned direction
-----------------
This package will be repopulated from the run directories, not from the old
code. The configurations under ``runs/o3a`` and ``runs/o3b`` carry the
substantive, validated methodology — detector combinations, priors, noise
handling, architecture and optimiser settings — currently expressed as
hand-edited Python, one copy per run.

The intent is to lift that into named presets here, so that running Sage means
supplying a short YAML naming a preset plus a few options, rather than copying
and editing a ``config.py``. Until that lands the run directories remain the
source of truth — start from ``runs/o3b/config.py``.
"""

__all__ = []

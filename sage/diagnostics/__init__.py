#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : __init__.py
Description     : Standalone diagnostic scripts.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan


Scripts you run by hand when investigating something -- "why is the SNR off?",
"does this waveform still match LALSim?", "what does the prior actually look
like?". Each is self-contained, has a ``__main__``, prints what it finds and
saves any figures it makes::

    python -m sage.diagnostics.plot_nrt_corner_mismatch --n_samples 200
    python -m sage.diagnostics.diagnose_loss_curve <run_export_dir>

Nothing here is imported by the pipeline, and nothing here runs automatically.

Not to be confused with :mod:`sage.plotting`
--------------------------------------------
The two have deliberately different contracts:

``sage.plotting``
    A *library* of ``plot_*()`` functions plus a driver, called
    programmatically against a finished run's ``export_dir`` to produce the
    standard (and publication) figures. Reusable, no side effects on import.

``sage.diagnostics`` (here)
    *Scripts*, run by a human, usually while something is being debugged. They
    may print, chdir, or assume a particular run layout.

If something here becomes a routine part of looking at every run, it belongs
in ``sage.plotting`` instead.
"""

__all__ = []

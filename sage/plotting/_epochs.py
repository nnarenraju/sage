#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : _epochs.py
Description     : Helpers for turning epoch keys into numbers, labels and tags.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan


Epochs reach the plotting layer in two forms: as plain integers when called
directly, and as HDF5 group keys like ``"epoch_0127"`` when they come from
:class:`~sage.plotting.manager.ValidationPlotManager`.

Plot code that formatted these as ``f"epoch_{epoch}"`` therefore produced
``epoch_epoch_0127`` in filenames, directory names and titles whenever the key
form was used. These helpers normalise both forms so the output reads the same
either way.
"""

from __future__ import annotations

import re

__all__ = ["epoch_number", "epoch_tag", "epoch_title"]

_EPOCH_RE = re.compile(r"(\d+)")


def epoch_number(epoch) -> int:
    """Return the integer epoch for either ``127`` or ``"epoch_0127"``.

    Falls back to ``-1`` for anything with no digits in it, so a caller that
    passes something unexpected gets a sortable value rather than a crash.
    """
    if isinstance(epoch, (int,)) and not isinstance(epoch, bool):
        return int(epoch)
    m = _EPOCH_RE.search(str(epoch))
    return int(m.group(1)) if m else -1


def epoch_tag(epoch) -> str:
    """Filesystem-safe tag: ``epoch_0127`` for both ``127`` and ``"epoch_0127"``.

    Use this wherever a filename or directory name is built, in place of
    ``f"epoch_{epoch}"``.
    """
    n = epoch_number(epoch)
    return f"epoch_{n:04d}" if n >= 0 else f"epoch_{epoch}"


def epoch_title(epoch) -> str:
    """Human-facing label for titles: ``epoch 127``."""
    n = epoch_number(epoch)
    return f"epoch {n}" if n >= 0 else str(epoch)


def epoch_numbers(epochs) -> list:
    """Vectorised :func:`epoch_number` over an iterable."""
    return [epoch_number(e) for e in epochs]

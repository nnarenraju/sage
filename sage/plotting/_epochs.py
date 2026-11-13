#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Epoch keys (``127`` or ``"epoch_0127"``) as numbers, file tags and
title labels.

Filename        : _epochs.py
Description     : Helpers for turning epoch keys into numbers, labels and tags.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan
"""

from __future__ import annotations

import re

__all__ = ["epoch_number", "epoch_tag", "epoch_title"]

_EPOCH_RE = re.compile(r"(\d+)")


def epoch_number(epoch) -> int:
    """
    Integer epoch of ``127`` or ``"epoch_0127"``.

    Parameters
    ----------
    epoch : int or str
        Epoch number or epoch key.

    Returns
    -------
    int
        The first run of digits in ``str(epoch)``; -1 when it has none.
    """
    if isinstance(epoch, (int,)) and not isinstance(epoch, bool):
        return int(epoch)
    m = _EPOCH_RE.search(str(epoch))
    return int(m.group(1)) if m else -1


def epoch_tag(epoch) -> str:
    """
    File name tag ``epoch_0127`` of ``127`` or ``"epoch_0127"``.

    Parameters
    ----------
    epoch : int or str
        Epoch number or epoch key.

    Returns
    -------
    str
        ``f"epoch_{epoch}"`` when ``epoch`` has no digits.
    """
    n = epoch_number(epoch)
    return f"epoch_{n:04d}" if n >= 0 else f"epoch_{epoch}"


def epoch_title(epoch) -> str:
    """
    Title label ``epoch 127`` of ``127`` or ``"epoch_0127"``.

    Parameters
    ----------
    epoch : int or str
        Epoch number or epoch key.

    Returns
    -------
    str
        ``str(epoch)`` when ``epoch`` has no digits.
    """
    n = epoch_number(epoch)
    return f"epoch {n}" if n >= 0 else str(epoch)


def epoch_numbers(epochs) -> list:
    """
    :func:`epoch_number` of each element of ``epochs``.

    Parameters
    ----------
    epochs : iterable of int or str
        Epoch numbers or epoch keys.

    Returns
    -------
    list of int
    """
    return [epoch_number(e) for e in epochs]

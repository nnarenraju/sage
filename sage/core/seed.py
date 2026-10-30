#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : seed.py
Description     : Short description of the file

Created on 2026-01-19 23:34:03

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import numpy as np
import torch


# Every random stream of a run, by its spawn key (append only: a key's meaning never changes).
STREAMS = {"prior": 0, "snr": 1, "gmst": 2, "noise": 3, "recolour": 4, "placement": 5, "mining": 6, "global": 7}


def stream_seed(seed: int, stream: str, epoch: int = 0) -> int:
    """
    Seed of one random stream: NumPy's ``SeedSequence(seed)`` child keyed by the stream and the epoch.

    Parameters
    ----------
    seed : int
        The run's seed (``train.seed`` or ``validation.seed``).
    stream : str
        One of ``STREAMS``.
    epoch : int
        Epoch a training segment starts at; 0 for validation.

    Returns
    -------
    int
        A 64-bit seed for a torch or NumPy generator.

    Raises
    ------
    KeyError
        If the stream is unknown.
    """
    if stream not in STREAMS:
        raise KeyError(f"unknown random stream {stream!r}; streams: {', '.join(STREAMS)}")
    return int(np.random.SeedSequence(seed, spawn_key=(STREAMS[stream], epoch)).generate_state(1, np.uint64)[0])


def generator(seed: int, stream: str, epoch: int, device) -> torch.Generator:
    """A torch generator on ``device`` seeded with ``stream_seed(seed, stream, epoch)``."""
    return torch.Generator(device=device).manual_seed(stream_seed(seed, stream, epoch))

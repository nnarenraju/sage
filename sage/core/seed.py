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

import random
import numpy as np
import torch


class SeedManager:
    """
    Central seed manager for reproducible experiments.

    Seeds Python's :mod:`random`, NumPy's global RNG, and exposes a
    dedicated :class:`numpy.random.Generator` instance (``self.rng``) that
    is isolated from the global state.  Child generators derived via
    :meth:`spawn` are deterministic functions of the root seed and a name
    string, enabling independent reproducible streams for different pipeline
    components without interfering with each other.

    Parameters
    ----------
    seed : int
        Root random seed.  All child generators are derived from this value.

    Attributes
    ----------
    seed : int
        The root seed as an integer.
    rng : numpy.random.Generator
        Independent NumPy generator for library-internal use.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)

        # Global seeding
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Independent RNG stream for library code
        self.rng = np.random.default_rng(self.seed)

    def spawn(self, name: str):
        """
        Create a deterministic child RNG derived from the root seed and a name.

        Parameters
        ----------
        name : str
            Unique identifier for the child stream (e.g. ``"noise_sampler"``).

        Returns
        -------
        numpy.random.Generator
            A fresh generator that is reproducible given ``(seed, name)``.
        """
        sub_seed = abs(hash((self.seed, name))) % (2**32)
        return np.random.default_rng(sub_seed)


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

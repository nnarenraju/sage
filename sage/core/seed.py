#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : seed.py
Description     : Short description of the file

Created on 2026-01-19 23:34:03

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
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

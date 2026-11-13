#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampling probabilities of a set of classes, keyed by class name.

Filename      : manager.py
Description   : Short description of the file

Created on 2026-01-19 16:47:21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import numpy as np

from typing import List, Type


class ProbabilityManager:
    """Sampling probabilities of classes, keyed by class name."""

    def __init__(self):
        self.probs = {}  # name -> probability

    def register(self, cls: Type, prob: float):
        """
        Assign a sampling probability to a class.

        Parameters
        ----------
        cls : type
            Class whose name is used as the registry key.
        prob : float
            Desired sampling probability in ``[0, 1]``.
        """
        name = cls.__name__
        self.probs[name] = prob

    def get_normalized_probs(self, classes: List[Type]) -> np.ndarray:
        """
        Sampling probabilities of ``classes``.

        Registered classes keep their probability; the others share
        ``max(0, 1 - registered total)`` equally. If every class is
        registered and the total is below 1, the total is normalised.

        Parameters
        ----------
        classes : list[type]
            Candidate classes.

        Returns
        -------
        numpy.ndarray
            One probability per class.
        """
        names = [cls.__name__ for cls in classes]
        probs = np.zeros(len(classes), dtype=float)
        assigned_total = 0.0
        unassigned_idx = []

        for i, name in enumerate(names):
            if name in self.probs:
                p = self.probs[name]
                probs[i] = p
                assigned_total += p
            else:
                unassigned_idx.append(i)

        remaining = max(0, 1.0 - assigned_total)
        if unassigned_idx:
            split = remaining / len(unassigned_idx)
            for i in unassigned_idx:
                probs[i] = split
        else:
            # All registered and summing below 1: normalise
            if assigned_total > 0 and assigned_total < 1:
                probs /= probs.sum()
        return probs

    def sample(self, classes: List[Type]) -> Type:
        """
        One class drawn from ``classes`` with ``get_normalized_probs``.

        Parameters
        ----------
        classes : list[type]
            Candidate classes to sample from.

        Returns
        -------
        type
            The sampled class.
        """
        probs = self.get_normalized_probs(classes)
        idx = np.random.choice(len(classes), p=probs)
        return classes[idx]

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : manager.py
Description   : Short description of the file

Created on 2026-01-19 16:47:21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import numpy as np

from typing import List, Type


class ProbabilityManager:
    """Internal manager to handle probabilities of classes/options."""

    def __init__(self):
        self.probs = {}  # name -> probability

    def register(self, cls: Type, prob: float):
        name = cls.__name__
        self.probs[name] = prob

    def get_normalized_probs(self, classes: List[Type]) -> np.ndarray:
        """Return normalized probability array for a list of classes"""
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
            # All assigned, sum <1 -> normalize
            if assigned_total > 0 and assigned_total < 1:
                probs /= probs.sum()
        return probs

    def sample(self, classes: List[Type]) -> Type:
        probs = self.get_normalized_probs(classes)
        idx = np.random.choice(len(classes), p=probs)
        return classes[idx]

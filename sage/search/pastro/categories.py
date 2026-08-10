#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : categories.py
Description   : Mixture components.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The search covers binary black holes only, so the mixture is a single astrophysical
component against terrestrial noise and the reported quantity is the probability that a
candidate is a binary black hole, not that it is astrophysical in general. The axis is
left extensible so further components can be added without reworking the inference.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Category:
    """One mixture component."""

    name: str
    astrophysical: bool
    description: str = ""


BBH = Category("BBH", True, "Binary black hole, within the searched mass range")
TERRESTRIAL = Category("Terrestrial", False, "Instrumental and environmental noise")

DEFAULT_CATEGORIES: Tuple[Category, ...] = (BBH, TERRESTRIAL)


def resolve(names: Sequence[str]) -> Tuple[Category, ...]:
    """Look up categories by name, raising with suggestions on a typo."""
    raise NotImplementedError


def astrophysical_names(categories: Sequence[Category]) -> Tuple[str, ...]:
    """Names of the astrophysical components."""
    raise NotImplementedError

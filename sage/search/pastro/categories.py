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
    """
    Look up categories by name, raising with suggestions on a typo.

    Suggestions rather than a bare KeyError: the names travel through configuration
    files and a mistyped component would otherwise surface as a missing column several
    stages later, where nothing points back at the typo that caused it.
    """
    import difflib

    known = {category.name: category for category in DEFAULT_CATEGORIES}
    folded = {name.lower(): name for name in known}
    out = []
    for name in names:
        if name in known:
            out.append(known[name])
            continue
        if name.lower() in folded:
            out.append(known[folded[name.lower()]])
            continue
        close = difflib.get_close_matches(name, known, n=3, cutoff=0.4)
        hint = f"; did you mean {close}?" if close else ""
        raise ValueError(
            f"unknown category {name!r}; known categories are {sorted(known)}{hint}"
        )
    if len(set(category.name for category in out)) != len(out):
        raise ValueError(
            f"categories {list(names)} name the same component more than once; the "
            "mixture would then carry two rates for one population"
        )
    return tuple(out)


def astrophysical_names(categories: Sequence[Category]) -> Tuple[str, ...]:
    """
    Names of the astrophysical components.

    What ``p_astro`` sums over. Sage's only astrophysical component is BBH, so the sum is
    a p_BBH and is documented as one; the plural is kept because the axis is extensible
    and a search that later adds a component must not need this rewritten.
    """
    return tuple(
        category.name for category in categories if bool(category.astrophysical)
    )




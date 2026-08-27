#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : ood.py
Description   : In-distribution / out-of-distribution classification of events.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

An event is in-distribution when enough of its mass posterior lies inside the box the
network was trained on. The fraction is computed from a random subsample of posterior
samples rather than a truncation of the sample list, since sample files are not always
stored in a random order.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OODResult:
    """Posterior mass inside the trained region."""

    id_fraction: float
    n_samples: int
    frame: str
    is_ood: bool

    def as_dict(self) -> dict:
        """Flat dict for the candidate table."""
        return {
            "id_fraction": float(self.id_fraction),
            "ood_n_samples": int(self.n_samples),
            "ood_frame": str(self.frame),
            "is_ood": bool(self.is_ood),
        }


def id_fraction(
    mass1: np.ndarray,
    mass2: np.ndarray,
    box: Tuple[float, float] = (7.0, 50.0),
    n_subsample: Optional[int] = 10000,
    seed: int = 0,
) -> OODResult:
    """
    Fraction of posterior mass with both components inside the trained box.

    sgwc-1's ``check_is_ood`` (``catalogue.ipynb`` cell 16), with its box: 7 to 50 solar
    masses on both components, and the ordering condition ``mass2 <= mass1`` that comes
    with it. The ordering term is part of the reference formula rather than a tidy-up --
    it is a no-op for a posterior file that already orders the components, and not for
    one that does not.

    An event is out of distribution when less than half its posterior mass lies inside,
    which is sgwc-1's threshold in the same cell.

    .. note::

       The subsample is random where sgwc-1 truncates to its first 10,000 samples. That
       is a deliberate departure, stated in this module's own docstring: posterior files
       are not always stored in a random order, and a head truncation of an ordered file
       measures a different region of the posterior than the posterior has. The count is
       sgwc-1's; the selection is not.
    """
    mass1 = np.asarray(mass1, dtype=np.float64).ravel()
    mass2 = np.asarray(mass2, dtype=np.float64).ravel()
    if mass1.size != mass2.size:
        raise ValueError(
            f"{mass1.size} primary masses against {mass2.size} secondary; paired "
            "componentwise, they would describe binaries neither posterior contains"
        )
    if mass1.size == 0:
        raise ValueError(
            "an empty posterior has no mass inside or outside the box; a read that "
            "returned nothing must not be reported as a verdict"
        )

    if n_subsample is not None and mass1.size > int(n_subsample):
        rng = np.random.default_rng(seed)
        taken = rng.choice(mass1.size, size=int(n_subsample), replace=False)
        mass1, mass2 = mass1[taken], mass2[taken]

    low, high = float(box[0]), float(box[1])
    inside = (
        (mass1 >= low)
        & (mass1 <= high)
        & (mass2 >= low)
        & (mass2 <= high)
        & (mass2 <= mass1)
    )
    fraction = float(np.count_nonzero(inside) / mass1.size)
    return OODResult(
        id_fraction=fraction,
        n_samples=int(mass1.size),
        frame="detector",
        is_ood=fraction < 0.5,
    )


def classify_event(
    posterior,
    box: Tuple[float, float] = (7.0, 50.0),
    frame: str = "detector",
    threshold: float = 0.5,
) -> OODResult:
    """
    Classify one event from its posterior samples.

    The frame matters: the network is trained on detector-frame masses, so a
    source-frame comparison mislabels redshifted events. Both are computed and reported.
    """
    raise NotImplementedError


def read_posterior_masses(path, frame: str = "detector") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load component masses from a posterior file.

    Raises on a missing or unreadable dataset rather than returning empty arrays, so a
    read failure cannot be mistaken for an out-of-distribution verdict.
    """
    raise NotImplementedError

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : candidates.py
Description   : The candidate table and the tiered inclusion criteria.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

This table is the single source of truth for every downstream table, figure and release
artefact; nothing downstream re-derives a candidate quantity.

Tiers follow the catalogue convention: a broad public candidate list, a confident subset,
and a further subset worth full parameter estimation.

Tiers are assigned twice. The search assigns them from significance and probability
alone, which is everything it can know without per-event work, and marks them
provisional. If candidates are later vetted, that verdict is recorded and the tiers are
re-derived, demoting anything the vetting rejected. A provisional tier is therefore an
upper bound on the final one, and the two are distinguishable so a provisional
classification is never mistaken for a vetted one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TIER_UNDETERMINED: int = -1
TIER_CANDIDATE: int = 0
TIER_CONFIDENT: int = 1
TIER_PE: int = 2

CANDIDATE_COLUMNS: Tuple[str, ...] = (
    "name",
    "gps",
    "tc_gps",
    "stat",
    # Single-arm significance. These keep their names and their meaning whether or not a
    # trials correction has been applied, so nothing downstream changes silently.
    "far_per_yr",
    "ifar_yr",
    "far_hierarchical_per_yr",
    "p_value",
    # Trials-corrected significance, carried alongside rather than replacing the above.
    # See sage.search.trials: the factor is per candidate, because a network only had a
    # chance at times it actually analysed.
    "n_trials",
    "trials_convention",
    "covered_by",
    "found_by",
    "best_arm",
    "far_trials_per_yr",
    "ifar_trials_yr",
    "p_value_trials",
    "p_astro",
    "p_astro_lo",
    "p_astro_hi",
    "mchirp",
    "mchirp_sigma",
    "observing_run",
    "detectors",
    "tier",
    "tier_trials",
    "dq_vetoed",
    "dq_p_value",
    "cat1_clean",
    "catalogue_match",
    "catalogue_dt_s",
    "id_fraction",
    "is_new",
)


@dataclass
class CandidateTable:
    """The candidate list for one observing run."""

    columns: Dict[str, np.ndarray]
    attrs: Dict[str, object]

    def __len__(self) -> int:
        """Number of candidates."""
        raise NotImplementedError

    def tier(self, tier: int) -> "CandidateTable":
        """Subset at or above a tier; raises if any row is undetermined."""
        raise NotImplementedError

    def new_events(self) -> "CandidateTable":
        """Candidates with no catalogue counterpart."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write ``candidates.h5``."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path, allow_undetermined: bool = False) -> "CandidateTable":
        """Read the candidate table."""
        raise NotImplementedError


def from_triggers(
    clustered: "object",
    far_curve,
    pastro_table: Optional[object] = None,
    spec=None,
) -> CandidateTable:
    """Assemble candidates from clustered zero-lag triggers, FAR and p_astro."""
    raise NotImplementedError


def apply_tiers(
    table: CandidateTable,
    far_candidate_per_day: float = 2.0,
    p_astro_confident: float = 0.5,
    far_pe_per_yr: float = 1.0,
    use_dataquality: bool = False,
) -> CandidateTable:
    """
    Assign tiers from significance and probability.

    Thresholds are strict inequalities so a candidate sitting exactly on a boundary is
    excluded rather than admitted.

    Both tier columns are written whenever the trials columns are present: ``tier`` from
    the single-arm rates and ``tier_trials`` from the corrected ones. They are kept
    separate rather than reconciled, because a candidate that qualifies under one and not
    the other is exactly what a reader needs to see.

    Parameters
    ----------
    use_dataquality : bool
        When false the result is provisional: no vetting verdict is required, and every
        row is marked as such. When true a verdict must be present for each candidate,
        and one that fails it is demoted regardless of its significance.
    """
    raise NotImplementedError


def retier(table: CandidateTable, dq_reports) -> CandidateTable:
    """
    Re-derive tiers using vetting verdicts gathered after the search.

    Only demotes: vetting can reject a candidate but never promote one, so a tier can
    fall when a verdict arrives and never rise.
    """
    raise NotImplementedError


def recompute_far(table: CandidateTable, far_curve) -> CandidateTable:
    """Re-assign FAR after a change to the background, without re-running inference."""
    raise NotImplementedError


def expected_contamination(table: CandidateTable) -> dict:
    """
    Expected number of noise events in a candidate set.

    The sum of ``1 - p_astro`` over a set is its expected terrestrial count, which is
    reported alongside any detection claim.
    """
    raise NotImplementedError

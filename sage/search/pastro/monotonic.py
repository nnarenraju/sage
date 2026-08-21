#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : monotonic.py
Description   : The likelihood-ratio monotonicity gate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The mixture treats the ranking statistic as an ordering of evidence, which holds only
while the signal-to-noise density ratio increases with it. Where the ratio is not
monotone, a threshold on the statistic is not a threshold on evidence: rates are then
driven by whichever region holds the most triggers, which is the quiet bulk rather than
the loud tail, and the result moves with the threshold instead of converging.

A network trained to classify at an operating point has no reason to be calibrated as a
likelihood ratio across its whole range, so this is measured before the rates are fit
and blocks the stage on failure.

Three responses are available: stop; restrict the analysis to the region where the ratio
is monotone; or re-express the statistic by the rank of its regressed likelihood ratio
and re-estimate both densities in the new variable. The last is a monotone change of
variable and leaves the mixture valid, provided both densities are rebuilt in it.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MonotonicityReport:
    """Whether the density ratio orders evidence, and where it fails."""

    stat: np.ndarray
    log_ratio: np.ndarray
    is_monotone: bool
    first_violation: Optional[float]
    largest_decrease: float
    monotone_region: Optional[Tuple[float, float]]

    def as_dict(self) -> dict:
        """Flat dict for the validation record."""
        return {
            "is_monotone": bool(self.is_monotone),
            "first_violation": (
                None if self.first_violation is None else float(self.first_violation)
            ),
            "largest_decrease": float(self.largest_decrease),
            "monotone_region": (
                None
                if self.monotone_region is None
                else [float(value) for value in self.monotone_region]
            ),
            "n_nodes": int(np.asarray(self.stat).size),
        }


def check_monotonicity(
    signal: object,
    noise: object,
    support,
    tolerance: float = 0.0,
) -> MonotonicityReport:
    """
    Evaluate the log density ratio across the support and test that it increases.

    The gate the whole stage rests on. ``log(p_s / p_n)`` is computed on the support's own
    grid -- the same nodes both densities are normalised over -- and required to be
    non-decreasing. Where it decreases, a louder trigger is *less* signal-like than a
    quieter one, so a cut on the statistic is not a cut on evidence and the low-statistic
    bulk, which outnumbers the tail by orders of magnitude, sets the inferred rate.

    ``tolerance`` allows a decrease of that size in the log ratio before the gate fails,
    for the numerical roughness of two kernel estimates divided by one another. It is not
    a way to pass a genuinely non-monotone ratio: the largest decrease is reported
    whatever the tolerance, so a caller can see what it was asked to ignore.

    Nodes where either density underflows to zero are dropped rather than compared. The
    ratio there is not small, it is undefined, and treating it as a number would put a
    violation wherever the support is wider than the data.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    nodes = support.grid()[0]
    log_ratio = np.asarray(signal.log_prob(nodes), dtype=np.float64) - np.asarray(
        noise.log_prob(nodes), dtype=np.float64
    )
    usable = np.isfinite(log_ratio)
    if usable.sum() < 2:
        raise ValueError(
            "fewer than two support nodes have both densities defined, so the ratio "
            "cannot be tested for ordering"
        )
    nodes, log_ratio = nodes[usable], log_ratio[usable]

    steps = np.diff(log_ratio)
    decreases = np.flatnonzero(steps < -tolerance)
    largest = float(-steps.min()) if steps.size and steps.min() < 0 else 0.0
    monotone = decreases.size == 0
    return MonotonicityReport(
        stat=nodes,
        log_ratio=log_ratio,
        is_monotone=bool(monotone),
        first_violation=None if monotone else float(nodes[decreases[0]]),
        largest_decrease=largest,
        monotone_region=largest_monotone_region(nodes, log_ratio),
    )


def largest_monotone_region(
    stat: np.ndarray, log_ratio: np.ndarray, min_span: float = 0.0
) -> Optional[Tuple[float, float]]:
    """
    Widest interval over which the ratio increases.

    Widest in the statistic rather than in node count, since the nodes are uniform here
    but the region is quoted as an interval and read as one. Ties are broken toward the
    higher region: the tail is where candidates are assessed, and a restriction that kept
    the bulk instead would keep the part of the range the search does not report from.
    """
    stat = np.asarray(stat, dtype=np.float64)
    log_ratio = np.asarray(log_ratio, dtype=np.float64)
    if stat.size < 2:
        return None
    breaks = np.flatnonzero(np.diff(log_ratio) < 0.0)
    edges = np.concatenate([[0], breaks + 1, [stat.size]])
    best = None
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop - start < 2:
            continue
        span = float(stat[stop - 1] - stat[start])
        if span < min_span:
            continue
        if best is None or span >= best[0]:
            best = (span, (float(stat[start]), float(stat[stop - 1])))
    return None if best is None else best[1]


def apply_policy(report: MonotonicityReport, policy: str = "restrict"):
    """
    Act on a failed gate according to the configured policy.

    ``stop`` raises and ``restrict`` returns the monotone region to narrow the support to.
    A passing report returns ``None`` under either policy: there is nothing to act on, and
    a policy that changed the analysis anyway would make the gate a transformation rather
    than a check.

    ``restrict`` is the default. It throws data away and fits nothing, so the result is
    still a statement about the region it covers -- which is the property that makes it
    safe to apply automatically.

    There is deliberately no third policy that reparameterises the statistic. An earlier
    revision offered one, fitting a monotone regression to the observed density ratio and
    re-expressing the statistic by it. That has no counterpart in any source of truth:
    PyCBC's ExpFitCombinedSNR applies a *fixed analytic* monotone function of a fitted
    noise rate rather than regressing one, every iDQ rank map is chosen a priori, and
    neither sgwc-1 nor FGMC, Banagiri or any GWTC methods paper transforms a statistic
    this way. A fit estimated from the same densities it then reparameterises is a place
    where the answer can come from the model instead of the data, and the gate exists to
    detect exactly that failure rather than to repair it.
    """
    if policy not in ("stop", "restrict"):
        raise ValueError(f"unknown policy {policy!r}, expected stop or restrict")
    if report.is_monotone:
        return None
    if policy == "stop":
        raise ValueError(
            "the signal-to-noise density ratio is not monotone in the ranking statistic, "
            f"first decreasing at {report.first_violation} by up to "
            f"{report.largest_decrease} in the log; a threshold on the statistic is then "
            "not a threshold on evidence, and the low-statistic bulk drives the inferred "
            "rate. Restrict to the monotone region, but do not fit rates through this"
        )
    if report.monotone_region is None:
        raise ValueError(
            "no interval of the support has a monotone ratio, so there is nothing "
            "to restrict to"
        )
    return report.monotone_region

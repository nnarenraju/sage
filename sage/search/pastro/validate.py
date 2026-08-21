#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : validate.py
Description   : The blocking validation suite.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

These checks decide whether the result is publishable, so they gate the stage rather
than accompanying it.

Threshold invariance is the central one. A candidate's probability should not depend on
where the analysis threshold was placed, and drift there is the visible symptom of a
failure elsewhere: an unclustered trigger set, mismatched truncation between components,
a density whose smoothing follows the observed extremes, or a non-monotone ratio. The
comparison is made against the credible intervals rather than a fixed number, so the
tolerance follows the precision actually achieved.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ValidationReport:
    """Outcome of the full suite."""

    checks: Dict[str, dict]
    passed: bool

    def failures(self) -> Tuple[str, ...]:
        """Names of the checks that failed."""
        return tuple(
            name
            for name, result in self.checks.items()
            if not bool(result.get("passed", False))
        )

    def as_dict(self) -> dict:
        """Flat dict for the stage record."""
        return {
            "passed": bool(self.passed),
            "failures": list(self.failures()),
            "checks": {
                name: {
                    key: (value.tolist() if hasattr(value, "tolist") else value)
                    for key, value in result.items()
                }
                for name, result in self.checks.items()
            },
        }


def analytic_oracle(
    signal_loc: float = 3.0, noise_rate: float = 1e4, signal_rate: float = 1e2, threshold: float = 0.0
) -> Dict[str, float]:
    """
    Recover known rates from a problem with a closed-form answer.

    With half-normal components truncated at the mode of the noise distribution, exactly
    half the noise events survive the threshold while the signal is barely affected, so
    the observable rates follow analytically and the estimator can be checked against them.

    Returns the expected observable rates rather than running the estimator, so a test can
    compare against a number derived on paper instead of against another run of the code
    being tested.
    """
    from scipy.stats import norm

    if noise_rate <= 0 or signal_rate <= 0:
        raise ValueError("both rates must be positive")
    # The noise is standard normal about zero, so a threshold at zero keeps exactly half.
    noise_survival = float(norm.sf(threshold))
    signal_survival = float(norm.sf(threshold - signal_loc))
    return {
        "noise_survival": noise_survival,
        "signal_survival": signal_survival,
        "observable_noise_rate": noise_rate * noise_survival,
        "observable_signal_rate": signal_rate * signal_survival,
        "threshold": float(threshold),
    }


def quadrature_oracle(
    stats: np.ndarray, densities: Dict[str, object], support
) -> Dict[str, float]:
    """
    Cross-check the gridded posterior against adaptive quadrature.

    Integrates the same posterior with ``scipy.integrate.quad`` in the original rate
    variables, which shares no code with the grid: a bug in the grid layout, the
    quadrature weights or the Jacobian moves one and not the other. Returns the
    normalising constant and the signal-rate mean from each route, and their fractional
    difference.
    """
    from scipy.integrate import dblquad

    from sage.search.pastro.rates import fit_rates

    stats = np.asarray(stats, dtype=np.float64).ravel()
    signal, noise = tuple(densities)
    log_ps = np.asarray(densities[signal].log_prob(stats), dtype=np.float64)
    log_pn = np.asarray(densities[noise].log_prob(stats), dtype=np.float64)
    n = stats.size

    def log_joint(rate_s: float, rate_n: float) -> float:
        """Eq. (21) in the original rate variables, with the Jeffreys prior."""
        if rate_s <= 0 or rate_n <= 0:
            return -np.inf
        mixture = np.logaddexp(
            np.log(rate_s) + log_ps, np.log(rate_n) + log_pn
        ).sum()
        return mixture - rate_s - rate_n - 0.5 * np.log(rate_s) - 0.5 * np.log(rate_n)

    posterior = fit_rates(stats, densities, support, clustered=True, n_grid=256)
    peak = float(posterior.log_posterior.max())
    hi_s = max(4.0 * n, 20.0)
    hi_n = max(4.0 * n, 20.0)
    evidence = dblquad(
        lambda rn, rs: np.exp(log_joint(rs, rn) - peak),
        1e-9, hi_s, lambda _: 1e-9, lambda _: hi_n,
        epsabs=1e-10, epsrel=1e-8,
    )[0]
    first = dblquad(
        lambda rn, rs: rs * np.exp(log_joint(rs, rn) - peak),
        1e-9, hi_s, lambda _: 1e-9, lambda _: hi_n,
        epsabs=1e-10, epsrel=1e-8,
    )[0]
    quad_mean = first / evidence
    grid_mean = float(posterior.mean_rates[signal])
    return {
        "quadrature_signal_mean": float(quad_mean),
        "grid_signal_mean": grid_mean,
        "fractional_difference": float((grid_mean - quad_mean) / quad_mean),
    }


def threshold_invariance(
    triggers,
    densities_at: Dict[float, Dict[str, object]],
    thresholds: Sequence[float],
    k_sigma: float = 3.0,
) -> Dict[str, object]:
    """
    Refit at several thresholds and compare a common candidate's probability.

    Agreement is judged against the combined credible intervals, so the test tightens as
    the estimate becomes more precise instead of resting on a fixed allowance.

    Parameters
    ----------
    triggers : ndarray
        Statistics of the full trigger set. Each threshold analyses the subset above it.
    densities_at : dict
        Threshold to the ``{category: density}`` pair built on that threshold's support.
        Built by the caller, because rebuilding them here would hide the very step --
        re-truncating both components on the new support -- that this check exists to
        exercise.
    k_sigma : float
        How many combined standard errors two probabilities may differ by. The interval
        is used as the error, so the allowance follows the precision achieved.
    """
    from sage.search.pastro.assign import assign_pastro
    from sage.search.pastro.rates import fit_rates

    triggers = np.asarray(triggers, dtype=np.float64).ravel()
    thresholds = list(thresholds)
    if len(thresholds) < 2:
        raise ValueError("threshold invariance needs at least two thresholds")
    probe = float(triggers.max())
    values, widths = [], []
    for threshold in thresholds:
        densities = densities_at[threshold]
        support = next(iter(densities.values())).support
        kept = triggers[triggers >= threshold]
        posterior = fit_rates(kept, densities, support, clustered=True, n_grid=256)
        table = assign_pastro(np.array([probe]), densities, posterior)
        signal = posterior.categories[0]
        values.append(float(table.probabilities[signal][0]))
        widths.append(
            0.5 * float(table.upper[signal][0] - table.lower[signal][0])
        )
    values = np.asarray(values)
    widths = np.asarray(widths)
    worst, passed = 0.0, True
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            combined = np.sqrt(widths[i] ** 2 + widths[j] ** 2)
            deviation = abs(values[i] - values[j]) / max(combined, 1e-12)
            worst = max(worst, float(deviation))
            passed &= deviation <= k_sigma
    return {
        "thresholds": [float(value) for value in thresholds],
        "probe_stat": probe,
        "values": values.tolist(),
        "half_widths": widths.tolist(),
        "worst_deviation": worst,
        "k_sigma": float(k_sigma),
        "passed": bool(passed),
    }


def convergence_with_background(
    triggers, background_subsets: Sequence[np.ndarray], densities_builder
) -> Dict[str, object]:
    """
    Track a candidate's probability as background is accumulated.

    The value should settle and its interval should narrow. Continued drift indicates
    that the density estimate is following the sample extremes rather than converging.

    This is the property the user observed failing in the reference analysis: a bandwidth
    or a truncation tied to the largest observed background statistic moves the whole
    noise model every time more background arrives, so the answer never settles however
    much is accumulated.
    """
    from sage.search.pastro.assign import assign_pastro
    from sage.search.pastro.rates import fit_rates

    triggers = np.asarray(triggers, dtype=np.float64).ravel()
    probe = float(triggers.max())
    values, widths, sizes = [], [], []
    for subset in background_subsets:
        densities = densities_builder(np.asarray(subset, dtype=np.float64))
        support = next(iter(densities.values())).support
        posterior = fit_rates(triggers, densities, support, clustered=True, n_grid=256)
        table = assign_pastro(np.array([probe]), densities, posterior)
        signal = posterior.categories[0]
        values.append(float(table.probabilities[signal][0]))
        widths.append(float(table.upper[signal][0] - table.lower[signal][0]))
        sizes.append(int(np.asarray(subset).size))
    values, widths = np.asarray(values), np.asarray(widths)
    steps = np.abs(np.diff(values))
    settling = bool(steps.size < 2 or steps[-1] <= steps[0] + 1e-12)
    narrowing = bool(widths.size < 2 or widths[-1] <= widths[0] + 1e-12)
    return {
        "n_background": sizes,
        "values": values.tolist(),
        "widths": widths.tolist(),
        "steps": steps.tolist(),
        "settling": settling,
        "narrowing": narrowing,
        "passed": bool(settling and narrowing),
    }


def run_suite(
    triggers, densities, posterior, support, tolerance: Optional[Dict[str, float]] = None
) -> ValidationReport:
    """
    Run every check that needs only the fitted model, and return the combined verdict.

    Threshold invariance and convergence are not run here: both need models refitted on
    other thresholds and other background subsets, which this function is not given and
    must not invent. They are exposed separately and called by the stage that holds those
    inputs.
    """
    from sage.search.pastro.assign import assign_pastro, sum_consistency
    from sage.search.pastro.density import verify_normalisation
    from sage.search.pastro.monotonic import check_monotonicity

    tolerance = dict(tolerance or {})
    checks: Dict[str, dict] = {}

    normalisation = {}
    for name, density in densities.items():
        try:
            normalisation[name] = verify_normalisation(
                density, atol=tolerance.get("normalisation", 1e-3)
            )
        except ValueError as error:
            normalisation[name] = str(error)
    checks["normalisation"] = {
        "values": normalisation,
        "passed": all(isinstance(value, float) for value in normalisation.values()),
    }

    signal, noise = posterior.categories
    report = check_monotonicity(
        densities[signal], densities[noise], support,
        tolerance=tolerance.get("monotonicity", 0.0),
    )
    checks["monotonicity"] = {**report.as_dict(), "passed": bool(report.is_monotone)}

    triggers = np.asarray(triggers, dtype=np.float64).ravel()
    table = assign_pastro(triggers, densities, posterior)
    consistency = sum_consistency(table, posterior)
    # Judged against inferred - 1/2, which is the exact expectation under the Jeffreys
    # prior, not against inferred. See sum_consistency: gating on the naive residual
    # fails a correct run whenever the inferred rate is below about five.
    checks["sum_consistency"] = {
        **consistency,
        "passed": abs(consistency["fractional"])
        <= tolerance.get("sum_consistency", 0.1),
    }

    probabilities = table.probabilities[signal]
    checks["bounded"] = {
        "min": float(probabilities.min()),
        "max": float(probabilities.max()),
        "passed": bool(probabilities.min() >= 0.0 and probabilities.max() <= 1.0),
    }

    return ValidationReport(
        checks=checks,
        passed=all(bool(check["passed"]) for check in checks.values()),
    )

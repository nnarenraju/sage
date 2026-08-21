#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : assign.py
Description   : Per-trigger astrophysical probability.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Each trigger's probability is the component's share of the mixture at its position,
averaged over the rate posterior, following Eq. (11) of
``docs/references/arxiv_2305.00071.pdf``::

    p_astro(x) = int dLs dLn  [ Ls p(x|S) / ( Ls p(x|S) + Ln p(x|0) ) ]
                              * p(Ls, Ln | {x}, N)

The average is taken over the full rate grid rather than at a point estimate, which is
what produces a credible interval alongside the value. Section V of that reference adopts
a preliminary cut of one false alarm per half day when applying this to real triggers;
the equivalent threshold here is set once in :mod:`sage.search.pastro.support` and shared
by every component density.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from sage.search.pastro.rates import RatePosterior


# The exact offset between the summed probability and the inferred signal rate under the
# Jeffreys prior; see :func:`sum_consistency`. The same half that puts the FGMC rate
# posterior's mode at N - 1/2.
JEFFREYS_SUM_OFFSET: float = 0.5


@dataclass
class PAstroTable:
    """Per-trigger probabilities with credible intervals."""

    gps: np.ndarray
    stat: np.ndarray
    mchirp: Optional[np.ndarray]
    probabilities: Dict[str, np.ndarray]
    lower: Dict[str, np.ndarray]
    upper: Dict[str, np.ndarray]
    attrs: Dict[str, object]

    def __len__(self) -> int:
        """Number of triggers."""
        return int(np.asarray(self.stat).size)

    def astrophysical(self) -> np.ndarray:
        """
        Summed probability over the astrophysical components.

        For Sage that is the BBH component alone, so the result is a p_BBH and is
        recorded as one: the search has no model for any other astrophysical population,
        and a candidate outside the searched mass range is not represented by either
        component.
        """
        from sage.search.pastro.categories import DEFAULT_CATEGORIES

        astro = {
            category.name for category in DEFAULT_CATEGORIES if category.astrophysical
        }
        present = [name for name in self.probabilities if name in astro]
        if not present:
            raise ValueError(
                f"none of {sorted(self.probabilities)} is an astrophysical component, "
                "so there is no astrophysical probability to sum"
            )
        return np.sum([self.probabilities[name] for name in present], axis=0)

    def save(self, path: str | Path) -> None:
        """
        Write the per-trigger table.

        Each component's probability keeps its own dataset under its category name, so
        adding a component later is a new dataset rather than a change of column order --
        which a positional layout would make silent. The credible bounds are stored beside
        the values rather than derived on read: they come from the rate posterior, and a
        reader holding only this table cannot recompute them.
        """
        from sage.utils.atomic_io import atomic_h5

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_h5(target, mode="w") as handle:
            for key, value in (self.attrs or {}).items():
                handle.attrs[key] = value
            handle.attrs["categories"] = sorted(self.probabilities)
            handle.create_dataset("gps", data=np.asarray(self.gps, dtype=np.float64))
            handle.create_dataset("stat", data=np.asarray(self.stat, dtype=np.float64))
            if self.mchirp is not None:
                handle.create_dataset(
                    "mchirp", data=np.asarray(self.mchirp, dtype=np.float64)
                )
            for name, group in (
                ("probability", self.probabilities),
                ("lower", self.lower),
                ("upper", self.upper),
            ):
                target_group = handle.create_group(name)
                for category, values in group.items():
                    target_group.create_dataset(
                        category, data=np.asarray(values, dtype=np.float64)
                    )

    @classmethod
    def load(cls, path: str | Path) -> "PAstroTable":
        """
        Read a persisted table.

        The category list is taken from the file's own attribute rather than from
        whatever groups happen to be present, so a table written with a component that
        this build does not know about is read with that component intact instead of
        quietly dropped.
        """
        import h5py

        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"no p_astro table at {target}")
        with h5py.File(target, "r") as handle:
            if "categories" not in handle.attrs:
                raise ValueError(
                    f"{target} records no category list, so which components its "
                    "probabilities belong to cannot be established"
                )
            categories = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in handle.attrs["categories"]
            ]
            missing = [
                f"{group}/{name}"
                for group in ("probability", "lower", "upper")
                for name in categories
                if group not in handle or name not in handle[group]
            ]
            if missing:
                raise ValueError(
                    f"{target} declares categories {categories} but is missing "
                    f"{missing}; the file was truncated part-way through a write"
                )
            return cls(
                gps=np.asarray(handle["gps"]),
                stat=np.asarray(handle["stat"]),
                mchirp=(
                    np.asarray(handle["mchirp"]) if "mchirp" in handle else None
                ),
                probabilities={
                    name: np.asarray(handle["probability"][name])
                    for name in categories
                },
                lower={name: np.asarray(handle["lower"][name]) for name in categories},
                upper={name: np.asarray(handle["upper"][name]) for name in categories},
                attrs={
                    key: handle.attrs[key]
                    for key in handle.attrs
                    if key != "categories"
                },
            )


def assign_pastro(
    stats: np.ndarray,
    densities: Dict[str, object],
    posterior: RatePosterior,
    mchirp: Optional[np.ndarray] = None,
    credible_level: float = 0.9,
    gps: Optional[np.ndarray] = None,
) -> PAstroTable:
    """
    Evaluate each trigger's component probabilities, marginalised over the rates.

    Eq. (11) of ``arxiv_2305.00071``::

        p_astro(x) = int dLs dLn  [ Ls p(x|S) / (Ls p(x|S) + Ln p(x|0)) ]
                                  * p(Ls, Ln | {x}, N)

    Averaged over the whole rate posterior, not evaluated at a point estimate. That is
    what produces the credible interval reported beside the value: the spread comes from
    the rates being uncertain, and a plug-in estimate would report a probability with no
    uncertainty attached to it at all.

    The total rate drops out of the integrand exactly. Writing the ratio in the
    parameterisation the posterior was computed in::

        Ls p_s / (Ls p_s + Ln p_n)  =  1 / (1 + ((1 - a) / a) * (p_n / p_s))

    has no ``lam`` in it, so integrating over the total rate is integrating a constant and
    the double integral collapses to a single sum over the fraction axis, weighted by the
    fraction's own marginal. This is an identity rather than an approximation, and it is
    what makes the assignment cheap enough to run over a whole campaign's triggers.

    Parameters
    ----------
    credible_level : float
        Central coverage of the reported interval, from the same posterior weights.
    """
    if not 0.0 < credible_level < 1.0:
        raise ValueError(
            f"credible_level must lie in (0, 1), got {credible_level}"
        )
    stats = np.asarray(stats, dtype=np.float64).ravel()
    if stats.size == 0:
        raise ValueError("no triggers were given to assign probabilities to")
    signal, noise = posterior.categories
    if set(densities) != {signal, noise}:
        raise ValueError(
            f"the densities describe {sorted(densities)} but the posterior was inferred "
            f"for {sorted((signal, noise))}; the two would be paired by position"
        )
    # Refused, not silently NaN. Outside the support both densities are zero, so the log
    # odds is inf - inf; without this the loudest candidate in a campaign -- the one this
    # stage exists to assess -- comes back NaN with nothing but a RuntimeWarning. The
    # support has to be built to reach the candidates: see build_support's `must_include`.
    support = getattr(densities[signal], "support", None)
    if support is not None:
        outside = ~(
            support.contains(stats, mchirp)
            if mchirp is not None
            else support.contains(stats)
        )
        if outside.any():
            worst = float(np.max(stats[outside]))
            raise ValueError(
                f"{int(outside.sum())} of {stats.size} triggers lie outside the common "
                f"support [{support.stat_lo}, {support.stat_hi}], the loudest at {worst}. "
                "Both densities are zero there, so no ratio exists and the probability "
                "would be NaN. Rebuild the support with build_support(..., "
                "must_include=<candidate and injection statistics>): a candidate is "
                "confident because it is louder than the background, so a support bounded "
                "by the background excludes exactly the candidates worth assessing"
            )
    log_ps = np.asarray(densities[signal].log_prob(stats, mchirp), dtype=np.float64)
    log_pn = np.asarray(densities[noise].log_prob(stats, mchirp), dtype=np.float64)

    fraction = posterior.fraction_grid
    # Marginal of the signal fraction: the total rate has already integrated out.
    weights = posterior.weights.sum(axis=0)
    weights = weights / float(weights.sum())

    # log[(1 - a)/a] + log[p_n/p_s], so the odds are formed without ever exponentiating
    # a density that may be many orders of magnitude from one.
    log_odds = (
        np.log1p(-fraction)[None, :]
        - np.log(fraction)[None, :]
        + (log_pn - log_ps)[:, None]
    )
    pointwise = 1.0 / (1.0 + np.exp(np.clip(log_odds, -700.0, 700.0)))
    values = pointwise @ weights

    order = np.argsort(pointwise, axis=1)
    ordered = np.take_along_axis(pointwise, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    cumulative /= cumulative[:, -1][:, None]
    tail = 0.5 * (1.0 - credible_level)
    lower = np.array(
        [np.interp(tail, cumulative[k], ordered[k]) for k in range(stats.size)]
    )
    upper = np.array(
        [np.interp(1.0 - tail, cumulative[k], ordered[k]) for k in range(stats.size)]
    )
    # The reported interval must contain the reported value. Two things can put the mean
    # outside equal-tailed quantiles: rounding, where a saturated probability sits a
    # float above its own quantile, and genuine skew, where most of the fraction
    # posterior gives near-zero and a thin tail of it gives near-one. Both are resolved
    # the same way, by widening rather than by moving the value -- Eq. (11) defines
    # p_astro as the mean, so the mean is what is quoted.
    lower = np.minimum(lower, values)
    upper = np.maximum(upper, values)
    return PAstroTable(
        gps=(
            np.full(stats.size, np.nan)
            if gps is None
            else np.asarray(gps, dtype=np.float64).ravel()
        ),
        stat=stats,
        mchirp=None if mchirp is None else np.asarray(mchirp, dtype=np.float64).ravel(),
        probabilities={signal: values, noise: 1.0 - values},
        lower={signal: lower, noise: 1.0 - upper},
        upper={signal: upper, noise: 1.0 - lower},
        attrs={
            "credible_level": float(credible_level),
            "n_triggers": int(posterior.n_triggers),
            "prior": str(posterior.prior),
            "categories": tuple(posterior.categories),
        },
    )


def sum_consistency(table: PAstroTable, posterior: RatePosterior) -> Dict[str, float]:
    """
    Compare the summed probability against the inferred rate.

    Each term is the posterior probability that one trigger is signal, so the sum is the
    expected number of signals -- which is what the rate parameter means. A disagreement
    means the densities and the rate inference describe different data, the usual cause
    being an assignment run over a different trigger set from the one the rates were
    fitted to. This is the check sgwc-1 printed at 58.78 per cent and did not act on.

    **The expected value is not zero, it is minus one half.** Under the Jeffreys prior,

        sum_i p_astro_i  =  E[Ls] - 1/2

    holds exactly, not approximately: integrating ``d/dLs [Ls p(Ls, Ln)]`` to zero over
    the posterior gives ``1 + E[sum_i p_i] - E[Ls] - 1/2 = 0``. The half is the same half
    that puts the FGMC rate posterior's mode at ``N - 1/2`` -- it is contributed by the
    prior, not by any disagreement in the data. PyCBC's ``count_posterior`` obeys the same
    identity with the constant ``-(alpha + 1)`` at power-law prior ``alpha``.

    So the residual to judge is ``summed - (inferred - 1/2)``, and ``fractional`` is that
    residual over the inferred rate. Gating on ``summed - inferred`` instead would fail a
    perfectly correct run whenever ``E[Ls] < 5``, since the constant alone is then more
    than ten per cent -- and that is precisely the low-count regime a real search sits in.

    Returns
    -------
    dict
        ``summed``, ``inferred`` (posterior mean), ``expected`` (``inferred - 1/2``),
        ``difference`` (against ``expected``), ``fractional``, and ``raw_difference``
        against the naive zero, kept because it is what the reference analysis reported.
    """
    signal = posterior.categories[0]
    summed = float(np.sum(table.probabilities[signal]))
    inferred = float(posterior.mean_rates[signal])
    expected = inferred - JEFFREYS_SUM_OFFSET
    difference = summed - expected
    return {
        "summed": summed,
        "inferred": inferred,
        "expected": expected,
        "difference": difference,
        "fractional": difference / inferred if inferred > 0 else float("nan"),
        "raw_difference": summed - inferred,
    }

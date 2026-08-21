#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : rates.py
Description   : Joint inference of the component rates.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The rates are inferred from the trigger set itself under a Poisson mixture and then
marginalised over, so the reported probability reflects how well the rates are known.

The estimator is the one derived in ``docs/references/arxiv_1302.5341.pdf`` (Farr et al.,
"Counting And Confusion"). Its Eq. (12) gives the likelihood conditioned on per-event
foreground/background flags; marginalising those flags gives the rate posterior, Eq. (21)::

    p(Rf, Rb, th | d, N)  proportional to
        prod_i [ Rf fhat(x_i, th) + Rb bhat(x_i, th) ]
        * exp[-(Rf + Rb)] * p(th) / sqrt(Rf * Rb)

The trailing ``1/sqrt(Rf Rb)`` is the Jeffreys prior on the two rates. The same posterior
appears as Eq. (10) of ``docs/references/arxiv_2305.00071.pdf`` in count form, with
``Lambda_s = R_s T`` and ``Lambda_n = R_n T`` from its Eq. (4).

A check on any implementation: in the foreground-dominated limit Eq. (35) of the same
reference reduces the posterior to ``Rf^(N - 1/2) exp(-Rf)``, peaked at ``Rf = N - 1/2``,
where the half is contributed by the Jeffreys prior.

The inference is parameterised by the total rate and the fraction belonging to each
component. Working directly in the individual rates loses precision when one component
outnumbers the other by many orders of magnitude, which is the normal situation here; the
change of variables carries a Jacobian that must be applied with the prior.

The likelihood assumes independent triggers, so the input must already be clustered; the
constructor refuses an unclustered set rather than silently producing a rate inflated by
the number of windows per event. The grid is bracketed automatically from the data, so
the answer cannot depend on a hand-chosen range.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _fraction_nodes(n_grid: int) -> np.ndarray:
    """
    Signal-fraction nodes, spaced so the Jeffreys prior is uniform between them.

    The prior on the fraction is ``1 / sqrt(a * (1 - a))``, which diverges at both ends.
    The divergence is integrable, but a uniform grid resolves it badly exactly where the
    background-dominated solution lives -- ``a`` near zero is the normal situation for a
    search. Under ``a = sin(u)**2`` the prior becomes flat in ``u``, so uniform nodes in
    ``u`` place the fraction nodes where the mass is and the singular factor cancels
    against the Jacobian instead of being sampled.

    The endpoints are excluded: ``a = 0`` and ``a = 1`` are where the prior is infinite,
    and they are also the two statements the data can never support -- no signals at all,
    or no noise at all.
    """
    if n_grid < 2:
        raise ValueError(f"n_grid must be at least 2, got {n_grid}")
    u = np.linspace(0.0, 0.5 * np.pi, int(n_grid) + 2)[1:-1]
    return np.sin(u) ** 2


def _fraction_weights(fraction: np.ndarray) -> np.ndarray:
    """
    Quadrature weights in the fraction, for nodes laid out by :func:`_fraction_nodes`.

    ``da = sin(2u) du``. Multiplying uniform ``du`` weights by that Jacobian integrates in
    ``a`` while keeping the accuracy the substitution bought.
    """
    fraction = np.asarray(fraction, dtype=np.float64)
    u = np.arcsin(np.sqrt(np.clip(fraction, 0.0, 1.0)))
    if u.size < 2:
        raise ValueError("at least two fraction nodes are needed to integrate")
    du = float(u[1] - u[0])
    return du * np.sin(2.0 * u)


def _trapezoid_weights(nodes: np.ndarray) -> np.ndarray:
    """Trapezoid weights on a uniformly spaced axis."""
    nodes = np.asarray(nodes, dtype=np.float64)
    step = float(nodes[1] - nodes[0])
    out = np.full(nodes.size, step, dtype=np.float64)
    out[0] = out[-1] = 0.5 * step
    return out


@dataclass
class RatePosterior:
    """Posterior over the component rates."""

    categories: Tuple[str, ...]
    total_grid: np.ndarray
    fraction_grid: np.ndarray
    log_posterior: np.ndarray
    n_triggers: int
    clustered: bool
    prior: str

    def __post_init__(self) -> None:
        """Coerce the grids and refuse a posterior that cannot be integrated."""
        self.total_grid = np.asarray(self.total_grid, dtype=np.float64).ravel()
        self.fraction_grid = np.asarray(self.fraction_grid, dtype=np.float64).ravel()
        self.log_posterior = np.asarray(self.log_posterior, dtype=np.float64)
        expected = (self.total_grid.size, self.fraction_grid.size)
        if self.log_posterior.shape != expected:
            raise ValueError(
                f"the posterior has shape {self.log_posterior.shape} against grids of "
                f"{expected}; it would be summed against the wrong axis"
            )
        if not bool(self.clustered):
            raise ValueError(
                "this posterior was built from an unclustered trigger set: the mixture "
                "likelihood treats triggers as independent draws, and an unclustered "
                "glitch contributes one draw per window instead of one, inflating every "
                "inferred rate by the number of windows per event"
            )

    @property
    def weights(self) -> np.ndarray:
        """
        Normalised posterior mass per grid cell, summing to one.

        Mass rather than density: every integral here is a weighted sum over the same
        cells, so carrying the quadrature weights once and normalising is what keeps the
        rate, the interval and the per-candidate probability consistent with one another.
        """
        cell = np.outer(
            _trapezoid_weights(self.total_grid), _fraction_weights(self.fraction_grid)
        )
        mass = np.exp(self.log_posterior - float(self.log_posterior.max())) * cell
        return mass / float(mass.sum())

    @property
    def _rates(self) -> Dict[str, np.ndarray]:
        """Each component's rate at every grid cell."""
        total = self.total_grid[:, None]
        fraction = self.fraction_grid[None, :]
        signal, noise = self.categories
        return {signal: total * fraction, noise: total * (1.0 - fraction)}

    @property
    def map_rates(self) -> Dict[str, float]:
        """
        Rates at the posterior mode.

        The mode of the joint posterior in the parameterisation it was computed in, which
        is the quantity the closed forms in the references are stated for.
        """
        flat = int(np.argmax(self.log_posterior))
        i, j = np.unravel_index(flat, self.log_posterior.shape)
        return {name: float(values[i, j]) for name, values in self._rates.items()}

    @property
    def mean_rates(self) -> Dict[str, float]:
        """
        Posterior mean of each component rate.

        A property, matching :attr:`map_rates`. As a method it was the one accessor on
        this class that needed calling, and ``posterior.mean_rates`` then evaluated to a
        bound method -- truthy, and formattable straight into a report as
        ``<bound method ...>`` rather than failing.
        """
        weights = self.weights
        return {
            name: float(np.sum(values * weights))
            for name, values in self._rates.items()
        }

    def marginal(self, category: str, n_nodes: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Marginal posterior for one component rate, over its own axis.

        Neither component rate is a grid axis -- the grid is total and fraction -- so the
        joint is mapped back onto ``(Ls, Ln)`` and integrated over the other component.
        Binning the grid cells into a histogram of ``Ls`` instead would resolve the mode
        only to a bin, and the mode is what the closed form of Eq. (35) is stated for.

        The map carries the inverse Jacobian: ``p(Ls, Ln) = p(lam, a) / lam``.

        Returns
        -------
        tuple
            ``(nodes, density)``, the density normalised on those nodes.
        """
        rates = self._rates
        if category not in rates:
            raise ValueError(
                f"unknown category {category!r}; this posterior holds {sorted(rates)}"
            )
        from scipy.interpolate import RegularGridInterpolator

        other = next(name for name in rates if name != category)
        span = {
            name: self.credible_interval(name, level=1.0 - 1e-9) for name in rates
        }
        mine = np.linspace(max(span[category][0], 1e-9), span[category][1], int(n_nodes))
        theirs = np.linspace(max(span[other][0], 1e-9), span[other][1], int(n_nodes))
        grid_mine, grid_theirs = np.meshgrid(mine, theirs, indexing="ij")
        total = grid_mine + grid_theirs
        signal = self.categories[0]
        fraction = (grid_mine if category == signal else grid_theirs) / total

        interpolate = RegularGridInterpolator(
            (self.total_grid, self.fraction_grid),
            self.log_posterior - float(self.log_posterior.max()),
            bounds_error=False,
            fill_value=-np.inf,
        )
        log_values = interpolate(np.stack([total.ravel(), fraction.ravel()], axis=-1))
        values = np.exp(log_values).reshape(total.shape) / total
        density = np.trapezoid(values, theirs, axis=1)
        area = float(np.trapezoid(density, mine))
        return mine, (density / area if area > 0 else density)

    def credible_interval(
        self, category: str, level: float = 0.9
    ) -> Tuple[float, float]:
        """
        Credible interval on one component's rate.

        Equal-tailed, so ``level`` of the mass lies inside and half the remainder outside
        each end. A highest-density interval would be narrower but is not monotone in the
        level, and these are quoted beside one another across levels.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        rates = self._rates
        if category not in rates:
            raise ValueError(f"unknown category {category!r}")
        values = rates[category].ravel()
        weights = self.weights.ravel()
        order = np.argsort(values)
        ordered, mass = values[order], weights[order]
        cumulative = np.cumsum(mass)
        cumulative /= cumulative[-1]
        tail = 0.5 * (1.0 - level)
        return (
            float(np.interp(tail, cumulative, ordered)),
            float(np.interp(1.0 - tail, cumulative, ordered)),
        )

    def save(self, path: str | Path) -> None:
        """
        Write the rate posterior and its provenance.

        The log posterior is stored, not the normalised weights. Weights are derived from
        it through :attr:`weights`, which applies the quadrature the grids imply; storing
        them instead would freeze one quadrature into the file and let a reader integrate
        the same posterior differently from the run that produced it.

        ``clustered`` is written because :meth:`__post_init__` refuses to reconstruct an
        unclustered posterior, and a file that did not record the flag could only be
        reloaded by assuming the answer to the question the flag exists to ask.
        """
        from sage.utils.atomic_io import atomic_h5

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_h5(target, mode="w") as handle:
            handle.attrs["categories"] = list(self.categories)
            handle.attrs["n_triggers"] = int(self.n_triggers)
            handle.attrs["clustered"] = bool(self.clustered)
            handle.attrs["prior"] = str(self.prior)
            handle.create_dataset("total_grid", data=self.total_grid)
            handle.create_dataset("fraction_grid", data=self.fraction_grid)
            handle.create_dataset(
                "log_posterior", data=self.log_posterior, compression="gzip"
            )

    @classmethod
    def load(cls, path: str | Path) -> "RatePosterior":
        """
        Read a persisted rate posterior.

        Rebuilt through the constructor, so a file edited into an inconsistent state --
        a posterior whose shape no longer matches its grids, or one recorded as
        unclustered -- is refused here rather than producing rates from it.
        """
        import h5py

        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"no rate posterior at {target}")
        with h5py.File(target, "r") as handle:
            missing = [
                name
                for name in ("categories", "n_triggers", "clustered", "prior")
                if name not in handle.attrs
            ]
            if missing:
                raise ValueError(
                    f"{target} carries no {missing}; it is not a complete rate "
                    "posterior and no rate taken from it can be attributed"
                )
            return cls(
                categories=tuple(
                    v.decode() if isinstance(v, bytes) else str(v)
                    for v in handle.attrs["categories"]
                ),
                total_grid=np.asarray(handle["total_grid"]),
                fraction_grid=np.asarray(handle["fraction_grid"]),
                log_posterior=np.asarray(handle["log_posterior"]),
                n_triggers=int(handle.attrs["n_triggers"]),
                clustered=bool(handle.attrs["clustered"]),
                prior=str(handle.attrs["prior"]),
            )


def fit_rates(
    stats: np.ndarray,
    densities: Dict[str, object],
    support,
    mchirp: Optional[np.ndarray] = None,
    clustered: bool = False,
    prior: str = "jeffreys",
    n_grid: int = 512,
) -> RatePosterior:
    """
    Infer the component rates from the observed triggers.

    Evaluates Eq. (21) of ``arxiv_1302.5341`` -- Eq. (10) of ``arxiv_2305.00071`` in count
    form -- on a grid in total rate and signal fraction::

        log p(lam, a | d) = -lam + N log lam + log_prior(lam, a)
                            + sum_i log[ a * p_s(x_i) + (1 - a) * p_n(x_i) ]

    The mixture term is accumulated through ``logaddexp`` rather than as a product. With
    a background outnumbering the signal by many orders of magnitude, the product
    underflows long before the posterior does, and the failure is silent: the grid fills
    with zeros and the mode lands wherever the underflow stopped.

    Parameters
    ----------
    clustered : bool
        Must be true. The mixture likelihood treats triggers as independent draws, and an
        unclustered glitch supplies one draw per window rather than one.
    prior : str
        ``jeffreys`` by default, with the change-of-variable factor applied once; see
        :func:`log_prior`.
    """
    if not clustered:
        raise ValueError(
            "fit_rates needs a clustered trigger set: the mixture likelihood treats "
            "triggers as independent draws, so an unclustered glitch would contribute "
            "one draw per window and inflate the inferred rate by that factor"
        )
    stats = np.asarray(stats, dtype=np.float64).ravel()
    if stats.size == 0:
        raise ValueError("no triggers were given, so no rate can be inferred from them")
    if not np.isfinite(stats).all():
        raise ValueError("the trigger statistics contain a non-finite value")
    names = tuple(densities)
    if len(names) != 2:
        raise ValueError(
            f"the mixture is over exactly two components, got {list(names)}"
        )
    inside = support.contains(stats, mchirp) if mchirp is not None else (
        support.contains(stats)
    )
    if not inside.all():
        raise ValueError(
            f"{int((~inside).sum())} of {stats.size} triggers lie outside the common "
            "support; they were not analysed under the threshold the densities were "
            "built on and would be scored against a model that excludes them"
        )
    signal, noise = names
    log_ps = np.asarray(densities[signal].log_prob(stats, mchirp), dtype=np.float64)
    log_pn = np.asarray(densities[noise].log_prob(stats, mchirp), dtype=np.float64)
    if not np.isfinite(log_ps).all() or not np.isfinite(log_pn).all():
        raise ValueError(
            "a component density is zero at one of the observed triggers, so the "
            "mixture likelihood is minus infinity there and no rate can be inferred; "
            "widen the support or the bandwidth"
        )

    total_grid, fraction_grid = bracket_grid(stats, densities, n_grid)
    n = float(stats.size)
    # sum_i log[a p_s + (1-a) p_n], over the fraction axis.
    log_a = np.log(fraction_grid)[None, :]
    log_1ma = np.log1p(-fraction_grid)[None, :]
    mixture = np.logaddexp(
        log_a + log_ps[:, None], log_1ma + log_pn[:, None]
    ).sum(axis=0)
    count_term = -total_grid + n * np.log(total_grid)
    log_posterior = (
        count_term[:, None]
        + mixture[None, :]
        + log_prior(total_grid[:, None], fraction_grid[None, :], kind=prior)
    )
    return RatePosterior(
        categories=(signal, noise),
        total_grid=total_grid,
        fraction_grid=fraction_grid,
        log_posterior=log_posterior,
        n_triggers=int(stats.size),
        clustered=True,
        prior=str(prior),
    )


def bracket_grid(
    stats: np.ndarray, densities: Dict[str, object], n_grid: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose grid ranges wide enough to contain the posterior, from the data.

    The total count is Poisson with mean ``lam``, so the posterior on ``lam`` is within a
    few ``sqrt(N)`` of ``N``; ten standard deviations either side, plus a constant floor
    for the small-``N`` case where the square root is not yet a useful width. Derived
    rather than configured, so the answer cannot depend on a hand-chosen range -- which is
    what ``tests/test_search_pastro.py::test_grid_range_does_not_affect_the_result``
    exists to hold.

    The fraction spans the whole unit interval, laid out by :func:`_fraction_nodes`.
    Bracketing it from the data would be circular: how much of the observed set is signal
    is the question being asked.
    """
    stats = np.asarray(stats, dtype=np.float64).ravel()
    n = float(stats.size)
    width = 10.0 * np.sqrt(n + 1.0) + 10.0
    lo = max(1e-6, n - width)
    total = np.linspace(lo, n + width, int(n_grid))
    return total, _fraction_nodes(int(n_grid))


def log_prior(
    total: np.ndarray, fraction: np.ndarray, kind: str = "jeffreys"
) -> np.ndarray:
    """
    Log prior in the total-and-fraction parameterisation, including its Jacobian.

    The Poisson Jeffreys prior of Eq. (17) of ``arxiv_1302.5341`` is
    ``1 / sqrt(Ls * Ln)`` on the two rates. Changing variables to ``Ls = a * lam`` and
    ``Ln = (1 - a) * lam`` carries a Jacobian of ``lam``, and::

        lam / sqrt(a * lam * (1 - a) * lam)  =  1 / sqrt(a * (1 - a))

    so the total rate cancels exactly and the transformed prior is flat in ``lam`` and
    Beta(1/2, 1/2) in the fraction. ``total`` is accepted so the signature matches the
    parameterisation and a caller cannot forget which variables the prior belongs to; it
    is broadcast against and otherwise unused, which is the correct answer rather than an
    omission.

    The sgwc-1 notebook defines this prior three times and the first definition, cell 36,
    omits the Jacobian: it evaluates ``1 / sqrt(Ls Ln)`` at the reparameterised point and
    integrates it on a uniform grid in ``(lam, a)``, which is a density with respect to
    ``dLs dLn`` summed against ``dlam da`` and so carries a spurious ``1 / lam`` -- small
    where the posterior is narrow, a genuine tilt where it is not. Cells 66 and 77
    redefine it correctly, and cell 67's grid is the pass that reaches the reported
    p_astro, so the published numbers are not affected. Applied analytically here so that
    there is one definition and no dependence on the order cells happened to run in.
    """
    total = np.asarray(total, dtype=np.float64)
    fraction = np.asarray(fraction, dtype=np.float64)
    if kind != "jeffreys":
        raise ValueError(
            f"unknown prior {kind!r}; only 'jeffreys' is implemented, and it is the one "
            "the references derive their closed forms under"
        )
    if np.any(total <= 0):
        raise ValueError("the total rate must be positive")
    if np.any((fraction <= 0) | (fraction >= 1)):
        raise ValueError(
            "the signal fraction must lie strictly inside (0, 1); the prior is infinite "
            "at both ends and the data can support neither"
        )
    return np.broadcast_arrays(
        np.zeros_like(total), -0.5 * np.log(fraction) - 0.5 * np.log1p(-fraction)
    )[1] + np.zeros_like(total)

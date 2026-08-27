#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_significance.py
Description   : Figure data for search significance and background validity.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from sage.search.figdata.product import FigData


def cumulative_vs_ifar(spec) -> FigData:
    """
    Candidate counts against inverse false-alarm rate, with the expected band.

    ``pycbc_page_ifar``, and the figure a search result is read off: the observed
    cumulative count against the background expectation ``T / IFAR``, with Poisson bands.
    A campaign with nothing astrophysical in it tracks the expectation; a real candidate
    is a point that leaves the band at high IFAR.

    Bands are Poisson *quantiles*, not ``expected +/- sigma * sqrt(expected)``. In the
    tail the expectation is well below one, where the normal approximation gives a
    negative lower edge and a band that does not contain the integers a count can
    actually take -- which is exactly the region a detection claim is made in.

    The counting is done by :func:`sage.search.far.cumulative_vs_ifar`, not repeated
    here, so this figure and any table quoting the same numbers cannot disagree.
    """
    from sage.search.candidates import CandidateTable
    from sage.search.far import FarCurve, cumulative_vs_ifar as _counts

    curve = FarCurve.load(
        spec.path("far", f"far_curve_{spec.data.observing_run}_inclusive.h5")
    )
    table = CandidateTable.load(
        spec.path("candidates", "candidates.h5"), allow_undetermined=True
    )
    counts = _counts(np.asarray(table.columns["stat"], dtype=np.float64), curve)

    arrays = {
        "ifar_yr": counts["ifar_yr"],
        "n_cumulative": counts["observed"],
        "expected": counts["expected"],
    }
    for sigma in (1, 2, 3):
        lower, upper = counts[f"band_{sigma}"]
        arrays[f"band_{sigma}sigma_lo"] = np.asarray(lower, dtype=np.float64)
        arrays[f"band_{sigma}sigma_hi"] = np.asarray(upper, dtype=np.float64)
    return FigData(
        figure="cumulative_vs_ifar",
        arrays=arrays,
        scalars={
            "background_livetime_s": float(curve.background_livetime_s),
            "foreground_livetime_s": float(curve.foreground_livetime_s),
            "n_candidates": int(len(table)),
            "ifar_cap_yr": float(curve.ifar_cap_yr),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def statistic_distributions(spec) -> FigData:
    """
    Foreground and background ranking-statistic distributions.

    Both histograms come from the stages' own stored histograms rather than being
    recomputed from the trigger lists. The stored ones counted *every* analysed window,
    including those below the keep threshold that were never written as rows, so a
    histogram rebuilt from the shard would be missing the bulk of the distribution and
    would show a foreground that starts where the threshold does.

    The underflow and overflow counts travel with the bins. A raw logit is unbounded, so
    the outermost bins are not the ends of the distribution, and a figure drawn without
    them silently omits the loudest events -- which are the only ones anyone looks at.
    """
    from sage.search.background import BackgroundSet
    from sage.search.triggers import hist_edges, read_shard

    _, foreground = read_shard(spec.path("zerolag", "zerolag_slide0000.h5"))
    background = BackgroundSet.load(spec.path("background", "bg_inclusive.h5"))
    if foreground is None or background.histogram is None:
        raise ValueError(
            "both the zero-lag shard and the background must carry histograms; without "
            "them the distributions could only be rebuilt from the kept triggers, which "
            "start at the keep threshold"
        )
    # The fixed campaign-wide grid, not one derived per shard: histograms from separate
    # jobs are added, so edges chosen per shard would bin the same statistic differently
    # in different shards and the sum would be meaningless while still looking like one.
    edges = np.asarray(hist_edges(), dtype=np.float64)
    return FigData(
        figure="statistic_distributions",
        arrays={
            "stat_edges": edges,
            "counts_zerolag": np.asarray(foreground.counts, dtype=np.int64),
            "counts_background": np.asarray(
                background.histogram.counts, dtype=np.int64
            ),
        },
        scalars={
            "counts_background_underflow": int(background.histogram.underflow),
            "counts_background_overflow": int(background.histogram.overflow),
            "counts_zerolag_underflow": int(foreground.underflow),
            "counts_zerolag_overflow": int(foreground.overflow),
            "background_livetime_s": float(background.livetime_s),
            "foreground_livetime_s": float(
                background.foreground_livetime_s
                if background.foreground_livetime_s is not None
                else float("nan")
            ),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def statistic_ccdf(spec) -> FigData:
    """
    Log-scale complementary CDF of the zero-lag and background statistics.

    sgwc-1's ``plot_log_ccdf`` (``search.ipynb`` cells 229 and 324), which is the figure
    the two distributions are actually compared on: on a log CCDF an exponential tail is
    a straight line, so a background that departs from one is visible by eye.

    The empirical CCDF is ``1 - i/n`` over the sorted values, exactly as sgwc-1 computes
    it. That convention puts the largest value at zero rather than at ``1/n``, which on a
    log axis drops it off the plot -- so the loudest point is carried separately in
    ``loudest_stat`` rather than being left to vanish.
    """
    from sage.search.background import BackgroundSet
    from sage.search.triggers import read_shard

    background = BackgroundSet.load(spec.path("background", "bg_inclusive.h5"))
    zerolag, _ = read_shard(spec.path("zerolag", "zerolag_slide0000.h5"))

    series = {
        "background": np.asarray(background.stats, dtype=np.float64),
        "zerolag": np.asarray(zerolag.columns["stat"], dtype=np.float64),
    }
    labels, stats, ccdf = [], [], []
    for label, values in series.items():
        ordered = np.sort(values)
        labels.extend([label] * ordered.size)
        stats.append(ordered)
        ccdf.append(1.0 - np.arange(1, ordered.size + 1) / ordered.size)
    return FigData(
        figure="statistic_ccdf",
        arrays={
            "label": np.asarray(labels),
            "stat_sorted": np.concatenate(stats),
            "ccdf": np.concatenate(ccdf),
        },
        scalars={
            "n_background": int(series["background"].size),
            "n_zerolag": int(series["zerolag"].size),
            "loudest_stat": float(series["zerolag"].max()),
            "loudest_background_stat": float(series["background"].max()),
            "background_livetime_s": float(background.livetime_s),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def pastro_threshold_invariance(spec) -> FigData:
    """
    Each candidate's probability as the analysis threshold is moved.

    sgwc-1's "p_astro for different events vs Ranking Statistic Threshold"
    (``pastro.ipynb``). The mixture is refitted at every threshold, on both components
    re-truncated to the new support -- that re-truncation is the step the figure exists
    to exercise, so it is done rather than approximated by reweighting the old fit.

    p_astro should not depend on where the analysis was cut. It is a posterior from a
    rate mixture, and the rates are counts *above the threshold*; if the model is right,
    moving the threshold changes both the counts and the densities in compensating ways.
    A candidate whose probability drifts with the cut is the signature of a mixture that
    is not describing the data.

    The uncertainty is carried per point, because agreement has to be judged against the
    precision achieved rather than against a fixed allowance: an estimate that is stable
    to within a wide interval has not demonstrated much.
    """
    from sage.search.background import BackgroundSet, cluster_zerolag
    from sage.search.far import FarCurve
    from sage.search.injection.campaign import scored_stats
    from sage.search.pastro.assign import assign_pastro
    from sage.search.pastro.run import fit_at_threshold
    from sage.search.triggers import read_shard

    background = BackgroundSet.load(spec.path("background", "bg_inclusive.h5"))
    curve = FarCurve.load(
        spec.path("far", f"far_curve_{spec.data.observing_run}_inclusive.h5")
    )
    injection_stats = scored_stats(spec)
    zerolag, _ = read_shard(spec.path("zerolag", "zerolag_slide0000.h5"))
    clustered = cluster_zerolag(
        zerolag, window_s=float(spec.cluster.window_s), linkage=spec.cluster.linkage
    )
    stats = np.asarray(clustered.columns["stat"], dtype=np.float64)
    times = np.asarray(clustered.columns["gps"], dtype=np.float64)

    # A ladder around the configured threshold, in the units it is quoted in. Spread over
    # a decade either side: too narrow and every fit sees the same triggers, which would
    # show stability the analysis never demonstrated.
    nominal = float(spec.pastro.threshold_far_per_day)
    ladder = nominal * np.array([0.1, 0.3, 1.0, 3.0, 10.0])

    # The loudest candidates, which are the ones a claim rests on and the only ones
    # present in every fit -- a quieter one drops out as the threshold rises, and its
    # absence would read as a probability that fell to zero.
    order = np.argsort(-stats)[: min(5, stats.size)]
    tracked_stats, tracked_times = stats[order], times[order]

    thresholds, values, sigmas, names = [], [], [], []
    for far_per_day in ladder:
        try:
            # The same construction the pastro stage runs, called rather than repeated.
            # Two copies drifted apart once: this one kept asking for the generalised-
            # Pareto tail after the driver stopped, so the figure could have illustrated
            # an analysis nobody ran.
            support, densities, posterior, _ = fit_at_threshold(
                injection_stats,
                background.stats,
                stats,
                curve,
                threshold_far_per_day=float(far_per_day),
                n_rate_grid=int(spec.pastro.n_rate_grid),
            )
        except ValueError:
            # Outside what this background can resolve, or leaving no trigger inside the
            # support. Recorded by omission rather than by a fabricated point.
            continue
        # Only the tracked candidates this threshold's support actually covers. A
        # candidate below the cut was not in this fit, and both densities are zero there
        # -- assessing it anyway would ask for a ratio that does not exist, and plotting
        # the result would show a probability collapsing as the threshold rose when what
        # really happened is that the candidate left the analysis.
        covered = (tracked_stats >= support.stat_lo) & (
            tracked_stats <= support.stat_hi
        )
        if not covered.any():
            continue
        table = assign_pastro(
            tracked_stats[covered],
            densities,
            posterior,
            gps=tracked_times[covered],
        )
        astro = table.astrophysical()
        bounds = _astrophysical_bounds(table)
        for index, stat in enumerate(tracked_stats[covered]):
            thresholds.append(float(support.threshold_stat))
            values.append(float(astro[index]))
            sigmas.append(0.5 * float(bounds[1][index] - bounds[0][index]))
            names.append(f"stat={stat:.4f}")

    return FigData(
        figure="pastro_threshold_invariance",
        arrays={
            "threshold_stat": np.asarray(thresholds, dtype=np.float64),
            "p_astro_by_event": np.asarray(values, dtype=np.float64),
            "p_astro_sigma": np.asarray(sigmas, dtype=np.float64),
            "event_name": np.asarray(names),
        },
        scalars={
            "n_thresholds": int(len(set(thresholds))),
            "n_tracked": int(tracked_stats.size),
            "nominal_far_per_day": nominal,
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def far_versus_statistic(spec) -> FigData:
    """
    The statistic-to-rate mapping.

    ``pycbc_page_fars_vs_stat``. The rate is counted, and above the loudest background
    event it holds flat at the counting floor ``(1 + 1) / T_b`` -- ``is_extrapolated``
    marks exactly where that begins, because a reader has to be able to tell a measured
    rate from the floor the counting saturates at. Nothing is fitted past the count.
    """
    from sage.search.far import FarCurve, poisson_band

    curve = FarCurve.load(
        spec.path("far", f"far_curve_{spec.data.observing_run}_inclusive.h5")
    )
    stat = np.asarray(curve.stat, dtype=np.float64)

    # The band is the counting uncertainty on the rate: n counted above a statistic is
    # Poisson, so the rate inherits that spread. Only where counting happened.
    counted = np.asarray(curve.n_louder, dtype=np.float64)
    lo, hi = poisson_band(np.clip(counted, 0.0, None), 1)
    scale = np.where(
        counted > 0, np.asarray(curve.far_per_yr) / np.maximum(counted, 1.0), np.nan
    )
    return FigData(
        figure="far_versus_statistic",
        arrays={
            "stat": stat,
            "far_per_yr": np.asarray(curve.far_per_yr, dtype=np.float64),
            # Called, not read. `is_extrapolated` is a method taking the statistics to
            # test; handing the bound method to `np.asarray` produced an object array
            # that no reader could have used and no test caught, because this builder
            # had never run.
            "is_extrapolated": np.asarray(curve.is_extrapolated(stat), dtype=bool),
            "n_louder": np.asarray(curve.n_louder, dtype=np.int64),
            "count_band_lo": np.asarray(lo, dtype=np.float64) * scale,
            "count_band_hi": np.asarray(hi, dtype=np.float64) * scale,
        },
        scalars={
            "background_livetime_s": float(curve.background_livetime_s),
            "foreground_livetime_s": float(curve.foreground_livetime_s),
            "n_background": int(stat.size),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def pastro_curves(spec) -> FigData:
    """
    Astrophysical probability against ranking statistic.

    sgwc-1's "p_astro vs Ranking Statistic" (``pastro.ipynb``). The credible band is
    carried with the curve because p_astro is an average over the rate posterior, not a
    point estimate, and a curve drawn without its band claims a precision the rates do
    not have.
    """
    from sage.search.pastro.assign import PAstroTable

    table = PAstroTable.load(spec.path("pastro", "pastro_table.h5"))
    stat = np.asarray(table.stat, dtype=np.float64)
    order = np.argsort(stat)
    astro = table.astrophysical()
    signal = _astrophysical_bounds(table)
    return FigData(
        figure="pastro_curves",
        arrays={
            "stat": stat[order],
            "p_astro": np.asarray(astro, dtype=np.float64)[order],
            "p_astro_lo": np.asarray(signal[0], dtype=np.float64)[order],
            "p_astro_hi": np.asarray(signal[1], dtype=np.float64)[order],
        },
        scalars={
            "threshold_stat": float(stat.min()),
            "n_triggers": int(stat.size),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def _astrophysical_bounds(table):
    """The astrophysical components' credible bounds, summed as the value is."""
    from sage.search.pastro.categories import DEFAULT_CATEGORIES

    astro = {c.name for c in DEFAULT_CATEGORIES if c.astrophysical}
    present = [name for name in table.probabilities if name in astro]
    return (
        np.sum([table.lower[name] for name in present], axis=0),
        np.sum([table.upper[name] for name in present], axis=0),
    )


def pastro_densities(spec) -> FigData:
    """
    The two mixture components against the ranking statistic.

    sgwc-1's "Histogram-smoothed likelihoods" (``pastro.ipynb`` cell 43). Both densities
    are evaluated on the *shared* support they were fitted on: their ratio is what
    p_astro is built from, and a ratio of two densities defined on different regions is a
    property of the truncation rather than of the data.

    The trigger histogram is drawn underneath so the fit can be read against what it was
    fitted to, which is the whole point of showing the densities at all.
    """
    from sage.search.pastro.io import load_model
    from sage.search.triggers import read_shard

    model = load_model(spec.path("pastro", "pastro_model.h5"))
    support = model["support"]
    densities = model["densities"]
    signal, noise = model["categories"]

    nodes = np.linspace(float(support.stat_lo), float(support.stat_hi), 1024)
    zerolag, _ = read_shard(spec.path("zerolag", "zerolag_slide0000.h5"))
    stats = np.asarray(zerolag.columns["stat"], dtype=np.float64)
    inside = stats[(stats >= support.stat_lo) & (stats <= support.stat_hi)]
    edges = np.linspace(float(support.stat_lo), float(support.stat_hi), 65)
    counts, _ = np.histogram(inside, bins=edges)

    return FigData(
        figure="pastro_densities",
        arrays={
            "stat": nodes,
            "p_signal": np.exp(densities[signal].log_prob(nodes)),
            "p_noise": np.exp(densities[noise].log_prob(nodes)),
            "trigger_hist_edges": edges,
            "trigger_hist_counts": counts.astype(np.int64),
        },
        scalars={
            "support_lo": float(support.stat_lo),
            "support_hi": float(support.stat_hi),
            "threshold_stat": float(getattr(support, "threshold_stat", np.nan)),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def pastro_rate_posterior_reparam(spec) -> FigData:
    """
    The rate posterior in the parameterisation it was computed in.

    sgwc-1's "Posterior over (lambda, r)" (``pastro.ipynb``). The reparameterised grid is
    the one that was actually evaluated -- direct gridding in the two rates underflows,
    which the notebook says at cell 33 -- so this is the posterior as it exists rather
    than a transform of it.

    The log posterior is stored, not the normalised weights. Weights depend on the
    quadrature the grids imply, and a figure carrying one particular quadrature would
    disagree with any consumer that applied another.
    """
    from sage.search.pastro.rates import RatePosterior

    posterior = RatePosterior.load(spec.path("pastro", "rate_posterior.h5"))
    flat = int(np.argmax(posterior.log_posterior))
    i, j = np.unravel_index(flat, posterior.log_posterior.shape)
    return FigData(
        figure="pastro_rate_posterior_reparam",
        arrays={
            "total_grid": posterior.total_grid,
            "fraction_grid": posterior.fraction_grid,
            "log_posterior": posterior.log_posterior,
        },
        scalars={
            "map_total": float(posterior.total_grid[i]),
            "map_fraction": float(posterior.fraction_grid[j]),
            "n_triggers": int(posterior.n_triggers),
            "prior": str(posterior.prior),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def pastro_rate_posterior(spec) -> FigData:
    """
    The rate posterior in the two component rates.

    sgwc-1's "Posterior over (lambda_s, lambda_n)" (``pastro.ipynb``). Carried as the
    rates at each grid cell rather than as a regular grid in them: the evaluation grid is
    rectangular in total-and-fraction, so its image in the two rates is a fan, and
    presenting it as a rectangle would place probability where none was computed.
    """
    from sage.search.pastro.rates import RatePosterior

    posterior = RatePosterior.load(spec.path("pastro", "rate_posterior.h5"))
    signal, noise = posterior.categories
    total = posterior.total_grid[:, None]
    fraction = posterior.fraction_grid[None, :]
    rates = {signal: total * fraction, noise: total * (1.0 - fraction)}
    flat = int(np.argmax(posterior.log_posterior))
    i, j = np.unravel_index(flat, posterior.log_posterior.shape)
    return FigData(
        figure="pastro_rate_posterior",
        arrays={
            "lambda_s_grid": np.broadcast_to(rates[signal], posterior.log_posterior.shape),
            "lambda_n_grid": np.broadcast_to(rates[noise], posterior.log_posterior.shape),
            "log_posterior": posterior.log_posterior,
        },
        scalars={
            "map_lambda_s": float(rates[signal][i, j]),
            "map_lambda_n": float(rates[noise][i, j]),
            "signal_category": str(signal),
            "noise_category": str(noise),
        },
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )


def foreground_rate_from_injections(spec) -> FigData:
    """Predicted foreground counts from injections, against what was observed."""
    raise NotImplementedError


def window_offset_stability(spec) -> FigData:
    """Score stability under analysis-window shifts, for a signal and a noise trigger."""
    raise NotImplementedError


def background_validity(spec) -> FigData:
    """Background over-dispersion and per-slide livetime retention."""
    raise NotImplementedError


def build(spec, figures: Optional[list] = None) -> Dict[str, Path]:
    """Build every significance figure data product."""
    raise NotImplementedError

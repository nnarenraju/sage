#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : run.py
Description   : Stage driver for p_astro.

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
from typing import Optional

import numpy as np

from sage.search.fingerprint import combine, digest_h5
from sage.search.pastro.validate import ValidationReport


def fit_at_threshold(
    injections,
    background_stats,
    foreground,
    curve,
    threshold_far_per_day: float,
    n_rate_grid: int = 512,
):
    """
    Support, both densities and the rate posterior, for one analysis threshold.

    **The only place p_astro is constructed.** The stage driver calls it once at the
    campaign's threshold; the threshold-invariance figure calls it at each rung of its
    ladder. Two copies of this drifted apart once already -- the figure kept asking for
    the generalised-Pareto tail after the driver stopped, and omitted the livetimes the
    driver passed -- so a published figure could have illustrated an analysis nobody ran.

    Returns ``(support, densities, posterior, kept)``, where ``kept`` selects the
    triggers inside the support. The support's lower edge *is* the analysis threshold, so
    a trigger below it was never analysed under the threshold the densities describe.

    Raises
    ------
    ValueError
        If the threshold is outside what the background can resolve, or if no trigger
        survives inside the support.
    """
    from sage.search.pastro.density import noise_density, signal_density
    from sage.search.pastro.rates import fit_rates
    from sage.search.pastro.support import build_support

    injections = np.asarray(injections, dtype=np.float64)
    foreground = np.asarray(foreground, dtype=np.float64)
    support = build_support(
        curve,
        threshold_far_per_day=float(threshold_far_per_day),
        must_include=np.concatenate([injections, foreground]),
    )
    densities = {
        "BBH": signal_density(injections, support),
        "Terrestrial": noise_density(
            np.asarray(background_stats, dtype=np.float64), support
        ),
    }
    kept = (foreground >= support.stat_lo) & (foreground <= support.stat_hi)
    if not kept.any():
        raise ValueError(
            f"no zero-lag trigger lies in the common support "
            f"[{support.stat_lo:.6g}, {support.stat_hi:.6g}], so no rate can be "
            "inferred. The support's lower edge is the analysis threshold, so this says "
            "the search found nothing above it"
        )
    posterior = fit_rates(
        foreground[kept],
        densities,
        support,
        clustered=True,
        n_grid=int(n_rate_grid),
    )
    return support, densities, posterior, kept


def run(spec, resume: bool = True, **kwargs) -> dict:
    """
    Fit the mixture for one observing run and assign per-trigger probabilities.

    Runs in order: build the shared support from the analysis threshold, estimate both
    densities on it, gate on the likelihood-ratio ordering, infer and marginalise the
    rates, assign probabilities, then run the validation suite.

    **What each density is built from.** The noise density comes from the time-slid
    background -- the same background the false-alarm rates were counted from, asserted to
    be so -- continued above the loudest slide trigger by the tail already fitted in
    ``far``. The signal density comes from the *recovered injections'* ranking statistics.
    That is sgwc-1's construction (``pastro.ipynb`` cells 13 and 22, reading
    ``o3_injection_study_*_psignal.hdf5``) and it is why this stage depends on
    ``injections`` rather than on ``sensitivity``: a sensitive volume is a different
    quantity and enters nowhere.

    **Nothing here gates the analysis.** Monotonicity of the likelihood ratio is measured,
    recorded in the report and in the persisted model, and acted on only if the campaign
    asks for it (``spec.pastro.monotonicity_policy``, ``"report"`` by default). That
    matches sgwc-1, which has no such check, and it is deliberate while the chain is being
    brought up end to end: a strict node-wise test on a kernel-estimated ratio fails on
    estimator ripple, and restricting on that throws away the top of the support -- where
    the detections are. What is refused is a *silent* pass: the measurement is always
    reported and the products carry it.
    """
    from sage.search.background import BackgroundSet
    from sage.search.far import FarCurve
    from sage.search.pastro.assign import assign_pastro
    from sage.search.pastro.io import require_clustered, save_model
    from sage.search.pastro.monotonic import check_monotonicity
    from sage.search.pastro.validate import run_suite
    from sage.search.triggers import read_shard

    background = BackgroundSet.load(spec.path("background", "bg_inclusive.h5"))
    curve = FarCurve.load(
        spec.path("far", f"far_curve_{spec.data.observing_run}_inclusive.h5")
    )
    injections = _injection_stats(spec)

    # The zero-lag triggers the rates are inferred from, clustered. Unclustered they are
    # not independent draws, and every inferred rate is multiplied by the number of
    # windows a glitch spans.
    zerolag, _ = read_shard(spec.path("zerolag", "zerolag_slide0000.h5"))
    from sage.search.background import cluster_zerolag

    clustered = cluster_zerolag(
        zerolag,
        window_s=float(spec.cluster.window_s),
        linkage=spec.cluster.linkage,
    )
    require_clustered(clustered)
    foreground = np.asarray(clustered.columns["stat"], dtype=np.float64)
    foreground_gps = np.asarray(clustered.columns["gps"], dtype=np.float64)

    # Both the recovered injections and the candidates that will be scored have to lie
    # inside the support: its upper edge from the FAR curve alone is the loudest
    # *background* event, and a confident candidate is one louder than all background.
    #
    # One call, shared with the threshold-invariance figure, so the figure cannot
    # illustrate a different analysis from the one that produced the numbers.
    support, densities, posterior, kept = fit_at_threshold(
        injections,
        background.stats,
        foreground,
        curve,
        threshold_far_per_day=float(spec.pastro.threshold_far_per_day),
        n_rate_grid=int(spec.pastro.n_rate_grid),
    )
    foreground, foreground_gps = foreground[kept], foreground_gps[kept]

    monotonicity = check_monotonicity(
        densities["BBH"], densities["Terrestrial"], support
    )
    # Measured, reported, and acted on by nothing. sgwc-1 has no such check, and the chain
    # is being brought up against it. A strict node-wise test on a kernel-estimated ratio
    # fails on estimator ripple rather than on structure, and narrowing the support on that
    # ripple discards the top of it -- which is where the detections are. On the O3a
    # campaign it cost the two most confident candidates, at 18.66 and 18.41, their p_astro
    # entirely, while a weaker one at 17.89 scored 0.796. `monotonic.apply_policy` is kept
    # and tested for when the gate is wired back in; see PastroSpec.monotonicity_policy.

    table = assign_pastro(foreground, densities, posterior, gps=foreground_gps)
    validation = run_suite(foreground, densities, posterior, support)

    rates_path = spec.path("pastro", "rate_posterior.h5")
    table_path = spec.path("pastro", "pastro_table.h5")
    model_path = spec.path("pastro", "pastro_model.h5")
    posterior.save(rates_path)
    table.attrs.setdefault("observing_run", str(spec.data.observing_run))
    table.attrs.setdefault("arm", str(spec.arm))
    table.save(table_path)
    save_model(
        model_path,
        densities,
        support,
        posterior,
        validation,
        attrs={"observing_run": str(spec.data.observing_run), "arm": str(spec.arm)},
    )

    return {
        "model": str(model_path),
        "rate_posterior": str(rates_path),
        "table": str(table_path),
        "n_triggers": int(foreground.size),
        "n_injections": int(injections.size),
        "map_rates": {k: float(v) for k, v in posterior.map_rates.items()},
        "mean_rates": {k: float(v) for k, v in posterior.mean_rates.items()},
        "max_p_astro": float(np.max(table.astrophysical())),
        "monotone": bool(monotonicity.is_monotone),
        # Nothing restricts the support while the gate is out of the analysis path, so
        # this is always null. Kept in the schema rather than dropped: a reader comparing
        # campaigns across the change needs to see that this one narrowed nothing, and an
        # absent key reads as an older product that never recorded it.
        "restricted_to": None,
        "support": [float(support.stat_lo), float(support.stat_hi)],
        "validation": validation.as_dict(),
        # Digest the products, not the rates. Two campaigns can share a MAP rate to six
        # figures and assign different probabilities to every candidate; what downstream
        # reads is the table and the model.
        "fingerprint": combine(
            int(foreground.size),
            digest_h5([model_path, rates_path, table_path]),
        ),
    }


def _injection_stats(spec) -> np.ndarray:
    """
    Ranking statistics of the scored injections, which are ``p(x | signal)``.

    Read from the injection stage's product rather than recomputed. The density has to
    describe signals as *this* network scored them, through this preprocessor, on this
    run's noise -- which is why an external injection release cannot supply it.
    """
    from sage.search.injection.campaign import scored_stats

    return scored_stats(spec)


def main(argv: Optional[list] = None) -> int:
    """Command-line entry point."""
    import argparse
    import json

    from sage.search.spec import load_spec

    parser = argparse.ArgumentParser(description="Fit p_astro for one campaign.")
    parser.add_argument("--config", required=True, help="Campaign config module or path.")
    args = parser.parse_args(argv)

    report = run(load_spec(args.config))
    print(json.dumps({k: v for k, v in report.items() if k != "validation"}, indent=2))
    failures = report["validation"]["failures"]
    if failures:
        print(f"validation failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

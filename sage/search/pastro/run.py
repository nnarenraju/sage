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

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np

from sage.search.fingerprint import combine, digest_h5
from sage.search.pastro.validate import ValidationReport


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

    **A failed gate does not stop the stage.** The monotonicity gate decides whether the
    likelihood ratio orders the statistic, which is the condition under which FGMC is
    interpretable at all -- and the policy for a failure is configuration
    (``spec.pastro.monotonicity_policy``), not a decision taken here. Every check is
    recorded in the report and in the persisted model whatever the outcome, so a campaign
    that produced probabilities under a failed check says so rather than looking like one
    that never checked. What is refused is a *silent* pass: ``validation.passed`` is
    reported and the products carry it.
    """
    from sage.search.background import BackgroundSet
    from sage.search.far import FarCurve
    from sage.search.pastro.assign import assign_pastro
    from sage.search.pastro.density import noise_density, signal_density
    from sage.search.pastro.io import require_clustered, save_model
    from sage.search.pastro.monotonic import apply_policy, check_monotonicity
    from sage.search.pastro.rates import fit_rates
    from sage.search.pastro.support import build_support
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
    support = build_support(
        curve,
        threshold_far_per_day=float(spec.pastro.threshold_far_per_day),
        must_include=np.concatenate([injections, foreground]),
    )

    densities = {
        "BBH": signal_density(injections, support),
        "Terrestrial": noise_density(
            np.asarray(background.stats, dtype=np.float64),
            support,
            tail=getattr(curve, "tail", None),
            background_livetime_s=float(background.livetime_s),
            foreground_livetime_s=float(curve.foreground_livetime_s),
            far_curve=curve,
        ),
    }

    monotonicity = check_monotonicity(
        densities["BBH"], densities["Terrestrial"], support
    )
    # apply_policy returns the interval to narrow to, or None when the gate passed. Under
    # "restrict" the densities are rebuilt on the narrowed support rather than truncated
    # after the fact: each is normalised over the region it is defined on, so a density
    # fitted on the wide support and evaluated on a narrow one integrates to less than
    # one, and the ratio p_astro is built from would then be a property of the truncation.
    region = apply_policy(monotonicity, policy=str(spec.pastro.monotonicity_policy))
    if region is not None:
        support = dataclasses.replace(
            support, stat_lo=float(region[0]), stat_hi=float(region[1])
        )
        densities = {
            "BBH": signal_density(injections, support),
            "Terrestrial": noise_density(
                np.asarray(background.stats, dtype=np.float64),
                support,
                tail=getattr(curve, "tail", None),
                background_livetime_s=float(background.livetime_s),
                foreground_livetime_s=float(curve.foreground_livetime_s),
                far_curve=curve,
            ),
        }
        # The rates are inferred from the triggers inside the region, since that is what
        # the densities now describe. Keeping the ones outside would ask the mixture to
        # account for triggers neither component has a density for.
        keep = (foreground >= support.stat_lo) & (foreground <= support.stat_hi)
        foreground, foreground_gps = foreground[keep], foreground_gps[keep]
        if foreground.size == 0:
            raise ValueError(
                "restricting to the monotone region "
                f"[{support.stat_lo:.6g}, {support.stat_hi:.6g}] leaves no zero-lag "
                "triggers, so no rate can be inferred. The gate is telling you the "
                "densities do not order this statistic anywhere the data lives"
            )

    posterior = fit_rates(
        foreground,
        densities,
        support,
        clustered=True,
        n_grid=int(spec.pastro.n_rate_grid),
    )
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
        "restricted_to": (
            None if region is None else [float(region[0]), float(region[1])]
        ),
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
    from sage.search.triggers import read_shard

    shard = spec.path("injections", "injection_triggers.h5")
    if not Path(shard).is_file():
        raise FileNotFoundError(
            f"no scored injections at {shard}; p(x | signal) is the distribution of the "
            "ranking statistic over recovered injections, so the injections stage must "
            "run before p_astro. It is not a sensitive volume and cannot be substituted "
            "by one"
        )
    table, _ = read_shard(shard)
    return np.asarray(table.columns["stat"], dtype=np.float64)


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

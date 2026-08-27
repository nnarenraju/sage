#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : spec.py
Description   : Declarations of the figure set.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Declaring each figure's required arrays here lets a missing input be caught when the
data product is built rather than when the figure is drawn, and documents in one place
what the paper contains.

The declarations are written before the builders, and deliberately so. ``requires`` is a
*backward* contract on every stage that writes a product: if a figure needs the per-slide
livetime array, the background stage has to persist it. Discovering that at the point of
drawing means re-running the stage that should have written it, and for the background
that is days of GPU time.

Each entry names the stages it draws on, so :func:`required_stages` can say what must have
run before a figure can be built, and so a stage can check on exit that everything
declared against it is present.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FigureDecl:
    """One figure: what it shows, what it needs, and where its numbers come from."""

    key: str
    title: str
    builder: str
    requires: Tuple[str, ...]
    sources: Tuple[str, ...]
    per_event: bool = False
    optional: bool = False
    # Non-empty when the capability behind this figure is deliberately not built yet, with
    # the reason. Distinct from `optional`, which means "produce it when the inputs happen
    # to exist": a deferred figure has no producer at all, and the per-layer exit
    # assertion -- for every figure whose `sources` names a stage in this layer, every name
    # in `requires` is actually written -- must skip it rather than fail on it. The
    # declaration stays as the design record so the arrays are not re-derived later.
    deferred: str = ""
    #: Where this figure comes from: ``"sgwc-1"``, ``"pycbc"``, or empty for one with no
    #: counterpart in either. The search follows sgwc-1's procedure and takes from PyCBC
    #: only what is agreed case by case, so a figure with no origin is one nobody has
    #: justified yet -- and every such figure is `deferred`, by the ruling of 2026-08-20.
    #: Recorded on the declaration rather than in a document so the two cannot drift.
    origin: str = ""
    #: Function inside ``builder`` that produces this figure, when it is not named after
    #: the key. Most builders name the function after the figure, so this is usually
    #: empty and :attr:`builder_function` falls back to the key. It exists because the
    #: mapping has to be *declared* somewhere: ``build()`` dispatches by it, and inferring
    #: it from the module's contents would pick a plausible neighbour when a name drifts.
    function: str = ""
    note: str = ""

    @property
    def builder_function(self) -> str:
        """The function name to call inside :attr:`builder`."""
        return self.function or self.key


FIGURES: Dict[str, FigureDecl] = {}


def declare(decl: FigureDecl) -> None:
    """Register a figure, refusing a duplicate key."""
    if decl.key in FIGURES:
        raise ValueError(f"figure {decl.key!r} is already declared")
    if not decl.requires:
        raise ValueError(f"figure {decl.key!r} declares no required arrays")
    if not decl.sources:
        raise ValueError(f"figure {decl.key!r} names no source stage")
    FIGURES[decl.key] = decl


def resolve(key: str) -> FigureDecl:
    """Look up a figure, raising with suggestions on a typo."""
    import difflib

    try:
        return FIGURES[key]
    except KeyError:
        close = difflib.get_close_matches(key, FIGURES, n=3, cutoff=0.4)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown figure {key!r}{hint}") from None


def pending(stage: str) -> Tuple[FigureDecl, ...]:
    """
    Figures a stage must actually produce arrays for.

    Deferred declarations are excluded. They are a design record of a capability that is
    not built, so counting them would make the per-layer exit assertion fail on work
    nobody has started -- and removing them instead would lose the array list and have it
    re-derived when the capability lands.
    """
    return tuple(
        decl
        for decl in FIGURES.values()
        if stage in decl.sources and not decl.deferred
    )


def deferred() -> Tuple[FigureDecl, ...]:
    """Every declaration whose producer is deliberately not built, with its reason."""
    return tuple(decl for decl in FIGURES.values() if decl.deferred)


def by_builder(builder: str) -> Tuple[FigureDecl, ...]:
    """Figures produced by one builder module."""
    return tuple(d for d in FIGURES.values() if d.builder == builder)


def by_stage(stage: str) -> Tuple[FigureDecl, ...]:
    """
    Figures drawing on one stage.

    Used at a stage's exit to check that every array declared against it was written,
    which is what makes ``requires`` a contract rather than documentation.
    """
    return tuple(d for d in FIGURES.values() if stage in d.sources)


def required_stages(keys: Sequence[str]) -> Tuple[str, ...]:
    """Analysis stages that must have run before these figures can be built."""
    needed: list = []
    for key in keys:
        for stage in resolve(key).sources:
            if stage not in needed:
                needed.append(stage)
    return tuple(needed)


def per_event_figures() -> Tuple[FigureDecl, ...]:
    """Figures built once per candidate rather than once per campaign."""
    return tuple(d for d in FIGURES.values() if d.per_event)


# --------------------------------------------------------------------- significance
declare(
    FigureDecl(
        key="cumulative_vs_ifar",
        origin="pycbc: pycbc_page_ifar",
        title="Cumulative candidates versus inverse false-alarm rate",
        builder="build_significance",
        requires=(
            "ifar_yr",
            "n_cumulative",
            "expected",
            "band_1sigma_lo",
            "band_1sigma_hi",
            "band_2sigma_lo",
            "band_2sigma_hi",
            "band_3sigma_lo",
            "band_3sigma_hi",
            "background_livetime_s",
            "foreground_livetime_s",
        ),
        sources=("far", "candidates"),
        note=(
            "The headline significance figure. Poisson bands are about the expected "
            "curve T/IFAR, and the extrapolated region is hatched from the measured "
            "background onward rather than from the loudest event."
        ),
    )
)
declare(
    FigureDecl(
        key="statistic_distributions",
        origin="sgwc-1: search.ipynb, trigger count vs ranking statistic",
        title="Ranking statistic: zero lag against time-slide background",
        builder="build_significance",
        requires=(
            "stat_edges",
            "counts_zerolag",
            "counts_background",
            "counts_background_underflow",
            "counts_background_overflow",
            "background_livetime_s",
            "foreground_livetime_s",
        ),
        sources=("zerolag", "background"),
        note="Overflow counts are carried so the loudest windows are visible, not clipped.",
    )
)
declare(
    FigureDecl(
        key="far_versus_statistic",
        origin="pycbc: pycbc_page_fars_vs_stat",
        title="False-alarm rate as a function of ranking statistic",
        builder="build_significance",
        requires=(
            "stat",
            "far_per_yr",
            "is_extrapolated",
            "background_livetime_s",
        ),
        sources=("far",),
        note=(
            "The counted rate, and only the counted rate. Above the loudest background "
            "event it holds flat at (1 + 1) / T_b and `is_extrapolated` marks where, so "
            "the floor is never read as a measurement. Nothing is fitted past the count."
        ),
    )
)
declare(
    FigureDecl(
        key="background_validity",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Background over-dispersion and per-slide livetime",
        builder="build_significance",
        requires=(
            "overdispersion_statistic",
            "overdispersion_p_value",
            "overdispersion_alpha",
            "index_of_dispersion",
            "n_slides",
            "livetime_per_slide_s",
        ),
        sources=("slides", "background"),
        note=(
            "The Poisson versus negative-binomial likelihood ratio on binned background "
            "counts, as sgwc-1 does it, plus the per-slide livetime array, which is also "
            "what proves T_b was measured rather than derived. The cumulative-count "
            "check against the expected background is 'cumulative_vs_ifar', which is the "
            "standard plot; there is deliberately no leave-one-slide-out figure here, "
            "since no published pipeline produces one."
        ),
    )
)
declare(
    FigureDecl(
        key="window_offset_stability",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Candidate ranking under window-offset perturbation",
        builder="build_significance",
        requires=("offset_s", "stat_by_offset", "candidate_gps", "candidate_name"),
        sources=("zerolag", "candidates"),
        note="A real signal should not depend strongly on where the window starts.",
    )
)
declare(
    FigureDecl(
        key="trials_comparison",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Significance with and without the trials factor",
        builder="build_significance",
        requires=(
            "candidate_name",
            "candidate_gps",
            "ifar_yr",
            "ifar_trials_yr",
            "n_trials",
            "covered_by",
            "tier",
            "tier_trials",
        ),
        sources=("trials", "candidates"),
        note=(
            "Both views side by side. Candidates crossing an inclusion threshold in one "
            "view and not the other are the ones worth showing."
        ),
    )
)
declare(
    FigureDecl(
        key="pastro_curves",
        origin="sgwc-1: pastro.ipynb, p_astro vs ranking statistic",
        title="Astrophysical probability against ranking statistic",
        builder="build_significance",
        requires=("stat", "p_astro", "p_astro_lo", "p_astro_hi", "threshold_stat"),
        sources=("pastro",),
    )
)

# ---------------------------------------------------------------------- sensitivity
declare(
    FigureDecl(
        key="vt_versus_far",
        origin="pycbc: pycbc_page_sensitivity",
        title="Sensitive volume-time against false-alarm threshold",
        builder="build_sensitivity",
        requires=(
            "far_threshold_per_yr",
            "vt",
            "vt_err",
            "n_effective",
            "n_found",
            "analysis_time_s",
        ),
        sources=("sensitivity",),
    )
)
declare(
    FigureDecl(
        key="vt_versus_parameter",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Sensitive volume-time against source parameters",
        builder="build_sensitivity",
        requires=("parameter", "bin_edges", "vt", "vt_err", "n_effective"),
        sources=("sensitivity",),
    )
)
declare(
    FigureDecl(
        key="sensitive_distance",
        origin="pycbc: pycbc_page_sensitivity",
        title="Sensitive distance across the mass plane",
        builder="build_sensitivity",
        requires=(
            "mass1",
            "mass2",
            "distance_mpc",
            "distance_err_mpc",
            "in_training_range",
        ),
        sources=("sensitivity",),
        note=(
            "Reference masses outside the training prior are marked, not quietly "
            "plotted as sensitivity."
        ),
    )
)
declare(
    FigureDecl(
        key="pipeline_comparison",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Sensitive volume-time against the reference pipelines",
        builder="build_sensitivity",
        requires=(
            "pipeline",
            "vt",
            "vt_err",
            "far_threshold_per_yr",
            "coincidence_restricted",
        ),
        sources=("sensitivity",),
        note=(
            "Restricted to each pipeline's coincident-time found injections, or labelled "
            "as unrestricted; the networks analysed differ and the comparison is "
            "otherwise not like for like."
        ),
    )
)
declare(
    FigureDecl(
        key="injection_recovery",
        origin="pycbc: pycbc_page_foundmissed",
        title="Found and missed injections in distance and chirp mass",
        builder="build_sensitivity",
        requires=(
            "injected_distance_mpc",
            "injected_mchirp",
            "found",
            "far_per_yr",
            "inside_analysed_segments",
        ),
        sources=("injections", "sensitivity"),
        note=(
            "Injections outside analysed time are shown as missed, which is what the "
            "wall-time convention requires."
        ),
    )
)
declare(
    FigureDecl(
        key="range_over_time",
        origin="pycbc: pycbc_plot_range",
        title="Detector range through the observing run",
        builder="build_sensitivity",
        requires=("gps", "range_mpc", "detector", "coincident_intervals"),
        sources=("segments", "sensitivity"),
    )
)
declare(
    FigureDecl(
        key="pastro_reliability",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Predicted against realised astrophysical fraction",
        builder="build_sensitivity",
        requires=("p_astro_bin_edges", "predicted_fraction", "realised_fraction", "count"),
        sources=("pastro", "injections"),
        note="A calibration check on p_astro itself, using the injection set as truth.",
    )
)

# ------------------------------------------------------------------------ catalogue
declare(
    FigureDecl(
        key="recovery_of_known_events",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Recovery of published events in the searched run",
        builder="build_catalogue",
        requires=(
            "event_name",
            "gps",
            "published_ifar_yr",
            "sage_ifar_yr",
            "sage_p_astro",
            "recovered",
            "inside_analysed_time",
        ),
        sources=("catalogue", "candidates"),
        note=(
            "The primary evidence that the pipeline works. Events outside analysed time "
            "are distinguished from events analysed and missed."
        ),
    )
)
declare(
    FigureDecl(
        key="comparison_matrix",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Event presence across Sage and the published catalogues",
        builder="build_catalogue",
        requires=(
            "event_name",
            "catalogue",
            "present",
            "significance",
            "significance_kind",
            "comparable",
        ),
        sources=("catalogue",),
        note=(
            "Significances that are not comparable between conventions are marked as "
            "such rather than placed on a shared axis."
        ),
    )
)
declare(
    FigureDecl(
        key="overlap_sets",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Overlap between Sage and the external catalogues",
        builder="build_catalogue",
        requires=("catalogue", "set_size", "intersection_labels", "intersection_size"),
        sources=("catalogue",),
    )
)
declare(
    FigureDecl(
        key="significance_agreement",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Significance agreement where conventions permit comparison",
        builder="build_catalogue",
        requires=("event_name", "sage_ifar_yr", "other_ifar_yr", "catalogue", "comparable"),
        sources=("catalogue",),
    )
)

# ----------------------------------------------------------------------- population
declare(
    FigureDecl(
        key="mass_plane",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Recovered chirp mass against significance",
        builder="build_population",
        requires=(
            "mchirp",
            "mchirp_sigma",
            "p_astro",
            "ifar_yr",
            "tier",
            "training_prior_bounds",
        ),
        sources=("candidates", "pastro"),
    )
)
declare(
    FigureDecl(
        key="population_shift",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Recovered against injected population",
        builder="build_population",
        requires=(
            "injected_mchirp",
            "recovered_mchirp",
            "weights",
            "ks_statistic",
            "ks_p_value",
        ),
        sources=("injections", "sensitivity"),
    )
)

# ----------------------------------------------------------------------------- meta
declare(
    FigureDecl(
        key="training_prior_and_coverage",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Training prior and where the candidates fall in it",
        builder="build_meta",
        requires=("prior_mass1", "prior_mass2", "candidate_mchirp", "id_fraction"),
        sources=("candidates",),
        note=(
            "Sage estimates only coalescence time and chirp mass, so this shows the "
            "prior against the one recovered parameter, not an inferred mass plane."
        ),
    )
)
declare(
    FigureDecl(
        key="livetime_and_duty_cycle",
        origin="pycbc: pycbc_page_segplot",
        title="Analysed livetime and where the rest of the run went",
        builder="build_meta",
        requires=(
            "arm",
            "observing_s",
            "coincident_s",
            "analysed_s",
            "lost_boundary_s",
            "lost_phase_restart_s",
            "lost_gaps_s",
            "duty_cycle",
        ),
        sources=("segments", "grid", "slides"),
        note=(
            "The coverage decomposition, per arm. Every rate the search reports is "
            "divided by the analysed time, so it is shown rather than asserted."
        ),
    )
)

# ------------------------------------------- background provenance and cross-checks
# Carried over from the SGWC-1 analysis notebooks, where each of these caught something.
declare(
    FigureDecl(
        key="background_by_network",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Background distribution for each detector network",
        builder="build_significance",
        requires=("arm", "stat_edges", "counts_clustered", "livetime_s"),
        sources=("background", "trials"),
        note=(
            "One curve per arm on shared bins. The arms have different livetimes, so the "
            "counts are shown with the livetime that normalises them."
        ),
    )
)
declare(
    FigureDecl(
        key="background_provenance",
        origin="",
        title="Background from different noise realisations",
        builder="build_significance",
        requires=("realisation", "stat_edges", "counts", "n_windows"),
        sources=("background",),
        deferred=(
            "Deferred with expected_vs_measured_background: the recoloured and "
            "coloured-Gaussian realisations this overlays are what the simulated "
            "expected background would produce."
        ),
        note=(
            "Real strain against recoloured and coloured-Gaussian variants. Divergence "
            "between them is what says the background is driven by non-Gaussian features "
            "rather than by the coloured noise floor."
        ),
    )
)
declare(
    FigureDecl(
        key="expected_vs_measured_background",
        origin="",
        title="Measured background against a simulated expectation",
        builder="build_significance",
        requires=(
            "stat_edges",
            "counts_measured",
            "counts_expected",
            "ad_statistic",
            "ad_p_value",
        ),
        sources=("background",),
        deferred=(
            "The simulated expected background is deferred by user ruling 2026-08-18. "
            "Nothing produces counts_expected, and no two-sample Anderson-Darling exists "
            "-- tail.anderson_darling is a one-sample goodness-of-fit against a fitted "
            "generalised Pareto and cannot compare two empirical samples."
        ),
        note=(
            "The two-sample statistic is reported on the figure. Agreement says the "
            "background is Gaussian-dominated; disagreement locates where it is not."
        ),
    )
)
declare(
    FigureDecl(
        key="statistic_ccdf",
        origin="sgwc-1: search.ipynb cell 229/324, plot_log_ccdf",
        title="Complementary cumulative distribution of the ranking statistic",
        builder="build_significance",
        requires=("label", "stat_sorted", "ccdf"),
        sources=("zerolag", "background"),
        note=(
            "A log-scale survival curve shows the tail behaviour a histogram hides, and "
            "is where a heavier-than-exponential tail first becomes visible."
        ),
    )
)
declare(
    FigureDecl(
        key="loudest_background_events",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="The loudest background events and where they came from",
        builder="build_significance",
        requires=("rank", "stat", "gps", "slide_id", "detector_peak", "segment_index"),
        sources=("background",),
        note=(
            "Carries provenance per event so a loud background trigger can be traced to "
            "its slide and segment, which is how a glitch family gets identified."
        ),
    )
)
declare(
    FigureDecl(
        key="known_event_ranking",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Published events ordered by Sage ranking statistic",
        builder="build_significance",
        requires=(
            "event_name",
            "stat",
            "network_snr",
            "far_threshold_stats",
            "far_threshold_labels",
            "is_ood",
            "recovered",
        ),
        sources=("catalogue", "candidates", "far"),
        note=(
            "Marker size scales with the published network signal-to-noise ratio, and "
            "the false-alarm thresholds are drawn as horizontal lines, so which known "
            "events clear which threshold is read directly off the figure."
        ),
    )
)

# ----------------------------------------------------- astrophysical probability set
declare(
    FigureDecl(
        key="pastro_densities",
        origin="sgwc-1: pastro.ipynb, histogram-smoothed likelihoods",
        title="Signal and noise likelihoods over the ranking statistic",
        builder="build_significance",
        requires=(
            "stat",
            "p_signal",
            "p_noise",
            "support_lo",
            "support_hi",
            "trigger_hist_edges",
            "trigger_hist_counts",
        ),
        sources=("pastro",),
        note=(
            "Both densities on one axis with the observed triggers behind them. The "
            "common support is marked, since a probability formed from two differently "
            "truncated densities is decided by the truncation."
        ),
    )
)
declare(
    FigureDecl(
        key="pastro_signal_model_comparison",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Signal likelihood from the population against the training distribution",
        builder="build_significance",
        requires=("stat", "p_signal_population", "p_signal_training", "p_noise"),
        sources=("pastro", "injections"),
        note=(
            "The signal model is the choice p_astro is most sensitive to; showing both "
            "makes the dependence explicit rather than buried in a configuration flag."
        ),
    )
)
declare(
    FigureDecl(
        key="pastro_rate_posterior",
        origin="sgwc-1: pastro.ipynb, posterior over (lambda_s, lambda_n)",
        title="Rate posterior over signal and noise counts",
        builder="build_significance",
        requires=("lambda_s_grid", "lambda_n_grid", "log_posterior", "map_lambda_s", "map_lambda_n"),
        sources=("pastro",),
    )
)
declare(
    FigureDecl(
        key="pastro_rate_posterior_reparam",
        origin="sgwc-1: pastro.ipynb, posterior over (lambda, r)",
        title="Rate posterior over total rate and signal fraction",
        builder="build_significance",
        requires=("total_grid", "fraction_grid", "log_posterior", "map_total", "map_fraction"),
        sources=("pastro",),
        note=(
            "The same posterior in the coordinates the inference is actually sensitive "
            "in. A posterior that looks well constrained in counts can be degenerate here."
        ),
    )
)
declare(
    FigureDecl(
        key="pastro_threshold_invariance",
        origin="sgwc-1: pastro.ipynb, p_astro vs threshold",
        title="Astrophysical probability as the inclusion threshold is varied",
        builder="build_significance",
        requires=("threshold_stat", "p_astro_by_event", "event_name", "p_astro_sigma"),
        sources=("pastro",),
        note=(
            "The diagnostic that exposed the earlier failure: a well-behaved p_astro "
            "settles as the threshold is lowered, and one driven by the low-statistic "
            "bulk does not. Plotted against the posterior uncertainty, not a fixed "
            "tolerance."
        ),
    )
)
declare(
    FigureDecl(
        key="pastro_trigger_populations",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Trigger populations entering the rate inference",
        builder="build_significance",
        requires=("stat_edges", "counts_all", "counts_signal", "counts_noise", "counts_above_threshold"),
        sources=("pastro", "background", "injections"),
        note="Shows how much of the trigger set the threshold actually removes.",
    )
)

# -------------------------------------------------- injections and catalogue extras
declare(
    FigureDecl(
        key="snr_versus_distance",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Network optimal signal-to-noise ratio against distance",
        builder="build_sensitivity",
        requires=("network_snr", "distance_mpc", "mchirp", "found"),
        sources=("injections",),
    )
)
declare(
    FigureDecl(
        key="search_timeline",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Candidates and published events across the observing run",
        builder="build_catalogue",
        requires=(
            "gps",
            "mchirp",
            "source",
            "is_new",
            "is_ood",
            "far_per_yr",
            "coincident_gps",
            "published_gps",
            "published_mchirp",
            "published_high_far",
        ),
        sources=("candidates", "catalogue", "far"),
        note=(
            "The summary figure of the whole analysis: every candidate below a stated "
            "false-alarm rate against time and chirp mass, with published events, new "
            "candidates, and out-of-distribution marks distinguished."
        ),
    )
)
declare(
    FigureDecl(
        key="missed_in_distribution",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Published events inside the search region that were not recovered",
        builder="build_catalogue",
        requires=("event_name", "mass1_samples", "mass2_samples", "id_fraction", "recovered"),
        sources=("catalogue", "candidates"),
        note=(
            "The events that most need explaining: inside the training distribution and "
            "still missed. An out-of-distribution miss is expected; this is not."
        ),
    )
)
declare(
    FigureDecl(
        key="mass_plane_posteriors",
        deferred=(
            "reads published GWTC posterior samples through pesummary. Parameter estimation is a separate effort on a separate timescale and is not this search's work; deferred 2026-08-20 until those posteriors exist to act on"
        ),
        origin="sgwc-1: catalogue.ipynb, m2 vs m1",
        title="Component masses with the search region overlaid",
        builder="build_population",
        requires=(
            "event_name",
            "mass1_samples",
            "mass2_samples",
            "credible_level",
            "search_region_vertices",
            "mass_ratio_lines",
            "is_ood",
        ),
        sources=("catalogue",),
        note=(
            "Posterior contours per event against the region the network was trained on, "
            "with the equal-mass and extreme-mass-ratio lines drawn. This is what "
            "in-distribution and out-of-distribution mean, shown rather than asserted."
        ),
    )
)
declare(
    FigureDecl(
        key="mchirp_q_coverage",
        deferred=(
            "reads published GWTC posterior samples through pesummary. Deferred with mass_plane_posteriors, 2026-08-20: the search is what is being built now"
        ),
        origin="sgwc-1: catalogue.ipynb, q vs chirp mass",
        title="Chirp mass against mass ratio, with the search boundary",
        builder="build_population",
        requires=("mchirp", "mass_ratio", "boundary_vertices"),
        sources=("candidates",),
    )
)
# Preprocessing, whitening and waveform-generation checks are deliberately absent: the
# search uses Sage's own generation and preprocessing, which sage.diagnostics already
# covers. Nothing here re-diagnoses machinery this package only consumes.


# ------------------------------------------------------------------------ per event
declare(
    FigureDecl(
        key="event_spectrograms",
        deferred=(
            "per-event follow-up, which runs after the search and in its own environment. Deferred 2026-08-20 so the search's own figures land first"
        ),
        function="spectrograms",
        origin="sgwc-1: qtransform.ipynb, make_scan",
        title="Time-frequency spectrograms around a candidate",
        builder="build_event",
        requires=("times", "frequencies", "energy", "detector", "gps", "q", "duration_s"),
        sources=("qscans",),
        per_event=True,
    )
)
declare(
    FigureDecl(
        key="event_whitened_strain",
        function="whitened_strain",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Whitened strain around a candidate",
        builder="build_event",
        requires=("time", "strain_whitened", "detector", "gps"),
        sources=("dataquality",),
        per_event=True,
    )
)
declare(
    FigureDecl(
        key="event_snr_series",
        function="snr_series",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Matched-filter signal-to-noise series",
        builder="build_event",
        requires=("time", "snr", "detector", "peak_time", "peak_snr"),
        sources=("followup_mf",),
        per_event=True,
        optional=True,
        note=(
            "A characterisation aid only. No inclusion criterion depends on a "
            "matched-filter result."
        ),
    )
)
declare(
    FigureDecl(
        key="event_spectra",
        function="spectra",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Amplitude spectral density around a candidate",
        builder="build_event",
        requires=("frequency", "asd", "detector", "gps"),
        sources=("dataquality",),
        per_event=True,
    )
)
declare(
    FigureDecl(
        key="event_consistency_summary",
        function="consistency_summary",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Signal-consistency tests for a candidate",
        builder="build_event",
        requires=("test_name", "value", "passed", "detail"),
        sources=("consistency",),
        per_event=True,
    )
)
declare(
    FigureDecl(
        key="event_posterior",
        function="posterior_samples",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Posterior samples for a candidate",
        builder="build_event",
        requires=("samples", "parameter_names", "weights"),
        sources=("pe",),
        per_event=True,
        optional=True,
        note="Requires external parameter estimation; absent for most candidates.",
    )
)
declare(
    FigureDecl(
        key="event_localisation",
        function="localisation",
        origin="",
        deferred="no counterpart in sgwc-1 or PyCBC; deferred by ruling of 2026-08-20 -- keep what those two produce, revisit the rest later",
        title="Sky localisation for a candidate",
        builder="build_event",
        requires=("skymap_path", "area_50_deg2", "area_90_deg2"),
        sources=("skymaps",),
        per_event=True,
        optional=True,
    )
)

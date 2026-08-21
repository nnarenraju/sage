#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : spec.py
Description   : The search configuration surface. One spec describes one observing run.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One spec describes one *arm*: a single detector network searching a single observing run.
Background, tail fit, FAR curve and the p_astro noise density all belong to
that arm and are never shared across arms. Running two networks over the same run, or one
network over two runs, is two specs; their candidate lists are combined afterwards, with
the trials factor from :mod:`sage.search.trials`.

The whitening spectra are the caller's choice, exactly as for a training run. They
default to the set recorded in the checkpoint, which is what the network was trained
with, and any other set may be given. Whichever is used is recorded in provenance.
"""

import hashlib
import json
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple

CLUSTER_LINKAGES: Tuple[str, ...] = ("peak", "single")
# Pinned to sage.search.pastro.monotonic.apply_policy, which accepts these two and no
# others. "remap" was deliberately removed: reparameterising the statistic by a monotone
# regression of the density ratio has no counterpart in sgwc-1, PyCBC, FGMC, Banagiri or
# any GWTC methods paper, and a fit estimated from the same densities it reparameterises
# is exactly the failure the gate exists to detect. "fail" was this vocabulary's name for
# what the implementation calls "stop".
MONOTONICITY_POLICIES: Tuple[str, ...] = ("stop", "restrict")
TC_SOURCES: Tuple[str, ...] = ("checkpoint", "gwconfig", "explicit")

# Duplicated from background.py and tail.py rather than imported: validate() runs in every
# stage, including ones that never touch a background, and importing those modules here
# would pull numpy's full stack into a submit script that only wanted to read the graph.
# The test suite pins the two copies together.
REMOVAL_MODES: Tuple[str, ...] = ("inclusive", "exclusive", "hierarchical")
STOP_RULES: Tuple[str, ...] = ("significance", "counted")
THRESHOLD_METHODS: Tuple[str, ...] = ("count", "stability")
TRIALS_CONVENTIONS: Tuple[str, ...] = ("coverage", "detection", "fixed", "none")


def read_tc_prior(gwconfig: str | Path) -> Tuple[float, float]:
    """
    The ``tc`` prior bounds from a training run's ``gwconfig.yaml``.

    Parsed from the YAML rather than through
    :func:`sage.data.waveform.read_from_config`, which builds a live parameter sampler and
    pulls the waveform stack -- torch included -- to deliver two floats. Importing
    ``sage.search`` must not pull torch, and every stage in a campaign reads the geometry.

    Only a uniform ``tc`` prior is accepted. Sage places the merger uniformly in a narrow
    band of the window and :func:`sage.search.decode.tc_to_gps` inverts that placement
    through the band's endpoints; a prior of another shape has endpoints that do not mean
    the same thing, and taking them anyway would decode every merger time through the
    wrong map.
    """
    path = Path(gwconfig)
    if not str(gwconfig) or path == Path():
        raise ValueError(
            "no gwconfig on this spec: the coalescence-time prior is not recorded in a "
            "Sage checkpoint, so the training run's own gwconfig.yaml is the only place "
            "it exists. Set engine.gwconfig, or state the bounds with tc_source='explicit'"
        )
    if not path.is_file():
        raise FileNotFoundError(f"no parameter prior at {path}")
    import yaml

    document = yaml.safe_load(path.read_text()) or {}
    prior = (document.get("priors") or {}).get("tc")
    if not isinstance(prior, dict):
        raise ValueError(
            f"{path} declares no tc prior; without it a decoded tc cannot be placed in "
            "the window, and every trigger's merger time would be offset by however far "
            "the assumed band sits from the trained one"
        )
    name = str(prior.get("name", "")).lower()
    if name != "uniform":
        raise ValueError(
            f"{path} declares a {name!r} tc prior; only 'uniform' is supported, because "
            "decode.tc_to_gps inverts a uniform placement through the band's endpoints "
            "and another shape's endpoints do not mean the same thing"
        )
    return float(prior["min"]), float(prior["max"])


@dataclass(frozen=True)
class GeometrySpec:
    """Window/stride/tc conventions; materialised into a ``SearchGeometry``."""

    stride_samples: int = 205
    tc_source: str = "checkpoint"  # checkpoint | gwconfig | explicit
    tc_lower_s: Optional[float] = None
    tc_upper_s: Optional[float] = None


@dataclass(frozen=True)
class DataSpec:
    """Strain release, detector network and the fiducial ASD directory."""

    observing_run: str = ""
    detectors: Tuple[str, ...] = ("H1", "L1")
    release_dir: Path = Path()
    fiducial_dir: Path = Path()
    apply_cat1: bool = True
    cat1_cache_dir: Optional[Path] = None
    gate_loud_glitches: bool = False


@dataclass(frozen=True)
class EngineSpec:
    """Inference-loop knobs. ``batch_size`` is an upper bound, clamped per segment."""

    checkpoint: Path = Path()
    #: Config module of the training run that produced ``checkpoint``. Provenance, and the
    #: anchor for :attr:`gwconfig`: a campaign whose outputs do not name the training
    #: configuration cannot be traced back to it.
    training_config: str = ""
    #: The parameter prior the network was trained under -- the training run's own
    #: ``gwconfig.yaml``. Not optional and not derivable from the checkpoint, which records
    #: the geometry but not the prior. Two things need it: the coalescence-time bounds,
    #: which fix where in the window a merger sits and therefore how a decoded ``tc``
    #: becomes a GPS time; and the mass bounds, which fix the dyadic multirate binning and
    #: therefore what the network is actually fed. Both fail silently if wrong.
    gwconfig: Path = Path()
    #: Registry name of the class the weights are loaded into, for a checkpoint that does
    #: not record one -- which every current Sage checkpoint is. Stated here so the choice
    #: is part of the configuration, the hash and the provenance, rather than a default
    #: inside the loader. See :data:`sage.search.checkpoint.ARCHITECTURES`.
    architecture: str = "mscnn1d_2dresnetcbam_hardmining"
    #: Seed for the parameter sampler whose buffers invert the point-estimate encoding.
    #: It reaches a physical number: ``_compile_batch_standardiser`` estimates each
    #: target's mean and standard deviation from a million draws, so the decode carries
    #: that estimate's Monte Carlo noise. ``runs/*/train_hard.py`` derives its own seed
    #: from the resume epoch, so no single value reproduces training exactly and this one
    #: is ``BASE_SEED``, the value a run that never resumed used. Measured cost of the
    #: choice, seeds 150914 vs 170817 on ``runs/o3b/gwconfig.yaml``: at most 1e-4 s in
    #: ``tc`` and 0.013 solar masses in ``mchirp``. Recorded here so it is in the hash and
    #: the provenance rather than buried in a default.
    sampler_seed: int = 150914
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    batch_size: int = 8192
    block_seconds: float = 32768.0
    keep_stream: bool = False
    #: Reuse per-detector frontend features across slides instead of rescoring each slide
    #: from raw strain. **Off by default.** The exact path scores every slide end to end
    #: and is correct for any architecture; the cache is ~3x cheaper but is valid only
    #: where the frontend is separable, which is a property of the trained network and is
    #: measured, never assumed. Turning it on makes
    #: :meth:`sage.search.network.SplitNetwork.separability` a hard gate at load time.
    use_frontend_cache: bool = False
    cache_device: str = "cuda"
    parity_mode: bool = False


@dataclass(frozen=True)
class SlideSpec:
    """Time-slide ladder. Lags are stratified, seeded and stored, never stacked."""

    n_slides: int = 82
    reference_detector: str = "H1"
    min_separation_s: float = 20.0
    tau_max_s: float = 8192.0
    guard_s: float = 4.0
    seed: int = 20260809


@dataclass(frozen=True)
class ClusterSpec:
    """Trigger clustering. ``peak`` for production, ``single`` pinned for parity."""

    window_s: float = 0.35
    linkage: str = "peak"  # peak | single
    halo_s: float = 1.0


@dataclass(frozen=True)
class TrialsSpec:
    """
    How many chances noise had, and which analyses count as chances.

    Searching one stretch of data with more than one network gives noise more than one
    opportunity to produce something loud, so a FAR measured *within* one arm understates
    how often the campaign as a whole throws up a candidate that significant. The factor
    is per candidate rather than per campaign: a three-detector arm only analyses time
    when all three were observing, so a candidate falling where Virgo was down was
    reachable by fewer analyses than one in triple-coincident time.
    """

    #: ``coverage`` counts the arms whose own analysed lattice contains the candidate's
    #: time, whether or not they produced a trigger there -- an arm that *could* have
    #: produced a false alarm is a chance noise had. ``detection`` counts only the arms
    #: that did. ``fixed`` uses :attr:`fixed_factor` throughout, and ``none`` is exactly
    #: one, which is the honest setting for a single-arm campaign.
    convention: str = "coverage"
    #: Campaign configs whose arms compete with this one, as ``load_spec`` accepts them.
    #: Each must have completed ``far``: the factor needs the other arm's analysed
    #: lattice and its livetime, and an arm that has not run has neither. Empty means a
    #: single-arm campaign, where every convention gives one.
    sibling_configs: Tuple[str, ...] = ()
    #: The factor under the ``fixed`` convention, for reproducing a published number that
    #: quoted one.
    fixed_factor: Optional[int] = None
    #: How close two arms' triggers must be to count as the same event. Wider than the
    #: light travel time across the network (27.3 ms for H1-V1), because the arms
    #: estimate the time independently and their errors do not cancel.
    match_window_s: float = 0.1


@dataclass(frozen=True)
class SignificanceSpec:
    """
    How triggers become false-alarm rates: what is kept, what is removed, what is fitted.

    Every field here changes a published number, so all of them belong in the spec hash
    and in the provenance rather than as defaults inside a driver.
    """

    #: Fraction of zero-lag windows whose individual triggers are written. Frozen once
    #: from the COMPLETE zero-lag histogram and applied to every slide, so the background
    #: and the foreground are thresholded identically. Calibrating per slide would let
    #: each keep a different amount of its own tail.
    keep_rate: float = 1e-4
    #: Removal modes to build. The inclusive set is always produced; the others are
    #: reported beside it, never instead of it, because each removes genuine noise along
    #: with any signal and quoting one alone would present that as a measurement.
    removal_modes: Tuple[str, ...] = ("inclusive", "exclusive", "hierarchical")
    #: Half-width of the exclusive background's coincidence test, PyCBC's ``--veto-window``.
    veto_window_s: float = 0.1
    #: Half-width of the hierarchical removal window, PyCBC's
    #: ``--hierarchical-removal-window``.
    hierarchical_window_s: float = 1.0
    #: Which published rule ends the hierarchical walk; see
    #: :data:`sage.search.background.STOP_RULES`.
    stop_rule: str = "significance"
    #: Consecutive clean background events that end a ``"counted"`` walk.
    ignore_limit: int = 200
    #: Floor on surviving background livetime. Zero means no floor, which is what both
    #: reference pipelines do.
    min_background_livetime_s: float = 0.0
    #: Hard bound on hierarchical removals, PyCBC's ``--max-hierarchical-removal``.
    max_removals: int = 100
    #: Exceedances the peaks-over-threshold fit is given, PyCBC's ``tail_threshold`` rule.
    n_exceedances: int = 500
    #: How the fit threshold is chosen; see
    #: :func:`sage.search.tail.choose_threshold`. ``"count"`` is the only rule with a
    #: counterpart in any reference pipeline.
    threshold_method: str = "count"
    #: Cap on a quoted inverse false-alarm rate, matching the GWTC convention.
    ifar_cap_yr: float = 1000.0
    #: Bin width for the Poisson-vs-negative-binomial dispersion check, in seconds.
    #: 10 s is what sgwc-1 uses (``search.ipynb`` cell 331). The check is reported beside
    #: the FAR curve and acts on nothing, but the width decides what it can see: too
    #: narrow and every bin holds zero or one event, too wide and a glitch-active hour is
    #: averaged into a quiet day.
    dispersion_bin_s: float = 10.0
    #: False-alarm rate bounding the public candidate list, per day. Distinct from
    #: ``pastro.threshold_far_per_day``, which sets the analysis threshold the mixture is
    #: fitted above: one decides which triggers are *published*, the other decides which
    #: are *counted*. They are conventionally the same number and are not the same knob,
    #: and tying them would make widening the list refit the rates.
    candidate_far_per_day: float = 2.0


@dataclass(frozen=True)
class InjectionSpec:
    """
    Injections drawn from the GWTC-3 population and injected into real strain.

    Not a published sensitivity-estimate release. The injections' ranking statistics are
    what p_astro uses for ``p(x|signal)``, so they have to be scored by this network,
    through this preprocessor, on this run's noise -- none of which an external release
    can supply. The population itself is the published one
    (:mod:`sage.search.injection.population`); it is the *drawing and scoring* that is
    local, and it follows sgwc-1's ``injection_study``.
    """

    #: Bilby result holding the GWTC-3 Power-Law + Peak hyperposterior. The MAP sample of
    #: this is the population injections are drawn from.
    hyperposterior_path: Optional[Path] = None
    #: Where the drawn parameter set is written, and reread from on a re-run. Drawing is
    #: seeded, so this is a cache rather than a source of truth -- but the campaign that
    #: scored a set must keep the set it scored.
    staged_path: Optional[Path] = None
    #: How many injections to draw before the chirp-mass cut. sgwc-1 draws 100,000 and
    #: keeps those whose chirp mass lies inside the training prior.
    n_draw: int = 100_000
    #: Independent draws, for splitting the campaign across array tasks.
    streams: Tuple[int, ...] = (0,)
    assoc_window_s: float = 12.0
    match_window_s: float = 0.25
    found_far_yr: float = 1.0


@dataclass(frozen=True)
class PastroSpec:
    """FGMC settings. BBH + Terrestrial; the category axis stays pluggable."""

    categories: Tuple[str, ...] = ("BBH", "Terrestrial")
    threshold_far_per_day: float = 2.0
    #: Astrophysical probability above which a candidate is called confident. The GWTC
    #: convention. Lives here rather than in SignificanceSpec because it is a statement
    #: about the probability this stage produces, and the tier ladder reads both.
    confident_p_astro: float = 0.5
    #: FAR below which a confident candidate is worth full parameter estimation, per year.
    pe_far_per_yr: float = 1.0
    resolve_mchirp: bool = True
    monotonicity_policy: str = "restrict"  # stop | restrict
    n_rate_grid: int = 512


@dataclass(frozen=True)
class CatalogueSpec:
    """
    Catalogue ingest and cross-match.

    Every source is compared on results only. Each carries its own ``Conventions``, and a
    comparison refuses to place two incompatible significances on the same axis rather
    than quietly doing it.
    """

    gwtc_endpoint: str = "https://gwosc.org/eventapi/json/GWTC/"
    external: Tuple[str, ...] = (
        "IAS-O3a",
        "IAS-O3b",
        "IAS-HM",
        "4-OGC",
        "cWB-O3",
        "PyCBC-KDE",
        "AresGW",
    )
    match_tolerance_s: float = 1.0
    dedup_precedence: Tuple[str, ...] = ("LVK", "IAS", "OGC")
    cache_dir: Optional[Path] = None
    #: Files of event times to compare against, as
    #: :func:`sage.search.catalogue.eventlist.read_event_times` reads them: times written
    #: as GPS, UTC or event names, with or without a header. This is how an external
    #: catalogue, a subthreshold list or a glitch list enters the comparison -- one
    #: generic input rather than a parser per source, because every one of those
    #: restructures between releases and the layout is not what the comparison uses.
    #: Each entry is ``"key=path"``; the key names the source in the output columns.
    event_lists: Tuple[str, ...] = ()
    #: Include the marginal and sub-threshold GWOSC releases. A candidate matching a
    #: marginal event is not a discovery, it is a confirmation -- so leaving these out
    #: makes the new-event list longer and wrong.
    include_marginal: bool = True
    #: Refuse to reach the network: every catalogue must already be in the cache. On by
    #: default, because an analysis whose inputs can change under it between runs is not
    #: reproducible, and because a compute node generally has no route out anyway.
    #: Populate the cache once with a deliberate fetch, freeze it, and run from that.
    offline: bool = True


@dataclass(frozen=True)
class FigureSpec:
    """Figure set B01-B29; every figure is built from a persisted intermediate."""

    figures: Tuple[str, ...] = ()
    style: str = "gwtc"
    formats: Tuple[str, ...] = ("pdf",)


@dataclass(frozen=True)
class SearchSpec:
    """Top-level, frozen search configuration."""

    tag: str = ""
    config_module: str = ""
    out_dir: Path = Path()
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    data: DataSpec = field(default_factory=DataSpec)
    engine: EngineSpec = field(default_factory=EngineSpec)
    slides: SlideSpec = field(default_factory=SlideSpec)
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    significance: SignificanceSpec = field(default_factory=SignificanceSpec)
    trials: TrialsSpec = field(default_factory=TrialsSpec)
    injection: InjectionSpec = field(default_factory=InjectionSpec)
    pastro: PastroSpec = field(default_factory=PastroSpec)
    catalogue: CatalogueSpec = field(default_factory=CatalogueSpec)
    figures: FigureSpec = field(default_factory=FigureSpec)
    seed: int = 20260809

    @property
    def arm(self) -> str:
        """
        Short key for this network, from the detector initials: ``"HL"``, ``"HLV"``.

        Identifies the arm in the trials bookkeeping and in product filenames, so two
        networks searching the same run never collide.
        """
        return "".join(d[0] for d in self.data.detectors)

    def validate(self) -> None:
        """
        Check the configuration is self-consistent before any work begins.

        Everything here would otherwise surface part-way through a campaign, or not at
        all: a reference detector outside the network produces slides that mean nothing,
        an unknown linkage silently falls back to a default, and a campaign root under
        the system temp directory is liable to vanish.
        """
        out_dir = Path(self.out_dir)
        if not out_dir.is_absolute():
            raise ValueError(
                f"out_dir must be absolute, got {out_dir!s}; a relative root resolves "
                "differently depending on where a job starts"
            )
        if out_dir == Path("/tmp") or "/tmp/" in f"{out_dir}/":
            raise ValueError(
                f"out_dir must not be under /tmp, got {out_dir!s}; a campaign writes "
                "tens of gigabytes and must survive a reboot"
            )

        detectors = tuple(self.data.detectors)
        if not detectors:
            raise ValueError("data.detectors must name at least one detector")
        if len(set(detectors)) != len(detectors):
            raise ValueError(f"detectors repeated in network {detectors}")
        if not self.data.observing_run:
            raise ValueError("data.observing_run must be set")

        if self.slides.reference_detector not in detectors:
            raise ValueError(
                f"slides.reference_detector {self.slides.reference_detector!r} is not "
                f"in the network {detectors}; slides would be measured against a "
                "detector the search does not read"
            )
        if self.slides.n_slides < 0:
            raise ValueError(f"slides.n_slides must not be negative, got {self.slides.n_slides}")
        if self.slides.tau_max_s <= self.slides.min_separation_s:
            raise ValueError(
                f"slides.tau_max_s ({self.slides.tau_max_s}) must exceed "
                f"min_separation_s ({self.slides.min_separation_s}), or no lag is "
                "admissible"
            )

        if not 0.0 < self.significance.keep_rate <= 1.0:
            raise ValueError(
                f"significance.keep_rate must lie in (0, 1], got "
                f"{self.significance.keep_rate}"
            )
        unknown_modes = [
            mode
            for mode in self.significance.removal_modes
            if mode not in REMOVAL_MODES
        ]
        if unknown_modes:
            raise ValueError(
                f"unknown significance.removal_modes {unknown_modes}; expected a subset "
                f"of {REMOVAL_MODES}"
            )
        if "inclusive" not in self.significance.removal_modes:
            raise ValueError(
                "significance.removal_modes must include 'inclusive': the removed sets "
                "are reported beside it, never instead of it, since each removes genuine "
                "noise along with any signal"
            )
        if self.significance.stop_rule not in STOP_RULES:
            raise ValueError(
                f"unknown significance.stop_rule {self.significance.stop_rule!r}; "
                f"expected one of {STOP_RULES}"
            )
        if self.significance.threshold_method not in THRESHOLD_METHODS:
            raise ValueError(
                f"unknown significance.threshold_method "
                f"{self.significance.threshold_method!r}; expected one of "
                f"{THRESHOLD_METHODS}"
            )

        if self.trials.convention not in TRIALS_CONVENTIONS:
            raise ValueError(
                f"unknown trials.convention {self.trials.convention!r}; expected one of "
                f"{TRIALS_CONVENTIONS}"
            )
        if self.trials.convention == "fixed" and self.trials.fixed_factor is None:
            raise ValueError(
                "trials.convention is 'fixed' but no trials.fixed_factor is given; the "
                "fixed convention exists to reproduce a published factor, and there is "
                "no defensible default for one"
            )
        if self.trials.fixed_factor is not None and self.trials.fixed_factor < 1:
            raise ValueError(
                f"trials.fixed_factor is {self.trials.fixed_factor}; a factor below one "
                "would make a candidate more significant than the analysis that found it "
                "measured it to be"
            )
        if self.trials.match_window_s <= 0:
            raise ValueError(
                f"trials.match_window_s is {self.trials.match_window_s}; two arms' "
                "triggers can never be closer than zero apart, so nothing would match "
                "and every candidate would report having been found by one arm"
            )

        if self.cluster.linkage not in CLUSTER_LINKAGES:
            raise ValueError(
                f"unknown cluster.linkage {self.cluster.linkage!r}; "
                f"expected one of {CLUSTER_LINKAGES}"
            )
        if self.pastro.monotonicity_policy not in MONOTONICITY_POLICIES:
            raise ValueError(
                f"unknown pastro.monotonicity_policy "
                f"{self.pastro.monotonicity_policy!r}; expected one of "
                f"{MONOTONICITY_POLICIES}"
            )

        if self.geometry.tc_source not in TC_SOURCES:
            raise ValueError(
                f"unknown geometry.tc_source {self.geometry.tc_source!r}; "
                f"expected one of {TC_SOURCES}"
            )
        if self.geometry.tc_source == "explicit" and (
            self.geometry.tc_lower_s is None or self.geometry.tc_upper_s is None
        ):
            raise ValueError(
                "geometry.tc_source is 'explicit' but tc_lower_s/tc_upper_s are unset"
            )

        # The three inputs every campaign reads. Checked for shape here and for existence
        # in validate_inputs, so a spec can be built and tested without the data present.
        for label, value in (
            ("engine.checkpoint", self.engine.checkpoint),
            ("engine.gwconfig", self.engine.gwconfig),
            ("data.release_dir", self.data.release_dir),
            ("data.fiducial_dir", self.data.fiducial_dir),
        ):
            path = Path(value)
            if not str(value) or path == Path():
                raise ValueError(f"{label} must be set; the campaign reads it")
            if not path.is_absolute():
                raise ValueError(
                    f"{label} must be absolute, got {path!s}; a relative input resolves "
                    "differently depending on where a job starts, and array tasks do not "
                    "share a working directory"
                )

    def validate_inputs(self) -> None:
        """
        Check the inputs are actually there, before a campaign spends anything.

        Separate from :func:`validate` because that one is pure: a spec must be
        constructible and checkable on a machine that does not hold the release, which is
        how every unit test builds one. This is the call a driver makes on the cluster,
        where the absence of a fiducial directory is a failure worth having in the first
        second rather than after the segment stage.
        """
        missing = []
        checkpoint = Path(self.engine.checkpoint)
        if not checkpoint.is_file():
            missing.append(f"engine.checkpoint {checkpoint!s} is not a file")
        if not Path(self.engine.gwconfig).is_file():
            missing.append(
                f"engine.gwconfig {self.engine.gwconfig!s} is not a file"
            )
        for label, value in (
            ("data.release_dir", self.data.release_dir),
            ("data.fiducial_dir", self.data.fiducial_dir),
        ):
            if not Path(value).is_dir():
                missing.append(f"{label} {value!s} is not a directory")
        if missing:
            raise FileNotFoundError(
                "this campaign's inputs are not present: " + "; ".join(missing)
            )

    def hash(self) -> str:
        """
        Resumability key: sha256 over the spec JSON plus cheap input fingerprints.

        Sidecar JSONs are hashed by content; ``.bin`` files by (name, size, mtime_ns).
        Full ``.bin`` checksums are a separate opt-in task, since a single release runs
        to hundreds of gigabytes.

        Returns
        -------
        str
            Hex digest, stable across processes on one filesystem.

        Notes
        -----
        Built from a canonical JSON rendering rather than from ``repr`` or the builtin
        ``hash``: string hashing is salted per process, so a key derived from it would
        differ between the job that wrote a product and the job that resumes it, and
        every stage would be recomputed.

        **Not portable between machines.** The ``.bin`` leg keys on ``mtime_ns``, which a
        copy, an rsync without ``--times``, or a restore from tape will change without
        changing a byte of data -- and which two machines holding the same release will
        generally disagree about. Within one campaign directory on one filesystem it is
        exactly what is wanted: it is cheap, and it does notice a rebuilt release. Moving
        a campaign between filesystems invalidates it, and re-running is the correct
        response, since nothing here has verified the data survived the move. A content
        checksum of the ``.bin`` files is the separate opt-in task, and is what would make
        this portable.

        It is also blind to the code. Changing a stage's implementation without touching
        the configuration leaves every hash unchanged, so a campaign is not invalidated by
        a bug fix; force the affected stages instead.
        """
        digest = hashlib.sha256()
        digest.update(self.to_json(identity_only=True).encode("utf-8"))

        release_dir = Path(self.data.release_dir)
        run = self.data.observing_run
        for detector in sorted(self.data.detectors):
            sidecar = release_dir / f"data_{detector}_{run}_segments.json"
            if sidecar.is_file():
                digest.update(sidecar.name.encode("utf-8"))
                digest.update(sidecar.read_bytes())
            binary = release_dir / f"data_{detector}_{run}.bin"
            if binary.is_file():
                stat = binary.stat()
                digest.update(
                    f"{binary.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
                )
        return digest.hexdigest()

    #: Fields recorded but excluded from :meth:`hash`. They describe how the campaign was
    #: *named*, not what it is. ``config_module`` is stamped by :func:`load_spec` with
    #: whatever the caller typed -- an absolute path from a submit script, a dotted name
    #: from a person -- so including it gave one campaign as many identities as it had
    #: spellings, each with its own manifest record, and moving the checkout produced
    #: another. Every field a stage reads is unaffected, so two spellings differ in
    #: nothing that changes a number.
    PROVENANCE_ONLY: ClassVar[Tuple[str, ...]] = ("config_module",)

    def to_json(self, identity_only: bool = False) -> str:
        """
        Serialise for provenance attrs.

        Canonical: keys sorted and paths rendered as strings, so two equal specs produce
        byte-identical output and the hash built from it is reproducible.

        Parameters
        ----------
        identity_only : bool
            Drop :data:`PROVENANCE_ONLY` fields. Used by :meth:`hash`; the full form is
            what gets stamped into products.
        """

        def encode(value):
            if isinstance(value, Path):
                return str(value)
            raise TypeError(f"cannot serialise {type(value).__name__} in a spec")

        payload = asdict(self)
        if identity_only:
            for name in self.PROVENANCE_ONLY:
                payload.pop(name, None)
        return json.dumps(payload, sort_keys=True, default=encode)

    def tc_prior(self) -> Tuple[float, float]:
        """
        The coalescence-time prior the network was trained under, in window seconds.

        ``tc_source`` selects where it comes from:

        ``"explicit"``
            The bounds stated on this spec, for a network whose training prior is known
            but whose configuration is not to hand.
        ``"gwconfig"``, ``"checkpoint"``
            The training run's ``gwconfig.yaml``. These are one source today: a Sage
            checkpoint records the window geometry but **not** the prior -- there is no
            ``tc`` key anywhere in its ``cfg`` or ``data_cfg`` -- so "from the checkpoint"
            resolves to the configuration that checkpoint was trained under. Both names
            are kept so a checkpoint format that does record it can be honoured under
            ``"checkpoint"`` without every campaign changing its spelling.

        Read rather than assumed because it is not a convention: it is where in the
        analysis window a merger was placed during training, and
        :func:`sage.search.decode.tc_to_gps` turns a decoded ``tc`` into a GPS time
        through it. A band wrong by the 0.2 s the real one spans puts every trigger's
        merger time out by up to twice the window the background is vetoed on.
        """
        if self.geometry.tc_source == "explicit":
            if self.geometry.tc_lower_s is None or self.geometry.tc_upper_s is None:
                raise ValueError(
                    "geometry.tc_source is 'explicit' but tc_lower_s/tc_upper_s are unset"
                )
            return float(self.geometry.tc_lower_s), float(self.geometry.tc_upper_s)
        if self.geometry.tc_source not in TC_SOURCES:
            raise ValueError(
                f"unknown geometry.tc_source {self.geometry.tc_source!r}; "
                f"expected one of {TC_SOURCES}"
            )
        return read_tc_prior(self.engine.gwconfig)

    def window_lengths(self) -> Dict[str, float]:
        """
        Window geometry as the checkpoint records it, falling back to Sage's defaults.

        Read from the checkpoint where one is present: the window length and its padding
        are the network's input shape, and a search that assumed them would hand a network
        trained on 12 s a different number of samples, whitened by the wrong frequency
        bins. The defaults let a spec be built and validated on a machine that does not
        hold the checkpoint, which is how every unit test builds one.
        """
        defaults = {
            "sample_rate": 2048.0,
            "sample_length_in_s": 12.0,
            "padding_length_in_s": 2.0,
        }
        snapshot = self._window_snapshot()
        if snapshot is None:
            return defaults
        return {
            name: float(snapshot.get(name, value)) for name, value in defaults.items()
        }

    def _window_snapshot(self):
        """The checkpoint's recorded data configuration, or ``None`` if there is none."""
        path = Path(self.engine.checkpoint).parent / "data_cfg_snapshot.json"
        if not path.is_file():
            return None
        try:
            recorded = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return recorded if isinstance(recorded, dict) else None

    def geometry_object(self):
        """
        Build the :class:`~sage.search.geometry.SearchGeometry` for this spec.

        The window and padding come from the configuration the checkpoint was trained
        under; only the stride and the coalescence-time bounds are the search's own.
        """
        from sage.search.geometry import SearchGeometry

        lower, upper = self.tc_prior()
        window = self.window_lengths()
        return SearchGeometry(
            sample_rate=window["sample_rate"],
            signal_length_s=window["sample_length_in_s"],
            padding_length_s=window["padding_length_in_s"],
            stride_samples=self.geometry.stride_samples,
            tc_lower_s=lower,
            tc_upper_s=upper,
        )

    def apply_shadow_overrides(self, cfg, data_cfg) -> None:
        """
        Set search-only attributes on the ``BaseConfig`` wrappers.

        The training configuration is imported to get the geometry the network was trained
        under; a handful of its fields then have to say what the *search* is doing rather
        than what the training run did. ``export_dir`` is the one that matters: a training
        config points at a live run's directory, and anything the search wrote through it
        would land among that run's checkpoints.

        Mutating the wrapper, not the underlying class. The wrapper is a per-process view
        and the class is shared, so setting attributes on the class would follow the
        import into any other code that touched the same training config in this process.

        Applied by :func:`sage.search.engine.run_search` before the configs are
        registered, because the sampler, the whitener and the multirate binning all read
        the global config rather than taking arguments -- so a campaign that skipped this
        would run on the *training* run's device and whiten with the *training* run's
        fiducial spectra, both of which look like working configurations.

        The geometry is deliberately **not** overridden. It is the network's input shape,
        it is what :func:`sage.search.checkpoint.validate_geometry` compares the checkpoint
        against, and a search that quietly rewrote it would make that comparison compare
        the configuration with itself.
        """
        overrides = (
            (cfg, "export_dir", str(self.out_dir)),
            (cfg, "device", str(self.engine.device)),
            (cfg, "batch_size", int(self.engine.batch_size)),
            (cfg, "fiducial_dir", str(self.data.fiducial_dir)),
            # data_dir, which is the name every training data config actually uses.
            # "release_dir" named no field on any config, so the override was silently a
            # no-op that also added an attribute nothing reads.
            (data_cfg, "data_dir", str(self.data.release_dir)),
        )
        for target, name, value in overrides:
            if target is not None:
                setattr(target, name, value)

    def path(self, *parts: str) -> Path:
        """Resolve a path under ``out_dir``."""
        return Path(self.out_dir).joinpath(*parts)


def _unleak_siblings(before: dict, parent: str, keep: str) -> None:
    """
    Undo the sibling imports a config performed under bare names.

    A config does ``from config_base import make_spec``, and that caches ``config_base``
    in ``sys.modules`` globally for the rest of the process. Every ``runs/`` directory has
    one, and they are different modules with different signatures: after a search config
    loads, ``runs/o3b/config_HL.py``'s ``from config_base import register`` binds the
    *search* registrar and raises on its own arguments -- and after a training config
    loads, the next search config fails to import at all. Symmetric, order-dependent, and
    invisible until the second config in a process needs the name.

    So a bare-name module that (a) was not there before and (b) was loaded out of the
    directory that was put on the path for this config is removed again. The config's own
    private entry is kept -- the caller still holds the module object.

    Only newly added names are touched: a module of the same name that was already
    imported for its own reasons stays exactly as it was.
    """
    import sys

    for name, module in list(sys.modules.items()):
        if name == keep or name in before or "." in name:
            continue
        origin = getattr(module, "__file__", None)
        if origin and str(Path(origin).resolve().parent) == str(Path(parent).resolve()):
            del sys.modules[name]


def load_spec(module_or_path: str) -> SearchSpec:
    """
    Import a ``runs/search/config_*.py`` module and return its ``SearchSpec``.

    Accepts either a dotted module name or a path to the file. A campaign is launched from
    a config module by every driver and by every SLURM array task, so both spellings turn
    up: a submit script naturally has the path, a person naturally types the name.

    The module must expose ``get_spec()``, matching ``runs/search/config_o4a_HL.py``, or a
    module-level ``SPEC``. Nothing is guessed beyond those two: a config that exposes
    neither is an error rather than a search for the first ``SearchSpec``-shaped object,
    because picking one of several would decide the campaign silently.

    ``config_module`` is stamped onto the returned spec when the config left it empty, so
    every product records which configuration produced it and a campaign can be traced back
    from its outputs. A file is stamped with its resolved path, not its stem: two runs
    directories both holding ``config_HL.py`` are a normal arrangement, and the stem alone
    would not say which one ran.

    A file loaded by path is registered in ``sys.modules`` under a private name, and the
    entry is removed if it fails to execute. Registering it under its bare stem would let
    ``config_HL.py`` displace an installed module of that name for the rest of the process,
    and a failed load would leave a half-executed module behind for the next import to
    find.

    Raises
    ------
    FileNotFoundError
        The path does not exist.
    ValueError
        The module has neither entry point, or its entry point returned something other
        than a :class:`SearchSpec`.
    ModuleNotFoundError
        A dotted name that is not importable, re-raised with the name it was given.
    """
    import importlib
    import importlib.util
    import sys

    target = str(module_or_path)
    path = Path(target)
    if path.suffix == ".py" or path.exists():
        if not path.is_file():
            raise FileNotFoundError(f"no config module at {path}")
        resolved = path.resolve()
        # Private and path-qualified, so loading a config can neither shadow an installed
        # module of the same stem nor collide with a second config of the same name.
        name = "sage_search_config_%s" % hashlib.sha256(
            str(resolved).encode("utf-8")
        ).hexdigest()[:16]
        spec_obj = importlib.util.spec_from_file_location(name, resolved)
        if spec_obj is None or spec_obj.loader is None:
            raise ValueError(f"{path} could not be imported as a Python module")
        module = importlib.util.module_from_spec(spec_obj)
        # A config imports its sibling `config_base`, so the file's own directory has to
        # be importable while it loads. Registered in sys.modules first so that a config
        # importing itself indirectly does not re-execute.
        parent = str(resolved.parent)
        added = parent not in sys.path
        if added:
            sys.path.insert(0, parent)
        sys.modules[name] = module
        before = dict(sys.modules)
        try:
            spec_obj.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        finally:
            if added:
                try:
                    sys.path.remove(parent)
                except ValueError:
                    pass
            _unleak_siblings(before, parent, keep=name)
        stamp = str(resolved)
    else:
        try:
            module = importlib.import_module(target)
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"{target!r} is neither an existing file nor an importable module; a "
                "config is given either as a path to its .py file or as a dotted name "
                "already on sys.path"
            ) from error
        stamp = target

    if hasattr(module, "get_spec"):
        result = module.get_spec()
    elif hasattr(module, "SPEC"):
        result = module.SPEC
    else:
        raise ValueError(
            f"{target} exposes neither get_spec() nor SPEC, so it does not describe a "
            "campaign; see runs/search/config_o4a_HL.py for the expected shape"
        )
    if not isinstance(result, SearchSpec):
        raise ValueError(
            f"{target} returned {type(result).__name__}, not a SearchSpec"
        )
    if not result.config_module:
        result = dataclasses.replace(result, config_module=stamp)
    return result

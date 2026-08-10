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
Background, tail fit, FAR curve, calibration and the p_astro noise density all belong to
that arm and are never shared across arms. Running two networks over the same run, or one
network over two runs, is two specs; their candidate lists are combined afterwards, with
the trials factor from :mod:`sage.search.trials`.

The whitening spectra are the caller's choice, exactly as for a training run. They
default to the set recorded in the checkpoint, which is what the network was trained
with, and any other set may be given. Whichever is used is recorded in provenance.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

CLUSTER_LINKAGES: Tuple[str, ...] = ("peak", "single")
MONOTONICITY_POLICIES: Tuple[str, ...] = ("fail", "restrict", "remap")
TC_SOURCES: Tuple[str, ...] = ("checkpoint", "gwconfig", "explicit")


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
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    batch_size: int = 8192
    block_seconds: float = 32768.0
    keep_stream: bool = False
    use_frontend_cache: bool = True
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
class InjectionSpec:
    """Official LVK Zenodo injection set; never regenerate the population."""

    zenodo_record: str = ""
    staged_path: Optional[Path] = None
    streams: Tuple[int, ...] = (0,)
    assoc_window_s: float = 12.0
    match_window_s: float = 0.25
    found_far_yr: float = 1.0


@dataclass(frozen=True)
class PastroSpec:
    """FGMC settings. BBH + Terrestrial; the category axis stays pluggable."""

    categories: Tuple[str, ...] = ("BBH", "Terrestrial")
    threshold_far_per_day: float = 2.0
    resolve_mchirp: bool = True
    monotonicity_policy: str = "restrict"  # fail | restrict | remap
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

    def hash(self) -> str:
        """
        Resumability key: sha256 over the spec JSON plus cheap input fingerprints.

        Sidecar JSONs are hashed by content; ``.bin`` files by (name, size, mtime_ns).
        Full ``.bin`` checksums are a separate opt-in task, since a single release runs
        to hundreds of gigabytes.

        Returns
        -------
        str
            Hex digest, stable across processes and machines.

        Notes
        -----
        Built from a canonical JSON rendering rather than from ``repr`` or the builtin
        ``hash``: string hashing is salted per process, so a key derived from it would
        differ between the job that wrote a product and the job that resumes it, and
        every stage would be recomputed.
        """
        digest = hashlib.sha256()
        digest.update(self.to_json().encode("utf-8"))

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

    def to_json(self) -> str:
        """
        Serialise for provenance attrs.

        Canonical: keys sorted and paths rendered as strings, so two equal specs produce
        byte-identical output and the hash built from it is reproducible.
        """

        def encode(value):
            if isinstance(value, Path):
                return str(value)
            raise TypeError(f"cannot serialise {type(value).__name__} in a spec")

        return json.dumps(asdict(self), sort_keys=True, default=encode)

    def geometry_object(self):
        """
        Build the :class:`~sage.search.geometry.SearchGeometry` for this spec.

        The window and padding come from the configuration the checkpoint was trained
        under; only the stride and the coalescence-time bounds are the search's own.
        """
        from sage.search.geometry import SearchGeometry

        if self.geometry.tc_source != "explicit":
            raise NotImplementedError(
                f"tc_source {self.geometry.tc_source!r} is resolved by the checkpoint "
                "loader, which lands with the model spine"
            )
        return SearchGeometry(
            sample_rate=2048.0,
            signal_length_s=12.0,
            padding_length_s=2.0,
            stride_samples=self.geometry.stride_samples,
            tc_lower_s=float(self.geometry.tc_lower_s),
            tc_upper_s=float(self.geometry.tc_upper_s),
        )

    def apply_shadow_overrides(self, cfg, data_cfg) -> None:
        """
        Set search-only attributes on the BaseConfig wrappers.

        Mutating the wrapper (not the underlying class) keeps a live training run's
        export directory untouched.
        """
        raise NotImplementedError

    def path(self, *parts: str) -> Path:
        """Resolve a path under ``out_dir``."""
        return Path(self.out_dir).joinpath(*parts)


def load_spec(module_or_path: str) -> SearchSpec:
    """Import a ``runs/search/config_*.py`` module and return its ``SearchSpec``."""
    raise NotImplementedError

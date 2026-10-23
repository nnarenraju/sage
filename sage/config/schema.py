#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : schema.py
Description     : User-facing run-specification schema.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan


What this is
------------
The schema for the small YAML a user writes to describe a run. It deliberately
exposes only the things that genuinely vary between runs:

  * which observing run's data to train / validate / test on, and on which
    detectors;
  * the handful of prior ranges people actually change (masses, spins,
    distance), by named set or explicit range;
  * which stages to execute.

Everything else - architecture, optimiser, loss weights, whitening, mining -
is fixed by the named ``preset``, which encodes methodology that has already
been validated by experiment. That is the point: a run spec records *intent*,
not implementation.

If a study genuinely needs to change something outside this surface, the answer
is not to widen the surface. Either promote the change into a new preset (once
it is validated, so it becomes reusable and named), or use the ``custom``
escape hatch to supply your own module for one stage while still using the
standard data selection, staging and diagnostics.

Validation
----------
Unknown keys are an error, not a silent no-op. The previous config system
forwarded attribute access to a plain object, so a typo'd field simply never
took effect and the run completed with the wrong settings. Every rejection here
carries a "did you mean" suggestion where one is plausible.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "ConfigError",
    "DataSelection",
    "DataSection",
    "PriorsSection",
    "CustomSection",
    "RunSpec",
    "KNOWN_STAGES",
]


class ConfigError(ValueError):
    """Raised when a run specification is invalid.

    Carries the YAML path of the offending key (e.g. ``data.train.detectors``)
    so the message points at a location in the user's file rather than at an
    internal field name.
    """


# Stages the runner knows how to execute, in canonical execution order.
# `search` and `benchmark` both depend on a trained model; `plots` consumes
# whatever earlier stages produced.
KNOWN_STAGES = (
    "train",
    "calibrate",
    "validate",
    "search",
    "benchmark",
    "diagnostics",
    "plots",
)


def _suggest(name: str, valid) -> str:
    """Return a ' (did you mean ...?)' fragment, or '' if nothing is close."""
    close = difflib.get_close_matches(name, list(valid), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _reject_unknown(data: Dict[str, Any], cls, path: str) -> None:
    """Raise if `data` carries keys that `cls` does not declare."""
    valid = {f.name for f in fields(cls)}
    for key in data:
        if key not in valid:
            raise ConfigError(
                f"{path}: unknown option {key!r}{_suggest(key, valid)}. "
                f"Valid options here: {', '.join(sorted(valid))}."
            )


def _require_mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(
            f"{path}: expected a mapping, got {type(value).__name__}."
        )
    return value


@dataclass
class DataSelection:
    """Which data to use for one role (train, validate or test).

    Parameters
    ----------
    run : str
        Observing-run name, e.g. ``"O3a"``. This is the primary axis a user
        varies: the methodology is identical between runs, only the data
        differs.
    detectors : list of str, optional
        Detector set, e.g. ``["H1", "L1"]``. When omitted, inherits the
        detectors given for ``train`` - validating or testing on a different
        detector set than you trained on is unusual and should be deliberate.
    """

    run: str
    detectors: Optional[List[str]] = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "DataSelection":
        # Allow the shorthand `train: O3a` for the common case.
        if isinstance(value, str):
            return cls(run=value)

        data = _require_mapping(value, path)
        _reject_unknown(data, cls, path)

        if "run" not in data:
            raise ConfigError(
                f"{path}: missing required option 'run' "
                f"(the observing run to use, e.g. 'O3b')."
            )

        run = data["run"]
        if not isinstance(run, str):
            raise ConfigError(
                f"{path}.run: expected an observing-run name as a string, "
                f"got {type(run).__name__}."
            )

        dets = data.get("detectors")
        if dets is not None:
            if not isinstance(dets, list) or not all(isinstance(d, str) for d in dets):
                raise ConfigError(
                    f"{path}.detectors: expected a list of detector names, "
                    f"e.g. [H1, L1]."
                )
            if not dets:
                raise ConfigError(f"{path}.detectors: must not be empty.")
            dupes = {d for d in dets if dets.count(d) > 1}
            if dupes:
                raise ConfigError(
                    f"{path}.detectors: repeated detector(s) "
                    f"{', '.join(sorted(dupes))}."
                )

        return cls(run=run, detectors=dets)


@dataclass
class DataSection:
    """Data selection for each role in the run."""

    train: DataSelection
    validate: Optional[DataSelection] = None
    test: Optional[DataSelection] = None

    @classmethod
    def parse(cls, value: Any, path: str = "data") -> "DataSection":
        data = _require_mapping(value, path)
        _reject_unknown(data, cls, path)

        if "train" not in data:
            raise ConfigError(
                f"{path}: missing required section 'train' "
                f"(which observing run to train on)."
            )

        train = DataSelection.parse(data["train"], f"{path}.train")
        if train.detectors is None:
            raise ConfigError(
                f"{path}.train: 'detectors' is required here, e.g. "
                f"detectors: [H1, L1]. Other roles inherit it if omitted."
            )

        def _role(name: str) -> Optional[DataSelection]:
            if name not in data or data[name] is None:
                return None
            sel = DataSelection.parse(data[name], f"{path}.{name}")
            # Inherit the training detector set unless explicitly overridden.
            if sel.detectors is None:
                sel.detectors = list(train.detectors)
            return sel

        return cls(train=train, validate=_role("validate"), test=_role("test"))


# Prior groups a user may adjust, and the named sets available for each.
# Explicit ranges are also accepted; see PriorsSection.parse.
PRIOR_GROUPS: Dict[str, Dict[str, Any]] = {
    "masses": {
        "bounds": ("min", "max"),
        "sets": ("bbh_broad", "bbh_narrow", "bns", "nsbh"),
    },
    "spins": {
        "bounds": ("min", "max"),
        "sets": ("aligned_default", "precessing_default", "zero", "high"),
    },
    "distance": {
        "bounds": ("min", "max"),
        "sets": ("default", "near", "far"),
    },
}


@dataclass
class PriorsSection:
    """Adjustments to the waveform priors.

    Each group is either the name of a validated set (``bbh_broad``) or an
    explicit ``{min, max}`` range. Anything not named here comes from the
    prior file the preset points at - sky location, inclination, phase and
    polarisation are isotropic and essentially never varied.
    """

    masses: Optional[Union[str, Dict[str, float]]] = None
    spins: Optional[Union[str, Dict[str, float]]] = None
    distance: Optional[Union[str, Dict[str, float]]] = None

    @classmethod
    def parse(cls, value: Any, path: str = "priors") -> "PriorsSection":
        if value is None:
            return cls()
        data = _require_mapping(value, path)
        _reject_unknown(data, cls, path)

        parsed: Dict[str, Any] = {}
        for group, spec in PRIOR_GROUPS.items():
            if group not in data or data[group] is None:
                continue
            entry = data[group]
            gpath = f"{path}.{group}"

            if isinstance(entry, str):
                if entry not in spec["sets"]:
                    raise ConfigError(
                        f"{gpath}: unknown prior set {entry!r}"
                        f"{_suggest(entry, spec['sets'])}. "
                        f"Available: {', '.join(spec['sets'])}. "
                        f"You may also give an explicit range, "
                        f"e.g. {{min: 7.0, max: 50.0}}."
                    )
                parsed[group] = entry
                continue

            rng = _require_mapping(entry, gpath)
            lo_key, hi_key = spec["bounds"]
            missing = [k for k in (lo_key, hi_key) if k not in rng]
            if missing:
                raise ConfigError(
                    f"{gpath}: explicit range is missing "
                    f"{', '.join(repr(m) for m in missing)}. "
                    f"Give both, e.g. {{min: 7.0, max: 50.0}}, or name a set: "
                    f"{', '.join(spec['sets'])}."
                )
            extra = set(rng) - {lo_key, hi_key}
            if extra:
                raise ConfigError(
                    f"{gpath}: unexpected key(s) {', '.join(sorted(extra))}. "
                    f"An explicit range takes only {lo_key} and {hi_key}."
                )
            for k in (lo_key, hi_key):
                if not isinstance(rng[k], (int, float)) or isinstance(rng[k], bool):
                    raise ConfigError(f"{gpath}.{k}: expected a number.")
            if rng[lo_key] >= rng[hi_key]:
                raise ConfigError(
                    f"{gpath}: {lo_key} ({rng[lo_key]}) must be less than "
                    f"{hi_key} ({rng[hi_key]})."
                )
            parsed[group] = {lo_key: float(rng[lo_key]), hi_key: float(rng[hi_key])}

        return cls(**parsed)


@dataclass
class CustomSection:
    """Escape hatch: supply your own module for a stage.

    This exists so that one unusual requirement does not cost you the whole
    framework. A custom module replaces that stage only; data selection,
    staging, diagnostics and plotting continue to work normally.

    Prefer promoting a validated change into a new preset over using this.
    """

    train: Optional[str] = None
    config: Optional[str] = None
    priors: Optional[str] = None

    @classmethod
    def parse(cls, value: Any, path: str = "custom") -> "CustomSection":
        if value is None:
            return cls()
        data = _require_mapping(value, path)
        _reject_unknown(data, cls, path)
        for k, v in data.items():
            if not isinstance(v, str):
                raise ConfigError(
                    f"{path}.{k}: expected a path to a module or file, "
                    f"got {type(v).__name__}."
                )
        return cls(**data)


@dataclass
class RunSpec:
    """A complete, validated run specification.

    This is what a user's YAML parses into. It is intentionally small; see the
    module docstring for the reasoning.
    """

    preset: str
    name: str
    data: DataSection
    stages: List[str]
    priors: PriorsSection = field(default_factory=PriorsSection)
    custom: CustomSection = field(default_factory=CustomSection)
    export_dir: Optional[str] = None

    @classmethod
    def parse(cls, value: Any, path: str = "<run spec>") -> "RunSpec":
        data = _require_mapping(value, path)
        _reject_unknown(data, cls, path)

        for required in ("preset", "name", "data"):
            if required not in data:
                raise ConfigError(
                    f"{path}: missing required option {required!r}."
                )

        for key in ("preset", "name"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ConfigError(f"{path}.{key}: expected a non-empty string.")

        stages = data.get("stages")
        if stages is None:
            stages = ["train"]
        if not isinstance(stages, list) or not stages:
            raise ConfigError(
                f"{path}.stages: expected a non-empty list, "
                f"e.g. [train, benchmark, plots]."
            )
        for s in stages:
            if s not in KNOWN_STAGES:
                raise ConfigError(
                    f"{path}.stages: unknown stage {s!r}{_suggest(s, KNOWN_STAGES)}. "
                    f"Known stages: {', '.join(KNOWN_STAGES)}."
                )
        dupes = {s for s in stages if stages.count(s) > 1}
        if dupes:
            raise ConfigError(
                f"{path}.stages: repeated stage(s) {', '.join(sorted(dupes))}."
            )
        # Normalise to canonical execution order so the runner is deterministic
        # regardless of the order the user happened to list them in.
        stages = [s for s in KNOWN_STAGES if s in stages]

        export_dir = data.get("export_dir")
        if export_dir is not None and not isinstance(export_dir, str):
            raise ConfigError(f"{path}.export_dir: expected a path string.")

        return cls(
            preset=data["preset"],
            name=data["name"],
            data=DataSection.parse(data["data"], "data"),
            stages=stages,
            priors=PriorsSection.parse(data.get("priors")),
            custom=CustomSection.parse(data.get("custom")),
            export_dir=export_dir,
        )

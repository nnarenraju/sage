#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : loader.py
Description     : Load and validate a run-specification YAML.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan


Reads the small YAML described in :mod:`sage.config.schema` and returns a
validated :class:`~sage.config.schema.RunSpec`.

Errors are reported against the user's file - path, and key path within it -
rather than as a traceback through the parser, because the person hitting them
is usually setting up their first run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Union

from sage.config.schema import ConfigError, RunSpec

__all__ = ["load_run_spec", "loads_run_spec", "resolve_export_dir"]


def _read_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConfigError(
            "PyYAML is required to read run specifications. "
            "Install it with `pip install pyyaml`, or `pip install -e .` "
            "which pulls in all declared dependencies."
        ) from exc

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"Run specification not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from None

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # yaml's own message already carries line/column; keep it, but make
        # clear which file failed to parse.
        raise ConfigError(f"{path} is not valid YAML:\n{exc}") from None


def loads_run_spec(text: str, origin: str = "<string>") -> RunSpec:
    """Parse and validate a run specification from a YAML string."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to read run specifications.") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{origin} is not valid YAML:\n{exc}") from None

    if raw is None:
        raise ConfigError(f"{origin} is empty.")
    return RunSpec.parse(raw, origin)


def load_run_spec(path: Union[str, os.PathLike]) -> RunSpec:
    """Load, parse and validate a run specification from a YAML file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the run-specification YAML.

    Returns
    -------
    RunSpec
        The validated specification.

    Raises
    ------
    ConfigError
        If the file is missing, unparseable, or fails validation. The message
        names the file and the offending key path.
    """
    p = Path(path).expanduser()
    raw = _read_yaml(p)
    if raw is None:
        raise ConfigError(f"{p} is empty.")
    return RunSpec.parse(raw, str(p))


def resolve_export_dir(spec: RunSpec, root: Union[str, os.PathLike, None] = None) -> Path:
    """Work out where this run should write its outputs.

    Precedence: an explicit ``export_dir`` in the spec, then ``$SAGE_RUN_ROOT``,
    then ``root``, then the current directory. The run's ``name`` is appended
    unless ``export_dir`` was given explicitly, so two runs from the same root
    cannot silently share an output directory.

    Run outputs are large and must not land in a home directory that has a
    quota; prefer a scratch or work filesystem for the root.
    """
    if spec.export_dir:
        return Path(spec.export_dir).expanduser()

    base = os.environ.get("SAGE_RUN_ROOT") or root or Path.cwd()
    return Path(base).expanduser() / spec.name

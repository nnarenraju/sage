"""A search's settings, its directory in the run, and its data checks.

``search.yaml`` holds every setting; ``sage search-init`` writes it.
"""

from pathlib import Path

import yaml

from sage.config.loader import check, read_yaml, yaml_text
from sage.config.machine import (
    MissingData,
    bad_time_file,
    search_dataset_file,
    search_strain_file,
    segments_file,
    zero_runs_file,
)

SETTINGS_FILE = "search.yaml"
COMMENT = "# Edit before the first sage search; then these settings are fixed."

# Sources: stride, sgwc-1 (user ruling 2026-10-21); batch size, the
# measured best on an H100 (2026-08-21); background, user rulings
# 2026-11-13.
ROLL = {
    "name": str,
    "step_in_s": (float, 7200.0),
    "live_time_in_years": (float, 10.0),
    "years_per_file": (float, 1.0),
}
SCHEMA = {
    "observing_run": str,
    "checkpoint": str,
    "stride_in_samples": (int, 204),
    "batch_size": (int, 4096),
    "device": (str, "cuda"),
    "background": {"name": {"roll": ROLL}},
}


def search_dir(run_dir: Path, name: str) -> Path:
    """Directory of one search, ``<run_dir>/search/<name>``.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    name : str
        Search name.

    Returns
    -------
    Path
        The search directory.
    """
    return Path(run_dir) / "search" / name


def initial_settings(observing_run: str, checkpoint: str) -> dict:
    """
    Every setting of a new search, the defaults filled in.

    Parameters
    ----------
    observing_run : str
        Observing run to search, e.g. ``"o3a"``.
    checkpoint : str
        Stored checkpoint of the run, e.g. ``"ema_calibrated"``.

    Returns
    -------
    dict
        Checked settings.
    """
    values = {
        "observing_run": observing_run,
        "checkpoint": checkpoint,
        "background": {"name": "roll"},
    }
    return check(values, SCHEMA, SETTINGS_FILE)


def read_settings(directory: Path) -> dict:
    """
    Read and check a search's ``search.yaml``.

    Parameters
    ----------
    directory : Path
        Search directory.

    Returns
    -------
    dict
        Checked settings.

    Raises
    ------
    ValueError
        On an unknown, missing or invalid setting.
    """
    path = directory / SETTINGS_FILE
    return check(read_yaml(path), SCHEMA, str(path))


def write_settings(directory: Path, settings: dict) -> Path:
    """
    Write ``search.yaml`` into a new search directory.

    Parameters
    ----------
    directory : Path
        Search directory; created, and must not exist.
    settings : dict
        Checked settings.

    Returns
    -------
    Path
        The settings file.
    """
    directory.mkdir(parents=True)
    path = directory / SETTINGS_FILE
    path.write_text(f"{COMMENT}\n{yaml_text(settings)}")
    return path


def check_search_data(
    data_dir: Path, observing_run: str, run_settings: dict, training: dict
) -> None:
    """
    Raise if the search release lacks a file or differs from training.

    The release must hold its record, and strain, segment list, bad
    time and zero spans for every detector of the run, at the run's
    sample rate and the training high-pass, scale and dtype.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    observing_run : str
        Observing run to search, e.g. ``"o3a"``.
    run_settings : dict
        Resolved settings of the run.
    training : dict
        Fixed numbers of the training strain,
        ``sage.data.download.DATASET``.

    Raises
    ------
    MissingData
        If any file is missing; the message lists them all.
    ValueError
        If the release was conditioned differently from training.
    """
    record = search_dataset_file(data_dir, observing_run)
    files = [record]
    for detector in run_settings["data"]["detectors"]:
        strain = search_strain_file(data_dir, observing_run, detector)
        files += [strain, segments_file(strain), bad_time_file(strain)]
        files.append(zero_runs_file(strain))
    missing = [path for path in files if not path.exists()]
    if missing:
        listed = "\n".join(f"  {path}" for path in missing)
        raise MissingData(f"search data files are missing:\n{listed}")

    release = yaml.safe_load(record.read_text())
    expected = {
        "sample_rate_in_hz": run_settings["data"]["sample_rate_in_hz"],
        "highpass_in_hz": training["highpass_in_hz"],
        "scale": training["scale"],
        "dtype": training["dtype"],
    }
    for key, value in expected.items():
        if release[key] != value:
            raise ValueError(
                f"{record}: {key} is {release[key]}, training used {value}"
            )

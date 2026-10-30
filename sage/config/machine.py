"""The machine file and the data layout under its ``data_dir``.

``~/.config/sage/machine.yaml`` holds only ``data_dir``; every
observing run's data and every fiducial live at fixed places below it.
"""

from pathlib import Path

import yaml

from sage.config.loader import check, read_yaml

MACHINE_FILE = Path.home() / ".config" / "sage" / "machine.yaml"
SCHEMA = {"data_dir": str}


class MissingData(FileNotFoundError):
    """Files a run reads are missing from the data directory."""


def read_data_dir() -> Path:
    """
    Read ``data_dir`` from the machine file.

    Returns
    -------
    Path
        Directory holding all downloaded data.

    Raises
    ------
    FileNotFoundError
        If the machine file is missing or ``data_dir`` is not a directory.
    """
    if not MACHINE_FILE.exists():
        raise FileNotFoundError(f"{MACHINE_FILE} missing; run `sage machine-setup`")
    values = check(read_yaml(MACHINE_FILE), SCHEMA, str(MACHINE_FILE))

    data_dir = Path(values["data_dir"])
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{MACHINE_FILE}: data_dir {data_dir} is not a directory")
    return data_dir


def write_data_dir(data_dir: Path) -> None:
    """Write ``data_dir`` to the machine file, creating the file's directory."""
    MACHINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MACHINE_FILE.write_text(yaml.safe_dump({"data_dir": str(data_dir)}))


def observing_run_dir(data_dir: Path, run: str, part: str) -> Path:
    """
    Directory of one part of an observing run's data.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run, e.g. ``"O3b"``.
    part : str
        ``"training"``, ``"asds"`` or ``"search"``.

    Returns
    -------
    Path
        ``<data_dir>/<run>/<part>``, with the run in lower case.
    """
    return Path(data_dir) / run.lower() / part


def strain_file(data_dir: Path, run: str, detector: str) -> Path:
    """
    Training strain file of one detector in one observing run.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run, e.g. ``"o3b"``.
    detector : str
        Detector, e.g. ``"H1"``.

    Returns
    -------
    Path
        ``<data_dir>/<run>/training/data_<detector>_<Run>.bin``, the run
        written as GWOSC names it (``O3b``).
    """
    label = run[0].upper() + run[1:].lower()
    return observing_run_dir(data_dir, run, "training") / f"data_{detector}_{label}.bin"


def fiducial_dir(data_dir: Path, runs: list) -> Path:
    """
    Directory of the fiducial ASDs for one observing run or a pool.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    runs : list of str
        Observing runs the fiducial covers.

    Returns
    -------
    Path
        ``<data_dir>/<run>/fiducial`` for one run, else
        ``<data_dir>/fiducials/<run>+<run>...`` with the runs sorted.
    """
    names = sorted({run.lower() for run in runs})
    if len(names) == 1:
        return Path(data_dir) / names[0] / "fiducial"
    return Path(data_dir) / "fiducials" / "+".join(names)


def segments_file(path: Path) -> Path:
    """The ``_segments.json`` sidecar of a strain or per-segment ASD ``.bin`` file."""
    path = Path(path)
    return path.parent / f"{path.stem}_segments.json"


def failed_file(path: Path) -> Path:
    """The ``_failed_segments.json`` list of pieces a strain file's download could not fetch."""
    path = Path(path)
    return path.parent / f"{path.stem}_failed_segments.json"


def bad_time_file(path: Path) -> Path:
    """The ``_bad_time.json`` data-quality time of a strain ``.bin`` file."""
    path = Path(path)
    return path.parent / f"{path.stem}_bad_time.json"


def zero_runs_file(path: Path) -> Path:
    """The ``_zero_runs.json`` spans of exactly-zero samples in a strain ``.bin`` file."""
    path = Path(path)
    return path.parent / f"{path.stem}_zero_runs.json"


def dataset_file(data_dir: Path, run: str) -> Path:
    """Record of an observing run's strain, ``<data_dir>/<run>/training/dataset.yaml``."""
    return observing_run_dir(data_dir, run, "training") / "dataset.yaml"


def asds_file(data_dir: Path, run: str) -> Path:
    """Record of an observing run's ASDs, ``<data_dir>/<run>/asds/asds.yaml``."""
    return observing_run_dir(data_dir, run, "asds") / "asds.yaml"


def segment_asd_file(asd_dir: Path, detector: str) -> Path:
    """Per-segment ASDs of one detector, ``<asd_dir>/segment_psds/data_<detector>_psds.bin``."""
    return Path(asd_dir) / "segment_psds" / f"data_{detector}_psds.bin"


def recolour_bank_file(asd_dir: Path, detector: str) -> Path:
    """Recolour ASD bank of one detector, ``<asd_dir>/recolour_psds/raw_<detector>_psds.bin``; ``.json`` beside it."""
    return Path(asd_dir) / "recolour_psds" / f"raw_{detector}_psds.bin"


def fiducial_file(fiducial_dir: Path, detector: str) -> Path:
    """Fiducial ASD of one detector, ``<fiducial_dir>/fiducial_<detector>_psd.bin``; ``.json`` beside it."""
    return Path(fiducial_dir) / f"fiducial_{detector}_psd.bin"

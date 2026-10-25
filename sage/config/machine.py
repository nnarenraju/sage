"""The machine file and the data layout under its ``data_dir``.

``~/.config/sage/machine.yaml`` holds only ``data_dir``; every
observing run's data and every fiducial live at fixed places below it.
"""

from pathlib import Path

from sage.config.loader import check, read_yaml

MACHINE_FILE = Path.home() / ".config" / "sage" / "machine.yaml"
SCHEMA = {"data_dir": str}


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

"""
The machine file and the data layout under its ``data_dir``.

``~/.config/sage/machine.yaml`` holds only ``data_dir``; every
observing run's data and every fiducial live at fixed places below it.
"""

from pathlib import Path

from sage.config.loader import check, read_yaml, yaml_text

MACHINE_FILE = Path.home() / ".config" / "sage" / "machine.yaml"
SCHEMA = {"data_dir": str}
COMMENT = "# data_dir: the directory that holds all data Sage downloads and builds."


class MissingData(FileNotFoundError):
    """Files a run reads are missing from the data directory."""


class MissingASDs(MissingData):
    """ASD files, built by ``sage build-asds``, are missing."""


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
        If the machine file is missing or ``data_dir`` is not a
        directory.
    ValueError
        If ``data_dir`` is missing or empty, or the file holds another
        key.
    """
    if not MACHINE_FILE.exists():
        raise FileNotFoundError(f"{MACHINE_FILE} missing; run `sage setup`")
    values = read_yaml(MACHINE_FILE)
    if "data_dir" in values and not values["data_dir"]:
        raise ValueError(f"data_dir in {MACHINE_FILE} is empty; fill it in")
    values = check(values, SCHEMA, str(MACHINE_FILE))

    data_dir = Path(values["data_dir"])
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{MACHINE_FILE}: data_dir {data_dir} is not a directory")
    return data_dir


def write_machine_file(data_dir: Path | None) -> None:
    """
    Write the machine file, creating its directory.

    Parameters
    ----------
    data_dir : Path or None
        Directory for all data; None leaves ``data_dir`` empty.
    """
    MACHINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if data_dir is None:
        value = "data_dir:\n"
    else:
        value = yaml_text({"data_dir": str(data_dir)})
    MACHINE_FILE.write_text(f"{COMMENT}\n{value}")


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
        ``"training"`` or ``"search"``.

    Returns
    -------
    Path
        ``<data_dir>/<run>/<part>``, with the run in lower case.
    """
    return Path(data_dir) / run.lower() / part


def gwosc_label(run: str) -> str:
    """
    An observing run as GWOSC writes it in file names, e.g. ``O3b``.

    Parameters
    ----------
    run : str
        Observing run, in any case.

    Returns
    -------
    str
    """
    return run[0].upper() + run[1:].lower()


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
    return observing_run_dir(data_dir, run, "training") / f"data_{detector}_{gwosc_label(run)}.bin"


def search_strain_file(data_dir: Path, run: str, detector: str) -> Path:
    """
    Search strain file of one detector in one observing run.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run, e.g. ``"o3a"``.
    detector : str
        Detector, e.g. ``"H1"``.

    Returns
    -------
    Path
        ``<data_dir>/<run>/search/data_<detector>_<Run>.h5``.
    """
    label = gwosc_label(run)
    return observing_run_dir(data_dir, run, "search") / f"data_{detector}_{label}.h5"


def asd_dir(data_dir: Path, run: str) -> Path:
    """
    Directory of the ASDs built from one observing run's strain.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run.

    Returns
    -------
    Path
        ``<data_dir>/asds/<run>``, with the run in lower case.
    """
    return Path(data_dir) / "asds" / run.lower()


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
        ``<data_dir>/asds/<runs>/fiducial``, where ``<runs>`` is the
        sorted, lower-case run names joined by ``+``.
    """
    return Path(data_dir) / "asds" / "+".join(sorted({run.lower() for run in runs})) / "fiducial"


def segments_file(path: Path) -> Path:
    """
    The ``_segments.json`` sidecar of a strain or per-segment ASD file.

    Parameters
    ----------
    path : Path
        The ``.bin`` file.

    Returns
    -------
    Path
    """
    path = Path(path)
    return path.parent / f"{path.stem}_segments.json"


def failed_file(path: Path) -> Path:
    """
    The ``_failed_segments.json`` list of pieces a download missed.

    Parameters
    ----------
    path : Path
        The strain file.

    Returns
    -------
    Path
    """
    path = Path(path)
    return path.parent / f"{path.stem}_failed_segments.json"


def bad_time_file(path: Path) -> Path:
    """
    The ``_bad_time.json`` data-quality time of a strain file.

    Parameters
    ----------
    path : Path
        The strain ``.bin`` file.

    Returns
    -------
    Path
    """
    path = Path(path)
    return path.parent / f"{path.stem}_bad_time.json"


def zero_runs_file(path: Path) -> Path:
    """
    The ``_zero_runs.json`` spans of exactly-zero strain samples.

    Parameters
    ----------
    path : Path
        The strain ``.bin`` file.

    Returns
    -------
    Path
    """
    path = Path(path)
    return path.parent / f"{path.stem}_zero_runs.json"


def dataset_file(data_dir: Path, run: str) -> Path:
    """
    Record of an observing run's training strain.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run.

    Returns
    -------
    Path
        ``<data_dir>/<run>/training/dataset.yaml``.
    """
    return observing_run_dir(data_dir, run, "training") / "dataset.yaml"


def search_dataset_file(data_dir: Path, run: str) -> Path:
    """
    Record of an observing run's search strain.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run.

    Returns
    -------
    Path
        ``<data_dir>/<run>/search/dataset.yaml``.
    """
    return observing_run_dir(data_dir, run, "search") / "dataset.yaml"


def asds_file(data_dir: Path, run: str) -> Path:
    """
    Record of an observing run's ASDs.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run.

    Returns
    -------
    Path
        ``<data_dir>/asds/<run>/asds.yaml``.
    """
    return asd_dir(data_dir, run) / "asds.yaml"


def segment_asd_file(asd_dir: Path, detector: str) -> Path:
    """
    Per-segment ASD file of one detector.

    Parameters
    ----------
    asd_dir : Path
        ASD directory of an observing run.
    detector : str
        Detector, e.g. ``"H1"``.

    Returns
    -------
    Path
        ``<asd_dir>/segment_psds/data_<detector>_psds.bin``.
    """
    return Path(asd_dir) / "segment_psds" / f"data_{detector}_psds.bin"


def recolour_bank_file(asd_dir: Path, detector: str) -> Path:
    """
    Recolour ASD bank of one detector; its ``.json`` sits beside it.

    Parameters
    ----------
    asd_dir : Path
        ASD directory of an observing run.
    detector : str
        Detector, e.g. ``"H1"``.

    Returns
    -------
    Path
        ``<asd_dir>/recolour_psds/raw_<detector>_psds.bin``.
    """
    return Path(asd_dir) / "recolour_psds" / f"raw_{detector}_psds.bin"


def fiducial_file(fiducial_dir: Path, detector: str) -> Path:
    """
    Fiducial ASD of one detector; its ``.json`` sits beside it.

    Parameters
    ----------
    fiducial_dir : Path
        Fiducial directory, from :func:`fiducial_dir`.
    detector : str
        Detector, e.g. ``"H1"``.

    Returns
    -------
    Path
        ``<fiducial_dir>/fiducial_<detector>_psd.bin``.
    """
    return Path(fiducial_dir) / f"fiducial_{detector}_psd.bin"

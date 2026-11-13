"""
MLGWSC-1's own code, every team's dataset-4 results and dataset 4,
made once in ``<data_dir>/mlgwsc1`` and read by every benchmark.

Each piece is built in a ``.part`` folder, then moved into place.
"""

import hashlib
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import h5py

from sage.benchmark.mlgwsc1.evaluator import find_injection_times
from sage.benchmark.mlgwsc1.snr import injection_snrs
from sage.core.logger import get_logger

logger = get_logger(__name__)

ZENODO = "https://zenodo.org/api/records/7107410/files"  # MLGWSC-1 release v1.4
CODE = ("gwastro/ml-mock-data-challenge-1-v1.4.zip", "ffd96432b8f7202b69616e6770723b77")
RESULTS = ("results.zip", "d27379a1d8136e9dee1ff9ab310221d3")
CODE_DIR = "gwastro-ml-mock-data-challenge-1-6d5e62f"  # the code archive's folder
# Dataset 4 challenge set; one month.
SEED, START_OFFSET, DURATION = 2514409456, 0, 2592000
MARGINS = (30, 30)  # evaluate.py: no injections in a segment's first and last 30 s
GRAVITYSPY = "https://zenodo.org/api/records/5649212/files"  # Gravity Spy v1.0.0
GRAVITYSPY_MD5 = {
    "H1_O3a.csv": "29aea278b622cd97496971f7c07f7d6a",
    "L1_O3a.csv": "3d3d0409817499c84fb2e0a2b211b5a5",
    "H1_O3b.csv": "590290fe3c7ddf8cd9fa85c3a09cd5e4",
    "L1_O3b.csv": "90d0bc16448bca72818ed9abf267979c",
}


def fetch(url: str, path: Path, md5: str) -> None:
    """
    Download ``url`` to ``path`` through ``<path>.part``.

    Parameters
    ----------
    url : str
        File to download.
    path : Path
        Where it goes.
    md5 : str
        Its published MD5 checksum.

    Raises
    ------
    ValueError
        If the download does not match ``md5``.
    """
    part = path.with_name(path.name + ".part")
    digest = hashlib.md5()
    with urllib.request.urlopen(url) as response, open(part, "wb") as file:
        while chunk := response.read(1 << 24):
            digest.update(chunk)
            file.write(chunk)
    if digest.hexdigest() != md5:
        part.unlink()
        raise ValueError(f"{url}: MD5 {digest.hexdigest()}, published {md5}")
    part.rename(path)


def fetch_code(directory: Path) -> Path:
    """
    MLGWSC-1's code (``generate_data.py``, ``evaluate.py``, PSDs).

    Parameters
    ----------
    directory : Path
        The MLGWSC-1 folder of the data directory.

    Returns
    -------
    Path
        The code's folder.

    Raises
    ------
    ValueError
        If the download does not match its published MD5.
    """
    code = directory / CODE_DIR
    if code.exists():
        return code
    archive = directory / "code.zip"
    fetch(f"{ZENODO}/{CODE[0]}/content", archive, CODE[1])
    part = directory / "code.part"
    with zipfile.ZipFile(archive) as file:
        file.extractall(part)
    (part / CODE_DIR).rename(code)
    part.rmdir()
    archive.unlink()
    return code


def fetch_results(directory: Path) -> Path:
    """
    Every team's dataset-4 results, ``<team>/ds4``.

    Parameters
    ----------
    directory : Path
        The MLGWSC-1 folder of the data directory.

    Returns
    -------
    Path
        The results folder.

    Raises
    ------
    ValueError
        If the download does not match its published MD5.
    """
    results = directory / "results"
    if results.exists():
        return results
    archive = directory / "results.zip"
    fetch(f"{ZENODO}/{RESULTS[0]}/content", archive, RESULTS[1])
    part = directory / "results.part"
    with zipfile.ZipFile(archive) as file:
        members = [name for name in file.namelist() if "/ds4/" in name]
        file.extractall(part, members=members)
    (part / "results").rename(results)
    part.rmdir()
    archive.unlink()
    return results


def fetch_gravityspy(directory: Path, name: str) -> Path:
    """
    One Gravity Spy glitch file, ``gravityspy/<name>``.

    Parameters
    ----------
    directory : Path
        The MLGWSC-1 folder of the data directory.
    name : str
        A key of ``GRAVITYSPY_MD5``, e.g. ``H1_O3a.csv``.

    Returns
    -------
    Path
        The file.

    Raises
    ------
    ValueError
        If the download does not match its published MD5.
    """
    folder = directory / "gravityspy"
    folder.mkdir(exist_ok=True)
    path = folder / name
    if not path.exists():
        fetch(f"{GRAVITYSPY}/{name}/content", path, GRAVITYSPY_MD5[name])
    return path


def make_dataset(directory: Path, code: Path) -> Path:
    """
    Dataset 4's challenge set, by MLGWSC-1's ``generate_data.py``.

    The script runs from its own folder, where it keeps the real noise
    it downloads.

    Parameters
    ----------
    directory : Path
        The MLGWSC-1 folder of the data directory.
    code : Path
        MLGWSC-1's code (:func:`fetch_code`).

    Returns
    -------
    Path
        The ``ds4`` folder: injection, foreground and background files.

    Raises
    ------
    subprocess.CalledProcessError
        If ``generate_data.py`` fails.
    """
    dataset = directory / "ds4"
    if dataset.exists():
        return dataset
    part = directory / "ds4.part"
    part.mkdir(exist_ok=True)
    # generate_data.py calls pycbc_create_injections, beside this Python
    path = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    environment = {**os.environ, "PATH": path}
    subprocess.run(
        [sys.executable, "generate_data.py", "--data-set", "4",
         "--output-injection-file", str(part / "injections.hdf"),
         "--output-foreground-file", str(part / "foreground.hdf"),
         "--output-background-file", str(part / "background.hdf"),
         "--seed", str(SEED), "--start-offset", str(START_OFFSET),
         "--duration", str(DURATION), "--verbose", "--force"],
        cwd=code, env=environment, check=True,
    )
    part.rename(dataset)
    return dataset


def make_snrs(dataset: Path, code: Path, workers: int) -> Path:
    """
    The injections' SNRs (``snr.hdf``, dataset ``snr``) beside them.

    Only injections the evaluation counts (``find_injection_times`` with
    its 30 s margins) are measured; the others are NaN.

    Parameters
    ----------
    dataset : Path
        The ``ds4`` folder (:func:`make_dataset`).
    code : Path
        MLGWSC-1's code, for its PSDs.
    workers : int
        Worker processes.

    Returns
    -------
    Path
        ``snr.hdf``.
    """
    path = dataset / "snr.hdf"
    if path.exists():
        return path
    injections = dataset / "injections.hdf"
    foreground = [str(dataset / "foreground.hdf")]
    inside = find_injection_times(foreground, str(injections), *MARGINS)[1]
    snrs = injection_snrs(injections, code / "psds", inside, workers)
    part = path.with_name(path.name + ".part")
    with h5py.File(part, "w") as file:
        file.create_dataset("snr", data=snrs)
    part.rename(path)
    return path

"""Score streams of a search: float32 rows written batch by batch.

A stream is ``<name>.bin``, ``(rows, columns)`` float32 values created
at full size, and ``<name>.json``: its columns, settings and windows
done. A resumed stream rewrites everything after ``done``.
"""

import json
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np

from sage.config.run_dir import SettingsChanged, differences


def stream_files(directory: Path, name: str) -> tuple:
    """The ``.bin`` rows and ``.json`` progress of one stream.

    Parameters
    ----------
    directory : Path
        Directory of the stream.
    name : str
        Stream name.

    Returns
    -------
    tuple of Path
        Paths of the ``.bin`` and ``.json`` files.
    """
    return directory / f"{name}.bin", directory / f"{name}.json"


def write_progress(path: Path, progress: dict) -> None:
    """Replace a progress file in one step.

    Parameters
    ----------
    path : Path
        Progress file.
    progress : dict
        Contents, JSON types only.
    """
    part = path.with_name(f"{path.name}.part")
    part.write_text(json.dumps(progress, indent=1))
    os.replace(part, path)


def start_stream(
    directory: Path, name: str, columns: list, rows: int, attrs: dict
) -> int:
    """
    Windows done in a stream; a new stream is created at full size.

    Parameters
    ----------
    directory : Path
        Directory of the stream.
    name : str
        Stream name.
    columns : list of str
        Column names, one float32 value each.
    rows : int
        Rows of the stream.
    attrs : dict
        Settings the stream is written with, JSON types only.

    Returns
    -------
    int
        Windows done.

    Raises
    ------
    SettingsChanged
        If the stream was started with other columns, rows or settings.
    """
    data, progress = stream_files(directory, name)
    started = {"columns": columns, "rows": rows, "attrs": attrs}
    if not progress.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with open(data, "wb") as handle:
            handle.truncate(rows * len(columns) * 4)
        write_progress(progress, {**started, "done": 0})
        return 0

    recorded = json.loads(progress.read_text())
    done = recorded.pop("done")
    changes = differences(recorded, started)
    if changes:
        listed = "\n".join(f"  {change}" for change in changes)
        raise SettingsChanged(f"{progress} was started with other settings:\n{listed}")
    return done


def read_stream(directory: Path, name: str) -> tuple:
    """
    Rows and progress of a stream, for reading while it is written.

    Parameters
    ----------
    directory : Path
        Directory of the stream.
    name : str
        Stream name.

    Returns
    -------
    rows : numpy.memmap
        Float32 ``(rows, columns)``, read only; rows of windows past
        ``done`` are not finished.
    progress : dict
        Columns, rows, settings and windows done.
    """
    data, progress = stream_files(directory, name)
    recorded = json.loads(progress.read_text())
    shape = (recorded["rows"], len(recorded["columns"]))
    return np.memmap(data, dtype="<f4", mode="r", shape=shape), recorded


def write_rows(handle: BinaryIO, first_row: int, rows: np.ndarray) -> None:
    """
    Write float32 rows into an open stream, starting at ``first_row``.

    Parameters
    ----------
    handle : BinaryIO
        The stream's ``.bin``, opened ``"r+b"``.
    first_row : int
        Row of the first value.
    rows : numpy.ndarray
        Float32 ``(rows, columns)``.
    """
    values = np.ascontiguousarray(rows, dtype="<f4")
    os.pwrite(handle.fileno(), values.tobytes(), first_row * values.shape[1] * 4)


def record_done(handle: BinaryIO, directory: Path, name: str, done: int) -> None:
    """
    Make the written rows durable, then record the windows done.

    Parameters
    ----------
    handle : BinaryIO
        The stream's ``.bin``.
    directory : Path
        Directory of the stream.
    name : str
        Stream name.
    done : int
        Windows done.
    """
    os.fsync(handle.fileno())
    progress = stream_files(directory, name)[1]
    recorded = json.loads(progress.read_text())
    recorded["done"] = done
    write_progress(progress, recorded)

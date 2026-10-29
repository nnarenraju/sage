"""sage.utils.atomic_io.append_h5: a damaged file is moved aside, a file held by another process is not."""

import subprocess
import sys

import h5py
import pytest

from sage.utils.atomic_io import append_h5


def test_append_h5(tmp_path):
    path = tmp_path / "results.h5"
    with append_h5(path) as file:
        file["x"] = [1, 2, 3]
    holder = subprocess.Popen(
        [sys.executable, "-c", f"import h5py, sys; f = h5py.File({str(path)!r}, 'a'); print(flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        holder.stdout.readline()
        with pytest.raises(BlockingIOError, match="open in another process"):
            with append_h5(path):
                pass
    finally:
        holder.communicate("")
    assert not (tmp_path / "results.h5.corrupt").exists()
    with h5py.File(path, "r") as file:
        assert list(file["x"][:]) == [1, 2, 3]

    path.write_bytes(path.read_bytes()[:600])  # truncated, as by a kill mid-write
    with append_h5(path) as file:
        assert list(file) == []
    assert (tmp_path / "results.h5.corrupt").exists()

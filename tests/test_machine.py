"""Tests for sage.config.machine: machine file and fiducial paths."""

from pathlib import Path

import pytest

from sage.config import machine


@pytest.fixture
def machine_file(tmp_path, monkeypatch):
    """Machine-file path in a temporary directory."""
    path = tmp_path / "machine.yaml"
    monkeypatch.setattr(machine, "MACHINE_FILE", path)
    return path


def test_missing(machine_file):
    """A missing machine file points to sage setup."""
    with pytest.raises(FileNotFoundError, match="run `sage setup`"):
        machine.read_data_dir()


def test_unknown_key(machine_file, tmp_path):
    """An unknown key is refused with the closest one."""
    machine_file.write_text(f"data_dir: {tmp_path}\ndatadir: {tmp_path}\n")
    with pytest.raises(ValueError, match="did you mean 'data_dir'"):
        machine.read_data_dir()


def test_data_dir_absent(machine_file, tmp_path):
    """A data_dir that does not exist is refused."""
    machine_file.write_text(f"data_dir: {tmp_path / 'absent'}\n")
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        machine.read_data_dir()


def test_fiducial_dir():
    """Fiducial directories are lower-case, sorted and deduplicated."""
    root = Path("/data")
    assert machine.fiducial_dir(root, ["O3b"]) == root / "asds" / "o3b" / "fiducial"
    pooled = machine.fiducial_dir(root, ["O3b", "o3a", "O3a"])
    assert pooled == root / "asds" / "o3a+o3b" / "fiducial"

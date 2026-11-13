"""Settings, data checks and initialisation of a search."""

import pytest
import yaml

from sage.cli import main
from sage.config import machine
from sage.config.loader import check, read_yaml
from sage.data.download import DATASET, SEARCH_DATASET
from sage.search.settings import (
    SCHEMA,
    check_search_data,
    initial_settings,
    write_settings,
)

RUN = {"data": {"detectors": ["H1"], "sample_rate_in_hz": 2048.0}}


def test_settings_round_trip(tmp_path):
    """search.yaml reads back as written, defaults included."""
    settings = initial_settings("o3a", "best")
    path = write_settings(tmp_path / "o3a_best", settings)
    assert check(read_yaml(path), SCHEMA, "search.yaml") == settings
    assert settings["background"]["step_in_s"] == 7200.0


def test_check_search_data(tmp_path):
    """Missing files and a release conditioned otherwise are refused."""
    with pytest.raises(machine.MissingData, match="data_H1_O3a.h5"):
        check_search_data(tmp_path, "o3a", RUN, DATASET)
    strain = machine.search_strain_file(tmp_path, "o3a", "H1")
    strain.parent.mkdir(parents=True)
    for path in (strain, machine.segments_file(strain), machine.bad_time_file(strain)):
        path.touch()
    machine.zero_runs_file(strain).touch()
    record = machine.search_dataset_file(tmp_path, "o3a")
    record.write_text(yaml.safe_dump({**SEARCH_DATASET, "highpass_in_hz": 10.0}))
    with pytest.raises(ValueError, match="highpass_in_hz is 10.0"):
        check_search_data(tmp_path, "o3a", RUN, DATASET)
    record.write_text(yaml.safe_dump(SEARCH_DATASET))
    check_search_data(tmp_path, "o3a", RUN, DATASET)


def test_unknown_checkpoint(tmp_path, monkeypatch, capsys):
    """search-init names the closest stored checkpoint."""
    (tmp_path / "run.yaml").write_text("extends: hl\n")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "best.pt").touch()
    monkeypatch.chdir(tmp_path)
    assert main(["search-init", "o3a", "--checkpoint", "bets"]) == 1
    error = capsys.readouterr().err
    assert error.endswith("fix: sage search-init o3a --checkpoint best\n")

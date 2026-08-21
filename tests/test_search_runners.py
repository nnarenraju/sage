#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_runners.py
Description   : The command-line entry points a campaign is launched through.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

``submit.sh`` invokes these, so a defect here is a failure on a compute node after a
scheduling round trip. Everything reachable without touching data is checked from the
login node instead: argument handling, the plan, and the refusals.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _runner(name):
    """Import one of the runner scripts by path, as its shebang line would."""
    path = REPO_ROOT / "runs" / "search" / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{path} not present in this checkout")
    spec = importlib.util.spec_from_file_location(f"_runner_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONFIG = str(REPO_ROOT / "runs" / "search" / "config_o3a_HL.py")


def _have_config():
    return Path(CONFIG).is_file()


@pytest.fixture
def fresh_campaign(monkeypatch, tmp_path):
    """
    Point the campaign at an empty directory for the duration of a test.

    The plan a driver prints is a function of the specification *and* of what the
    campaign has already recorded complete. Read from the live directory, these tests
    pass on a fresh checkout and fail the moment anyone runs a stage for real -- which is
    exactly backwards, since running stages is the point. Redirecting ``out_dir`` tests
    the planner rather than the state of whatever campaign happens to be on disk.
    """
    import dataclasses

    from sage.search import spec as spec_module

    original = spec_module.load_spec

    def _relocated(config_module):
        loaded = original(config_module)
        return dataclasses.replace(loaded, out_dir=tmp_path / loaded.tag)

    monkeypatch.setattr(spec_module, "load_spec", _relocated)
    return tmp_path


class TestRunStageArguments:
    """What the single-stage driver accepts, and what it refuses."""

    def test_config_and_stage_required(self):
        """A campaign cannot be inferred; both must be stated."""
        runner = _runner("run_stage")
        with pytest.raises(SystemExit):
            runner.parse_args([])
        with pytest.raises(SystemExit):
            runner.parse_args(["--config", "x"])

    def test_slide_is_repeatable(self):
        """
        An array task owns several slides, so the flag accumulates.

        A single-valued flag would make each task overwrite the previous one's request and
        the ladder would be scored with gaps.
        """
        runner = _runner("run_stage")
        args = runner.parse_args(
            ["--config", "c", "--stage", "background", "--slide", "3", "--slide", "7"]
        )

        assert args.slide == [3, 7]

    def test_defaults_are_the_safe_ones(self):
        """
        Cascade on, force off. Both defaults err towards rebuilding rather than reusing.

        The opposite defaults would let a stale product survive a re-run silently, which is
        the failure the fingerprint contract exists to prevent.
        """
        runner = _runner("run_stage")
        args = runner.parse_args(["--config", "c", "--stage", "grid"])

        assert args.force is False
        assert args.no_cascade is False
        assert args.track == "core"

    @pytest.mark.skipif(not _have_config(), reason="campaign config not present")
    def test_unknown_stage_refused(self):
        """A misspelled stage is refused before anything is scheduled."""
        runner = _runner("run_stage")
        with pytest.raises(KeyError):
            runner.main(["--config", CONFIG, "--stage", "zerolagg", "--dry-run"])

    @pytest.mark.skipif(not _have_config(), reason="campaign config not present")
    def test_dry_run_plans_without_running(self, capsys, fresh_campaign):
        """
        The plan is printed and nothing is executed.

        A dry run exists to fail before a scheduler has queued anything, so it validates
        the inputs too -- an absent release is exactly the failure worth having early.
        """
        runner = _runner("run_stage")
        code = runner.main(["--config", CONFIG, "--stage", "all", "--dry-run"])
        out = capsys.readouterr().out

        assert code == 0
        assert "would run segments" in out
        assert "spec hash" in out


class TestRunSearchPlan:
    """The whole-campaign driver."""

    @pytest.mark.skipif(not _have_config(), reason="campaign config not present")
    def test_stop_after_truncates_the_plan(self, capsys, fresh_campaign):
        """
        ``--stop-after`` keeps everything its target depends on and nothing beyond it.

        The dependencies are not optional: stopping after `far` still requires the five
        stages that produce what `far` reads.
        """
        runner = _runner("run_search")
        runner.main(["--config", CONFIG, "--stop-after", "far", "--dry-run"])
        out = capsys.readouterr().out

        assert "segments -> grid -> zerolag -> slides -> background -> far" in out
        assert "injections" not in out

    @pytest.mark.skipif(not _have_config(), reason="campaign config not present")
    def test_n_slides_changes_the_campaign_identity(self):
        """
        A shallow smoke ladder is a different campaign, and its hash says so.

        Sharing a hash with the production run would let a smoke background be reused as
        though it were the deep one -- 8 slides of livetime reported as 82.
        """
        runner = _runner("run_search")
        shallow = runner.build_spec(
            runner.parse_args(["--config", CONFIG, "--n-slides", "8"])
        )
        deep = runner.build_spec(runner.parse_args(["--config", CONFIG]))

        assert shallow.slides.n_slides == 8
        assert shallow.hash() != deep.hash()

    @pytest.mark.skipif(not _have_config(), reason="campaign config not present")
    def test_plan_reports_the_campaign_identity(self, capsys):
        """A job log must say which network searched which release, and under what hash."""
        runner = _runner("run_search")
        runner.main(["--config", CONFIG, "--dry-run"])
        out = capsys.readouterr().out

        for field in ("out_dir", "release", "network", "spec hash"):
            assert field in out

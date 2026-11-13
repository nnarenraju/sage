"""Tests for sage.core.logger: importing sage creates no files or
handlers; setup_logging and format_duration."""

import logging
import subprocess
import sys

import pytest

from sage.core.logger import (
    format_duration,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _restore_root_handlers():
    """Restore the root logger's handlers and level after each test."""
    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)
    root.setLevel(saved_level)


class TestGetLoggerIsInert:
    """get_logger and importing sage leave the filesystem alone."""

    def test_creates_no_files_or_directories(self, tmp_path, monkeypatch):
        """Logging through get_logger creates no files."""
        monkeypatch.chdir(tmp_path)
        get_logger("sage.test.y").info("this must not create anything")
        assert list(tmp_path.iterdir()) == []

    def test_installs_no_handlers(self):
        """get_logger installs no handlers."""
        log = get_logger("sage.test.z")
        assert log.handlers == []

    def test_importing_sage_modules_creates_nothing(self, tmp_path):
        """A fresh import of sage creates no files or root handlers."""
        code = (
            "import sage.dsp.utils, sage.dsp.filters, "
            "sage.core.errors, sage.core.decorators, "
            "sage.factory.training, sage.factory.validation; "
            "import logging; "
            "print(len(logging.getLogger().handlers))"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert res.returncode == 0, res.stderr
        assert list(tmp_path.iterdir()) == [], "import created files in the cwd"
        assert res.stdout.strip() == "0", "import installed root handlers"


class TestSetupLogging:
    """Tests of setup_logging."""

    def test_writes_into_export_dir_logs(self, tmp_path):
        """The log file is <export_dir>/logs/run.log."""
        path = setup_logging(tmp_path / "run")
        assert path == tmp_path / "run" / "logs" / "run.log"
        get_logger("sage.test.a").info("hello")
        assert "hello" in path.read_text()

    def test_no_export_dir_means_console_only(self, tmp_path, monkeypatch):
        """Without an export directory, nothing is written to disk."""
        monkeypatch.chdir(tmp_path)
        assert setup_logging() is None
        assert list(tmp_path.iterdir()) == []

    def test_file_keeps_debug_even_when_console_is_info(self, tmp_path):
        """The file keeps DEBUG records when the console is at INFO."""
        path = setup_logging(tmp_path / "run", level=logging.INFO)
        get_logger("sage.test.b").debug("quiet detail")
        # Turning the console down must not lose detail from the file.
        assert "quiet detail" in path.read_text()

    def test_other_libraries_log_warnings_only(self, tmp_path):
        """Other libraries reach the file only at WARNING and above."""
        path = setup_logging(tmp_path / "run")
        other = logging.getLogger("numba.core.ssa")
        other.debug("compiler detail")
        other.info("library chatter")
        other.warning("library warning")
        text = path.read_text()
        assert "compiler detail" not in text and "library chatter" not in text
        assert "library warning" in text

    def test_repeated_calls_do_not_duplicate_output(self, tmp_path):
        """Repeated setup does not duplicate records."""
        path = setup_logging(tmp_path / "run")
        setup_logging(tmp_path / "run")
        setup_logging(tmp_path / "run")
        get_logger("sage.test.c").info("once please")
        assert path.read_text().count("once please") == 1

    def test_respects_sage_log_level(self, tmp_path, monkeypatch, capsys):
        """SAGE_LOG_LEVEL sets the console level."""
        monkeypatch.setenv("SAGE_LOG_LEVEL", "DEBUG")
        setup_logging(tmp_path / "run")
        get_logger("sage.test.d").debug("visible in debug mode")
        assert "visible in debug mode" in capsys.readouterr().out

    def test_default_console_level_hides_debug(self, tmp_path, monkeypatch, capsys):
        """The default console level hides DEBUG records."""
        monkeypatch.delenv("SAGE_LOG_LEVEL", raising=False)
        setup_logging(tmp_path / "run")
        get_logger("sage.test.e").debug("should stay off the console")
        assert "should stay off the console" not in capsys.readouterr().out

    def test_invalid_level_is_rejected_not_ignored(self, tmp_path, monkeypatch):
        """An invalid SAGE_LOG_LEVEL is refused."""
        monkeypatch.setenv("SAGE_LOG_LEVEL", "LOUD")
        with pytest.raises(ValueError, match="SAGE_LOG_LEVEL"):
            setup_logging(tmp_path / "run")

    def test_console_can_be_disabled(self, tmp_path, capsys):
        """console=False writes to the file only."""
        setup_logging(tmp_path / "run", console=False)
        get_logger("sage.test.f").info("file only")
        assert "file only" not in capsys.readouterr().out

    def test_appends_rather_than_truncating_on_resume(self, tmp_path):
        """A second setup appends to the existing log file."""
        path = setup_logging(tmp_path / "run")
        get_logger("sage.test.g").info("first segment")
        setup_logging(tmp_path / "run")          # simulate a resumed segment
        get_logger("sage.test.g").info("second segment")
        text = path.read_text()
        assert "first segment" in text and "second segment" in text


class TestFormatDuration:
    """Tests of format_duration."""

    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (0, "0s"),
            (18, "18s"),
            (59, "59s"),
            (60, "1m 00s"),
            (2352, "39m 12s"),
            (3600, "1h 00m"),
            (7440, "2h 04m"),
            (-5, "0s"),
        ],
    )
    def test_rendering(self, seconds, expected):
        """Durations render as s, m s or h m; negatives as 0s."""
        assert format_duration(seconds) == expected

"""Tests for sage.config: the YAML run-specification schema and loader.

The point of this schema is that mistakes fail loudly. The previous config
system forwarded attribute access to a plain object, so a typo'd field silently
never took effect and the run completed with the wrong settings. Most of what
is asserted here is therefore about rejection, not acceptance.
"""

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML required for run specs")

from sage.config import (  # noqa: E402
    ConfigError,
    KNOWN_STAGES,
    RunSpec,
    load_run_spec,
    loads_run_spec,
    resolve_export_dir,
)

MINIMAL = """
preset: bbh_production
name: r
data:
  train: {run: O3a, detectors: [H1, L1]}
"""

FULL = """
preset: bbh_production
name: prod_HL
data:
  train:    {run: O3a, detectors: [H1, L1]}
  validate: {run: O3b}
  test:     {run: O3b, detectors: [H1, L1, V1]}
priors:
  masses:   bbh_broad
  spins:    aligned_default
  distance: {min: 100, max: 3000}
stages: [plots, train, benchmark]
"""


class TestValidSpecs:
    def test_minimal_spec_parses(self):
        s = loads_run_spec(MINIMAL)
        assert isinstance(s, RunSpec)
        assert s.preset == "bbh_production"
        assert s.data.train.run == "O3a"
        assert s.data.train.detectors == ["H1", "L1"]

    def test_stages_default_to_train_only(self):
        assert loads_run_spec(MINIMAL).stages == ["train"]

    def test_roles_inherit_training_detectors(self):
        s = loads_run_spec(FULL)
        assert s.data.validate.detectors == ["H1", "L1"]

    def test_explicit_detectors_are_not_overwritten_by_inheritance(self):
        s = loads_run_spec(FULL)
        assert s.data.test.detectors == ["H1", "L1", "V1"]

    def test_stages_normalise_to_canonical_order(self):
        # Listed as [plots, train, benchmark]; execution order must not depend
        # on the order the user happened to type.
        assert loads_run_spec(FULL).stages == ["train", "benchmark", "plots"]

    def test_named_prior_set_and_explicit_range_coexist(self):
        p = loads_run_spec(FULL).priors
        assert p.masses == "bbh_broad"
        assert p.distance == {"min": 100.0, "max": 3000.0}

    def test_shorthand_string_data_selection(self):
        s = loads_run_spec(
            "preset: p\nname: r\ndata:\n"
            "  train: {run: O3a, detectors: [H1]}\n"
            "  validate: O3b\n"
        )
        assert s.data.validate.run == "O3b"
        assert s.data.validate.detectors == ["H1"]

    def test_absent_optional_sections_give_empty_defaults(self):
        s = loads_run_spec(MINIMAL)
        assert s.priors.masses is None
        assert s.custom.train is None
        assert s.data.test is None


class TestRejections:
    """Every one of these silently did the wrong thing under the old system."""

    @pytest.mark.parametrize(
        "text, expected_fragment",
        [
            # Typos must be caught, with a suggestion.
            (MINIMAL.replace("detectors:", "detecters:"), "did you mean 'detectors'"),
            (MINIMAL + "stagez: [train]\n", "did you mean 'stages'"),
            (MINIMAL + "stages: [benchmarking]\n", "did you mean 'benchmark'"),
            (MINIMAL + "priors:\n  masses: bbh_narow\n", "did you mean 'bbh_narrow'"),
            # Structural errors.
            ("preset: p\nname: r\n", "missing required option 'data'"),
            ("preset: p\nname: r\ndata:\n  train: {detectors: [H1]}\n", "missing required option 'run'"),
            ("preset: p\nname: r\ndata:\n  train: {run: O3a}\n", "'detectors' is required"),
            ("name: r\ndata:\n  train: {run: O3a, detectors: [H1]}\n", "missing required option 'preset'"),
            # Detector-list sanity.
            ("preset: p\nname: r\ndata:\n  train: {run: O3a, detectors: []}\n", "must not be empty"),
            ("preset: p\nname: r\ndata:\n  train: {run: O3a, detectors: [H1, H1]}\n", "repeated detector"),
            # Prior ranges.
            (MINIMAL + "priors:\n  distance: {min: 3000, max: 100}\n", "must be less than"),
            (MINIMAL + "priors:\n  masses: {min: 7.0}\n", "missing 'max'"),
            (MINIMAL + "priors:\n  masses: {min: 7, max: 50, mean: 20}\n", "unexpected key"),
            (MINIMAL + "priors:\n  masses: {min: low, max: 50}\n", "expected a number"),
            # Stages.
            (MINIMAL + "stages: [train, train]\n", "repeated stage"),
            (MINIMAL + "stages: []\n", "non-empty list"),
            # Types.
            ("preset: 3\nname: r\ndata:\n  train: {run: O3a, detectors: [H1]}\n", "non-empty string"),
        ],
    )
    def test_invalid_spec_is_rejected(self, text, expected_fragment):
        with pytest.raises(ConfigError) as exc:
            loads_run_spec(text, "spec.yaml")
        assert expected_fragment in str(exc.value)

    def test_no_suggestion_when_nothing_is_close(self):
        # A wild typo should list the valid options rather than invent a
        # misleading "did you mean".
        with pytest.raises(ConfigError) as exc:
            loads_run_spec(MINIMAL + "priors:\n  masses: enormous_stars\n")
        assert "did you mean" not in str(exc.value)
        assert "bbh_broad" in str(exc.value)

    def test_error_names_the_key_path_not_an_internal_field(self):
        with pytest.raises(ConfigError) as exc:
            loads_run_spec(MINIMAL + "priors:\n  distance: {min: 5, max: 1}\n")
        assert "priors.distance" in str(exc.value)

    def test_malformed_yaml_names_the_file(self):
        with pytest.raises(ConfigError) as exc:
            loads_run_spec("preset: [unclosed\n", "myrun.yaml")
        assert "myrun.yaml" in str(exc.value)
        assert "not valid YAML" in str(exc.value)

    def test_empty_file_is_rejected(self):
        with pytest.raises(ConfigError, match="empty"):
            loads_run_spec("", "myrun.yaml")

    def test_every_known_stage_is_actually_accepted(self):
        # Guards against KNOWN_STAGES drifting out of sync with the validator.
        s = loads_run_spec(MINIMAL + f"stages: [{', '.join(KNOWN_STAGES)}]\n")
        assert s.stages == list(KNOWN_STAGES)


class TestLoaderAndPaths:
    def test_load_from_file(self, tmp_path):
        p = tmp_path / "run.yaml"
        p.write_text(FULL)
        assert load_run_spec(p).name == "prod_HL"

    def test_missing_file_is_a_config_error_not_an_oserror(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_run_spec(tmp_path / "nope.yaml")

    def test_explicit_export_dir_wins(self, tmp_path):
        s = loads_run_spec(MINIMAL + f"export_dir: {tmp_path}/somewhere\n")
        assert resolve_export_dir(s) == tmp_path / "somewhere"

    def test_env_root_is_used_and_name_appended(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SAGE_RUN_ROOT", str(tmp_path))
        # Name is appended, so two runs from one root cannot collide silently.
        assert resolve_export_dir(loads_run_spec(MINIMAL)) == tmp_path / "r"

    def test_explicit_export_dir_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SAGE_RUN_ROOT", str(tmp_path / "env"))
        s = loads_run_spec(MINIMAL + f"export_dir: {tmp_path}/explicit\n")
        assert resolve_export_dir(s) == tmp_path / "explicit"

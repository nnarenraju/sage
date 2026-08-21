#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_spec.py
Description   : The configuration surface, its validation, and the resumability key.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

``hash()`` decides whether a stage is rerun or its previous output is reused, so it has
to be stable across processes and sensitive to everything that would change a result.
The obvious implementation, hashing the object's repr, is neither: Python salts string
hashing per process, so a campaign resumed in a new process would recompute everything,
and a spec whose input data changed underneath it would silently reuse stale products.

The subprocess test is the one that matters. Checking stability inside a single process
passes for an implementation that is not stable at all.

Runs anywhere; needs no data, no GPU and no network.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import dataclasses

import pytest

from sage.search.spec import (
    load_spec,
    CatalogueSpec,
    ClusterSpec,
    DataSpec,
    EngineSpec,
    GeometrySpec,
    PastroSpec,
    SearchSpec,
    SlideSpec,
)


def _spec(tmp_path, **overrides):
    """A valid spec rooted under a writable directory."""
    base = dict(
        tag="o3a-HL",
        config_module="runs.search.config_o3a_HL",
        out_dir=tmp_path / "campaign",
        data=DataSpec(
            observing_run="O3a",
            detectors=("H1", "L1"),
            release_dir=tmp_path / "release",
            fiducial_dir=tmp_path / "fiducial",
        ),
        engine=EngineSpec(
            checkpoint=tmp_path / "best.pt",
            gwconfig=tmp_path / "gwconfig.yaml",
        ),
        geometry=GeometrySpec(tc_source="explicit", tc_lower_s=5.0, tc_upper_s=7.0),
    )
    base.update(overrides)
    return SearchSpec(**base)


class TestArmIdentity:
    """A spec names one network searching one run."""

    def test_arm_key_from_detectors(self, tmp_path):
        """The arm key is the detector initials, so HL and HLV are distinguishable."""
        assert _spec(tmp_path).arm == "HL"

    def test_arm_key_for_three_detectors(self, tmp_path):
        spec = _spec(
            tmp_path,
            data=DataSpec(
                observing_run="O3a",
                detectors=("H1", "L1", "V1"),
                release_dir=tmp_path / "release",
                fiducial_dir=tmp_path / "fiducial",
            ),
        )
        assert spec.arm == "HLV"


class TestValidation:
    """Configurations that would produce wrong results are refused up front."""

    def test_valid_spec_passes(self, tmp_path):
        _spec(tmp_path).validate()

    def test_out_dir_under_tmp_is_refused(self, tmp_path):
        """
        Nothing this project runs may write to the system temp directory.

        A campaign writes tens of gigabytes and must survive a reboot; /tmp guarantees
        neither.
        """
        with pytest.raises(ValueError, match="tmp"):
            _spec(tmp_path, out_dir=Path("/tmp/campaign")).validate()

    def test_relative_out_dir_is_refused(self, tmp_path):
        """A relative root resolves differently depending on where a job starts."""
        with pytest.raises(ValueError, match="absolute"):
            _spec(tmp_path, out_dir=Path("campaign")).validate()

    def test_empty_detector_network_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="detector"):
            _spec(
                tmp_path,
                data=DataSpec(observing_run="O3a", detectors=()),
            ).validate()

    def test_repeated_detector_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="repeated"):
            _spec(
                tmp_path,
                data=DataSpec(observing_run="O3a", detectors=("H1", "H1")),
            ).validate()

    def test_reference_in_network(self, tmp_path):
        """
        Sliding relative to a detector the search does not read is meaningless.

        Easy to hit when moving from HL to a network that excludes H1.
        """
        with pytest.raises(ValueError, match="reference"):
            _spec(tmp_path, slides=SlideSpec(reference_detector="V1")).validate()

    def test_unknown_cluster_linkage_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="linkage"):
            _spec(tmp_path, cluster=ClusterSpec(linkage="average")).validate()

    def test_unknown_policy_refused(self, tmp_path):
        with pytest.raises(ValueError, match="monotonicity"):
            _spec(tmp_path, pastro=PastroSpec(monotonicity_policy="ignore")).validate()

    def test_negative_slide_count_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="n_slides"):
            _spec(tmp_path, slides=SlideSpec(n_slides=-1)).validate()

    def test_tau_max_exceeds_floor(self, tmp_path):
        """A lag range with no room in it yields no admissible slides."""
        with pytest.raises(ValueError, match="tau_max"):
            _spec(
                tmp_path, slides=SlideSpec(min_separation_s=100.0, tau_max_s=50.0)
            ).validate()

    def test_missing_observing_run_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="observing_run"):
            _spec(tmp_path, data=DataSpec(observing_run="")).validate()

    def test_explicit_tc_source_needs_bounds(self, tmp_path):
        """Asking for explicit bounds without giving them is a configuration error."""
        with pytest.raises(ValueError, match="tc"):
            _spec(tmp_path, geometry=GeometrySpec(tc_source="explicit")).validate()

    def test_unknown_tc_source_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="tc_source"):
            _spec(tmp_path, geometry=GeometrySpec(tc_source="guess")).validate()


class TestGeometryObject:
    """The spec materialises the geometry the rest of the search uses."""

    def test_explicit_bounds_are_used(self, tmp_path):
        geometry = _spec(tmp_path).geometry_object()
        assert geometry.stride_samples == 205
        assert geometry.tc_lower_s == 5.0
        assert geometry.tc_upper_s == 7.0

    def test_geometry_is_validated_on_construction(self, tmp_path):
        """A bad stride surfaces here rather than part-way through a campaign."""
        with pytest.raises((ValueError, TypeError)):
            _spec(
                tmp_path,
                geometry=GeometrySpec(
                    stride_samples=0, tc_source="explicit", tc_lower_s=5.0, tc_upper_s=7.0
                ),
            ).geometry_object()


class TestPaths:
    """Products resolve under the campaign root."""

    def test_path_joins_under_out_dir(self, tmp_path):
        spec = _spec(tmp_path)
        assert spec.path("far", "curve.h5") == spec.out_dir / "far" / "curve.h5"

    def test_path_with_no_parts_is_the_root(self, tmp_path):
        spec = _spec(tmp_path)
        assert spec.path() == spec.out_dir


class TestSerialisation:
    """The spec round-trips as JSON for provenance."""

    def test_to_json_is_parseable(self, tmp_path):
        payload = json.loads(_spec(tmp_path).to_json())
        assert payload["tag"] == "o3a-HL"
        assert payload["data"]["observing_run"] == "O3a"

    def test_paths_serialise_as_strings(self, tmp_path):
        payload = json.loads(_spec(tmp_path).to_json())
        assert isinstance(payload["out_dir"], str)

    def test_json_is_canonical(self, tmp_path):
        """Keys are ordered, so two equal specs give byte-identical JSON."""
        a = _spec(tmp_path).to_json()
        b = _spec(tmp_path).to_json()
        assert a == b


class TestHash:
    """The resumability key."""

    def test_equal_specs_hash_equal(self, tmp_path):
        assert _spec(tmp_path).hash() == _spec(tmp_path).hash()

    def test_path_and_string_are_equivalent(self, tmp_path):
        """Passing a path as a string must not invalidate a campaign."""
        as_path = _spec(tmp_path)
        as_str = _spec(tmp_path, out_dir=str(tmp_path / "campaign"))
        assert as_path.hash() == as_str.hash()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tag": "different"},
            {"seed": 1},
            {"cluster": ClusterSpec(window_s=0.5)},
            {"slides": SlideSpec(n_slides=8)},
            {"pastro": PastroSpec(threshold_far_per_day=1.0)},
            {"catalogue": CatalogueSpec(match_tolerance_s=2.0)},
        ],
    )
    def test_any_field_change_changes_the_hash(self, tmp_path, overrides):
        """Every setting that could change a result participates in the key."""
        assert _spec(tmp_path).hash() != _spec(tmp_path, **overrides).hash()

    def test_hash_is_stable_across_processes(self, tmp_path, repo_root):
        """
        The same spec hashes identically in two separate interpreters.

        Hashing a repr would pass an in-process check and fail this one, because Python
        salts string hashing per process. A campaign resumed in a new job would then
        recompute every stage.
        """
        script = textwrap.dedent(
            f"""
            from pathlib import Path
            from sage.search.spec import SearchSpec, DataSpec, EngineSpec, GeometrySpec
            root = Path({str(tmp_path)!r})
            spec = SearchSpec(
                tag="o3a-HL",
                config_module="runs.search.config_o3a_HL",
                out_dir=root / "campaign",
                data=DataSpec(
                    observing_run="O3a",
                    detectors=("H1", "L1"),
                    release_dir=root / "release",
                    fiducial_dir=root / "fiducial",
                ),
                engine=EngineSpec(checkpoint=root / "best.pt"),
                geometry=GeometrySpec(
                    tc_source="explicit", tc_lower_s=5.0, tc_upper_s=7.0
                ),
            )
            print(spec.hash())
            """
        )
        digests = set()
        for salt in ("0", "1"):
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": salt},
                timeout=300,
            )
            assert out.returncode == 0, out.stderr
            digests.add(out.stdout.strip())
        assert len(digests) == 1, f"hash varied across processes: {digests}"

    def test_input_data_participates_in_the_hash(self, tmp_path, synthetic_release):
        """
        Changing the strain underneath a campaign changes its key.

        Otherwise a rebuilt release would be silently analysed with products computed
        from the previous one.
        """
        spec = _spec(
            tmp_path,
            data=DataSpec(
                observing_run="O3a",
                detectors=("H1", "L1"),
                release_dir=synthetic_release,
                fiducial_dir=tmp_path / "fiducial",
            ),
        )
        before = spec.hash()

        sidecar = synthetic_release / "data_H1_O3a_segments.json"
        records = json.loads(sidecar.read_text())
        records[0]["noise_low_freq_cutoff"] = 20.0
        sidecar.write_text(json.dumps(records))

        assert spec.hash() != before

    def test_missing_release_still_hashes(self, tmp_path):
        """A spec can be hashed before its data exists, for planning and dry runs."""
        assert isinstance(_spec(tmp_path).hash(), str)


class TestGeometryResolution:
    """Where the window and the coalescence-time band come from."""

    def test_tc_prior_read_from_gwconfig(self, tmp_path):
        """
        The tc band comes from the training run's prior, not from a default.

        A Sage checkpoint records the window geometry but no tc key at all, so the
        training configuration is the only place the band exists. It is where in the window
        a merger was placed during training, and decode.tc_to_gps inverts that placement.
        """
        from sage.search.spec import read_tc_prior

        path = tmp_path / "gwconfig.yaml"
        path.write_text("priors:\n  tc:\n    name: uniform\n    min: 11.0\n    max: 11.2\n")

        assert read_tc_prior(path) == (11.0, 11.2)

    def test_non_uniform_tc_prior_refused(self, tmp_path):
        """
        Only a uniform band is accepted, because only a uniform band is inverted.

        decode.tc_to_gps maps a decoded tc through the band's endpoints. Another shape's
        endpoints do not mean the same thing, and taking them anyway would decode every
        merger time through the wrong map.
        """
        from sage.search.spec import read_tc_prior

        path = tmp_path / "gwconfig.yaml"
        path.write_text("priors:\n  tc:\n    name: normal\n    min: 11.0\n    max: 11.2\n")
        with pytest.raises(ValueError, match="only 'uniform' is supported"):
            read_tc_prior(path)

    def test_missing_tc_prior_refused(self, tmp_path):
        """A prior with no tc entry cannot place a merger in the window."""
        from sage.search.spec import read_tc_prior

        path = tmp_path / "gwconfig.yaml"
        path.write_text("priors:\n  mass1:\n    name: uniform\n    min: 7\n    max: 50\n")
        with pytest.raises(ValueError, match="no tc prior"):
            read_tc_prior(path)

    def test_absent_gwconfig_says_where_to_look(self):
        """
        An unset gwconfig names the two ways out rather than failing obscurely.

        This was the blocker on every campaign: geometry_object refused every tc_source
        but 'explicit', which is not what any config sets.
        """
        from sage.search.spec import read_tc_prior

        with pytest.raises(ValueError, match="tc_source='explicit'"):
            read_tc_prior("")

    def test_explicit_source_does_not_read_a_file(self, tmp_path):
        """Stated bounds are used as stated, with no configuration consulted."""
        spec = dataclasses.replace(
            _spec(tmp_path),
            geometry=GeometrySpec(
                stride_samples=205, tc_source="explicit",
                tc_lower_s=5.0, tc_upper_s=7.0,
            ),
        )

        assert spec.tc_prior() == (5.0, 7.0)
        assert spec.geometry_object().tc_lower_s == 5.0

    def test_window_falls_back_without_a_checkpoint(self, tmp_path):
        """
        A spec is buildable on a machine that does not hold the checkpoint.

        Every unit test builds one that way, so the window lengths fall back to Sage's
        defaults rather than refusing.
        """
        lengths = _spec(tmp_path).window_lengths()

        assert lengths["sample_rate"] == 2048.0
        assert lengths["sample_length_in_s"] == 12.0
        assert lengths["padding_length_in_s"] == 2.0

    def test_window_read_from_the_checkpoint_snapshot(self, tmp_path):
        """
        Where a snapshot is present the window is read from it, not assumed.

        The window length and its padding are the network's input shape; assuming them
        would hand a network trained on one length a different number of samples, whitened
        by the wrong frequency bins.
        """
        import json

        checkpoints = tmp_path / "CHECKPOINTS"
        checkpoints.mkdir()
        (checkpoints / "data_cfg_snapshot.json").write_text(
            json.dumps({
                "sample_rate": 4096.0,
                "sample_length_in_s": 8.0,
                "padding_length_in_s": 1.0,
            })
        )
        spec = dataclasses.replace(
            _spec(tmp_path), engine=EngineSpec(checkpoint=checkpoints / "best.pt")
        )
        lengths = spec.window_lengths()

        assert lengths["sample_rate"] == 4096.0
        assert lengths["sample_length_in_s"] == 8.0

    def test_real_campaign_geometry_builds(self):
        """
        The shipped O3a campaign resolves a complete geometry.

        The end-to-end check that matters: config -> checkpoint -> training prior ->
        SearchGeometry, with nothing assumed along the way. The stride is exactly
        205/2048 s; the nominal 0.1 s is 0.098% high and would drift over 92 million
        windows.
        """
        import pathlib

        config = pathlib.Path("runs/search/config_o3a_HL.py")
        prior = pathlib.Path("runs/o3b/gwconfig.yaml")
        if not (config.is_file() and prior.is_file()):
            pytest.skip("campaign config or training prior not present in this checkout")
        geometry = load_spec(str(config)).geometry_object()

        assert geometry.window_samples == 32768
        assert geometry.stride_samples == 205
        assert geometry.stride_s == pytest.approx(205 / 2048.0, rel=0, abs=0)
        assert geometry.tc_upper_s > geometry.tc_lower_s


class TestLoadSpec:
    """Importing a campaign configuration by name or by path."""

    @staticmethod
    def _config(tmp_path, body):
        path = tmp_path / "config_probe.py"
        path.write_text(body)
        return path

    _VALID = """
from pathlib import Path
from sage.search.spec import DataSpec, EngineSpec, GeometrySpec, SearchSpec

def get_spec():
    return SearchSpec(
        tag="probe",
        out_dir=Path("/work/nagarajan/sage_runs/probe"),
        data=DataSpec(
            observing_run="O3a",
            detectors=("H1", "L1"),
            release_dir=Path("/work/nagarajan/release"),
            fiducial_dir=Path("/work/nagarajan/fiducial"),
        ),
        engine=EngineSpec(checkpoint=Path("/work/nagarajan/best.pt")),
        geometry=GeometrySpec(tc_source="explicit", tc_lower_s=5.0, tc_upper_s=7.0),
    )
"""

    def test_loads_from_a_path(self, tmp_path):
        """
        A submit script has the path, so the path spelling has to work.

        The file's own directory is made importable while it loads, because a real config
        imports its sibling ``config_base``.
        """
        spec = load_spec(str(self._config(tmp_path, self._VALID)))

        assert isinstance(spec, SearchSpec)
        assert spec.tag == "probe"
        assert spec.data.observing_run == "O3a"

    def test_load_does_not_shadow_an_installed_module(self, tmp_path):
        """
        A config file is not registered under its bare stem.

        ``config_HL.py`` registered as ``config_HL`` would displace any installed module
        of that name for the rest of the process -- and it is exactly how an older
        checkpoint format became unloadable, by pickling a class from a module named
        ``config_HL`` that no longer exists.
        """
        import sys

        path = self._config(tmp_path, self._VALID)
        before = set(sys.modules)
        load_spec(str(path))

        assert "config_probe" not in sys.modules
        assert path.stem not in set(sys.modules) - before

    def test_failed_load_leaves_no_module_behind(self, tmp_path):
        """
        A config that raises while executing does not stay in ``sys.modules``.

        A half-executed module left registered is found by the next import, which then
        sees a config missing whatever the exception interrupted -- and reports the
        failure somewhere far from its cause.
        """
        import sys

        path = tmp_path / "config_broken.py"
        path.write_text("raise RuntimeError('config is broken')\n")
        before = set(sys.modules)
        with pytest.raises(RuntimeError, match="config is broken"):
            load_spec(str(path))

        assert set(sys.modules) - before == set()

    def test_two_configs_of_one_name_are_distinct(self, tmp_path):
        """
        Two directories may both hold ``config_HL.py``; loading one must not serve the
        other. The module name is derived from the resolved path for this reason.
        """
        first = tmp_path / "a"
        second = tmp_path / "b"
        for directory, out in ((first, "alpha"), (second, "beta")):
            directory.mkdir()
            self._config(
                directory, self._VALID.replace('sage_runs/probe', 'sage_runs/' + out)
            )

        one = load_spec(str(first / "config_probe.py"))
        two = load_spec(str(second / "config_probe.py"))

        assert one.config_module != two.config_module
        assert Path(one.out_dir).name == "alpha"
        assert Path(two.out_dir).name == "beta"

    def test_unknown_dotted_name_says_so(self, tmp_path):
        """
        A dotted name that is not importable names itself in the error.

        Without this the caller sees a ModuleNotFoundError from deep inside importlib and
        no statement that a config was what failed to load.
        """
        with pytest.raises(ModuleNotFoundError, match="neither an existing file"):
            load_spec("runs.search.config_that_does_not_exist")

    def test_config_module_is_stamped(self, tmp_path):
        """
        A config that does not name itself gets named, so products can be traced back.

        ``config_module`` reaches every provenance block, and a campaign whose outputs do
        not say which configuration produced them cannot be reproduced from them. A file
        is stamped with its resolved path rather than its stem: two runs directories both
        holding ``config_HL.py`` is a normal arrangement, and the stem would not say which
        one ran.
        """
        path = self._config(tmp_path, self._VALID)
        spec = load_spec(str(path))
        assert spec.config_module == str(path.resolve())

    def test_spec_attribute_accepted(self, tmp_path):
        """A module-level SPEC is the other supported shape."""
        body = self._VALID.replace("def get_spec():\n    return", "SPEC = (lambda:") + ")()"
        spec = load_spec(str(self._config(tmp_path, body)))
        assert spec.tag == "probe"

    def test_module_without_an_entry_point_refused(self, tmp_path):
        """
        Neither entry point present is an error, not a search for something spec-shaped.

        Picking one of several candidates would decide the campaign silently.
        """
        path = self._config(tmp_path, "VALUE = 1\n")
        with pytest.raises(ValueError, match="neither get_spec"):
            load_spec(str(path))

    def test_wrong_type_refused(self, tmp_path):
        """An entry point returning the wrong thing fails here, not three stages later."""
        path = self._config(tmp_path, "def get_spec():\n    return {'tag': 'probe'}\n")
        with pytest.raises(ValueError, match="not a SearchSpec"):
            load_spec(str(path))

    def test_missing_file_refused(self, tmp_path):
        """A mistyped path names the path rather than failing inside importlib."""
        with pytest.raises(FileNotFoundError):
            load_spec(str(tmp_path / "config_absent.py"))

    def test_real_campaign_config_loads(self):
        """
        The shipped ``runs/search/config_o3a_HL.py`` loads through this path.

        Pinned against the actual configuration rather than only against fixtures: the
        convention this function implements is the one that file uses, and a change to
        either that breaks the pair should fail here. O3a is the live campaign -- a
        network trained on O3b searching O3a -- so it is the one that must load.
        """
        import pathlib

        config = pathlib.Path("runs/search/config_o3a_HL.py")
        if not config.is_file():
            pytest.skip("campaign config not present in this checkout")
        spec = load_spec(str(config))

        assert isinstance(spec, SearchSpec)
        assert spec.data.observing_run == "O3a"
        assert spec.arm == "HL"
        assert spec.engine.training_config

    def test_unconfigured_campaign_says_what_is_missing(self):
        """
        A campaign with no trained network refuses by name rather than by symptom.

        Assembling a spec from empty paths fails validation complaining about an unset
        checkpoint, which is true but does not say that no O4a network exists yet.
        """
        import pathlib

        config = pathlib.Path("runs/search/config_o4a_HL.py")
        if not config.is_file():
            pytest.skip("campaign config not present in this checkout")
        with pytest.raises(NotImplementedError, match="not configured"):
            load_spec(str(config))

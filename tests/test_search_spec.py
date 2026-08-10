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

import pytest

from sage.search.spec import (
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
        engine=EngineSpec(checkpoint=tmp_path / "best.pt"),
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

    def test_reference_detector_must_be_in_the_network(self, tmp_path):
        """
        Sliding relative to a detector the search does not read is meaningless.

        Easy to hit when moving from HL to a network that excludes H1.
        """
        with pytest.raises(ValueError, match="reference"):
            _spec(tmp_path, slides=SlideSpec(reference_detector="V1")).validate()

    def test_unknown_cluster_linkage_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="linkage"):
            _spec(tmp_path, cluster=ClusterSpec(linkage="average")).validate()

    def test_unknown_monotonicity_policy_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="monotonicity"):
            _spec(tmp_path, pastro=PastroSpec(monotonicity_policy="ignore")).validate()

    def test_negative_slide_count_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="n_slides"):
            _spec(tmp_path, slides=SlideSpec(n_slides=-1)).validate()

    def test_tau_max_must_exceed_minimum_separation(self, tmp_path):
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

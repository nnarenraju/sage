#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_fixtures.py
Description   : The synthetic fixtures really do have the properties they claim.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Layers 1 through 5 are tested almost entirely against these two fixtures, so a fixture
that is quietly tidier than reality would make every one of those tests vacuous. A
GPS-sorted release passes a reader that assumes sorting; a network whose frontend happens
to ignore its input passes a separability check under any normalisation.

These are the tests of the tests. They are not xfail: the fixtures are implemented.

Runs anywhere; needs no data, no GPU and no network.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.search_fixtures import (
    DYN_RANGE_FAC,
    REAL_OVERLAP_S,
    ToyFrontendNet,
    make_legacy_checkpoint,
    make_synthetic_checkpoint,
    make_synthetic_release,
    release_is_gps_sorted,
    toy_batch,
)


def _records(root, detector="H1", run="O3a"):
    path = Path(root) / f"data_{detector}_{run}_segments.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestReleaseRealism:
    """The release reproduces the properties that make the real one hard to read."""

    def test_index_is_contiguous(self, synthetic_release):
        """Records tile the binary file with no gap and no overlap in sample index."""
        records = _records(synthetic_release)
        for a, b in zip(records, records[1:]):
            assert a["sample_start_idx"] + a["nsamples"] == b["sample_start_idx"]

    def test_records_are_not_gps_sorted(self, synthetic_release):
        """
        Numbering is anti-correlated with time, as in the real sidecars.

        This is the property a reader most easily assumes and most easily gets wrong, so
        a fixture that happened to be sorted would hide the bug rather than expose it.
        """
        assert not release_is_gps_sorted(synthetic_release, "H1", "O3a")

    def test_sorted_variant_is_available_as_a_control(self, tmp_path):
        """The sorted release exists only to show a test would have passed regardless."""
        root = make_synthetic_release(
            tmp_path / "sorted", detectors=("H1",), shuffle_index=False
        )
        assert release_is_gps_sorted(root, "H1", "O3a")

    def test_consecutive_chunks_overlap_in_time(self, synthetic_release):
        """Chunks adjacent in time share an interval, at the measured real width."""
        records = sorted(_records(synthetic_release), key=lambda r: r["gps_start"])
        overlaps = [
            a["gps_end"] - b["gps_start"] for a, b in zip(records, records[1:])
        ]
        assert overlaps
        assert all(o == pytest.approx(REAL_OVERLAP_S, abs=1e-6) for o in overlaps)

    def test_overlapping_samples_differ_between_chunks(self, tmp_path):
        """
        The same GPS second holds different samples in the two chunks that cover it.

        This is what makes splicing across a boundary a detectable error rather than a
        harmless duplication, and it is true of the real release.
        """
        root = make_synthetic_release(
            tmp_path / "noisy", detectors=("H1",), fill="noise", n_chunks=3
        )
        records = sorted(_records(root), key=lambda r: r["gps_start"])
        first, second = records[0], records[1]
        raw = np.fromfile(root / "data_H1_O3a.bin", dtype="<f4") / DYN_RANGE_FAC
        rate = first["sample_rate"]

        overlap_start = second["gps_start"]
        n_overlap = int(round((first["gps_end"] - overlap_start) * rate))
        assert n_overlap > 0

        tail_of_first = raw[
            first["sample_start_idx"]
            + first["nsamples"]
            - n_overlap : first["sample_start_idx"]
            + first["nsamples"]
        ]
        head_of_second = raw[
            second["sample_start_idx"] : second["sample_start_idx"] + n_overlap
        ]
        assert not np.allclose(tail_of_first, head_of_second)

    def test_constant_fill_makes_boundary_crossing_detectable(self, synthetic_release):
        """
        Every sample in a chunk carries that chunk's value, so a spliced window shows it.

        A window wholly inside one chunk has zero peak-to-peak; one that crosses a
        boundary does not. That single assertion catches a reader that walks off the end
        of a segment.
        """
        records = _records(synthetic_release)
        raw = np.fromfile(synthetic_release / "data_H1_O3a.bin", dtype="<f4")
        raw = raw / DYN_RANGE_FAC

        first = records[0]
        inside = raw[first["sample_start_idx"] : first["sample_start_idx"] + 1024]
        assert np.ptp(inside) == 0.0

        boundary = first["sample_start_idx"] + first["nsamples"] - 512
        crossing = raw[boundary : boundary + 1024]
        assert np.ptp(crossing) > 0.0

    def test_sidecar_carries_the_real_schema(self, synthetic_release):
        """Every field a reader depends on is present."""
        required = {
            "segment_index",
            "detector",
            "observing_run",
            "gps_start",
            "gps_end",
            "sample_rate",
            "nsamples",
            "dtype",
            "endianness",
            "sample_start_idx",
            "byte_offset",
            "byte_length",
            "checksum",
            "dyn_range_fac",
            "noise_low_freq_cutoff",
        }
        for record in _records(synthetic_release):
            assert required <= set(record)

    def test_checksums_match_the_written_bytes(self, synthetic_release):
        """The sidecar's checksum verifies against the binary, as the real one does."""
        import hashlib

        raw = (synthetic_release / "data_H1_O3a.bin").read_bytes()
        for record in _records(synthetic_release):
            chunk = raw[
                record["byte_offset"] : record["byte_offset"] + record["byte_length"]
            ]
            assert hashlib.sha256(chunk).hexdigest() == record["checksum"]

    def test_three_detector_release(self, synthetic_release_hlv):
        """The fixture builds an HLV release, not only HL."""
        for detector in ("H1", "L1", "V1"):
            assert (synthetic_release_hlv / f"data_{detector}_O3a.bin").exists()
            assert _records(synthetic_release_hlv, detector)


class TestCheckpointFixture:
    """The checkpoint is in the format the loader must accept, and its twin is not."""

    def test_configs_are_flat_dicts(self, synthetic_checkpoint):
        """Loading needs no project class to be importable."""
        loaded = torch.load(synthetic_checkpoint, map_location="cpu", weights_only=False)
        assert isinstance(loaded["cfg"], dict)
        assert isinstance(loaded["data_cfg"], dict)
        assert all(
            isinstance(v, (str, int, float, bool, list, type(None)))
            for v in loaded["cfg"].values()
        )

    def test_geometry_keys_are_present(self, synthetic_checkpoint):
        """The fields the search validates its geometry against are all stored."""
        loaded = torch.load(synthetic_checkpoint, map_location="cpu", weights_only=False)
        data_cfg = loaded["data_cfg"]
        for key in (
            "sample_rate",
            "sample_length_in_s",
            "padding_length_in_s",
            "padded_length_in_nsamples",
            "noise_low_frequency_cutoff",
        ):
            assert key in data_cfg

    def test_legacy_checkpoint_stores_an_object(self, tmp_path):
        """
        The superseded format is reproducible, so the loader's refusal can be tested.

        Every checkpoint written before the flat-dict change stores a config object whose
        class no longer exists, which is why they cannot be reopened at all.
        """
        path = make_legacy_checkpoint(tmp_path / "legacy.pt")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert not isinstance(loaded["cfg"], dict)


class TestSeparability:
    """The property the frontend feature cache depends on, and its negative control."""

    @pytest.mark.parametrize("num_detectors", [2, 3])
    def test_instancenorm_frontend_is_separable(self, num_detectors):
        """
        Perturbing one detector leaves every other detector's frontend bitwise unchanged.

        Bitwise, not approximately: the cache reuses stored features verbatim across time
        slides, so anything short of exact equality means the cached value is not what a
        fresh forward pass would produce.
        """
        model = ToyFrontendNet(
            num_detectors=num_detectors, norm_type="instancenorm"
        ).eval()
        x = toy_batch(num_detectors)
        with torch.no_grad():
            for i in range(num_detectors):
                perturbed = x.clone()
                perturbed[:, i] += 1.0
                for j in range(num_detectors):
                    before = model.forward_frontend(x, j)
                    after = model.forward_frontend(perturbed, j)
                    if i == j:
                        assert not torch.equal(before, after)
                    else:
                        assert torch.equal(before, after), f"det {j} moved with det {i}"

    @pytest.mark.parametrize("num_detectors", [2, 3])
    def test_groupnorm_frontend_is_not_separable(self, num_detectors):
        """
        The negative control: under a group spanning detectors, caching is invalid.

        Without this, the separability test above would pass for a model whose frontend
        ignored its input, and the cache would be enabled on a network that cannot support
        it.
        """
        model = ToyFrontendNet(
            num_detectors=num_detectors, norm_type="groupnorm"
        ).eval()
        x = toy_batch(num_detectors)
        with torch.no_grad():
            perturbed = x.clone()
            perturbed[:, 0] += 1.0
            moved = [
                not torch.equal(
                    model.forward_frontend(x, j), model.forward_frontend(perturbed, j)
                )
                for j in range(1, num_detectors)
            ]
        assert all(moved), "groupnorm must couple the detector axis"

    def test_unknown_norm_is_rejected(self):
        """A typo in the norm name fails loudly rather than defaulting."""
        with pytest.raises(ValueError, match="norm_type"):
            ToyFrontendNet(norm_type="layernorm")

    @pytest.mark.parametrize("num_detectors", [2, 3])
    def test_forward_returns_the_expected_shapes(self, num_detectors):
        """Output matches the real network's contract: ranking, then blocked PE."""
        model = ToyFrontendNet(num_detectors=num_detectors, num_pe=2).eval()
        with torch.no_grad():
            ranking, point_estimates = model(toy_batch(num_detectors, batch=5))
        assert ranking.shape == (5, 1)
        assert point_estimates.shape == (5, 4)


class TestTempRootPolicy:
    """Temporary files must not land in the system temp directory."""

    def test_tmp_path_is_not_under_slash_tmp(self, tmp_path):
        """
        pytest's temporary directories are redirected off /tmp.

        Left at the default, pytest, tarfile, matplotlib and astropy all write there, and
        this project may not.
        """
        assert not str(tmp_path).startswith("/tmp/"), tmp_path

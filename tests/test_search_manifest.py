#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_manifest.py
Description   : Provenance stamping, the stage journal and the run manifest.

Created on 2026-08-13

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A product whose provenance is missing or wrong is worse than no product, because it is
believed. Everything here is aimed at that one failure.

The spec hash is the field that carries the weight: it is what a later job checks a
product against before reusing it, so a block recording a hash that drifts between
processes reattributes every product in the campaign. ``SearchSpec.hash`` already has the
stability the search needs, and the tests below check that ``provenance`` carries it
through unchanged rather than re-deriving something adjacent to it. The subprocess check
is the one that matters; an in-process comparison passes for an implementation that is not
stable at all.

The journal is written by every stage at once, from separate jobs, so its append property
is tested by actually running concurrent writers rather than by inspecting the open mode.

Runs anywhere; needs no data, no GPU and no network.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pathlib

import pytest

from sage.search.manifest import (
    PROVENANCE_KEYS,
    searched_run_is_held_out,
    UNRECORDED,
    RunManifest,
    journal,
    provenance,
    stamp,
    verify,
)
from sage.search.spec import DataSpec, EngineSpec, GeometrySpec, SearchSpec
from sage.utils.atomic_io import atomic_h5, write_h5


def _gwconfig(directory):
    """A minimal training prior, so the geometry can be resolved from a fixture."""
    path = pathlib.Path(directory) / "gwconfig.yaml"
    if not path.is_file():
        path.write_text("priors:\n  tc:\n    name: uniform\n    min: 11.0\n    max: 11.2\n")
    return path


def _spec(tmp_path, **overrides):
    """A valid spec whose geometry is fully determined, as test_search_spec builds it."""
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
            gwconfig=_gwconfig(tmp_path),
        ),
        geometry=GeometrySpec(tc_source="explicit", tc_lower_s=5.0, tc_upper_s=7.0),
    )
    base.update(overrides)
    return SearchSpec(**base)


def _stamped(path, attrs, **datasets):
    """Write a product carrying ``attrs``, through the writer the search itself uses."""
    with atomic_h5(path) as handle:
        for name, values in (datasets or {"stat": [1.0, 2.0]}).items():
            handle.create_dataset(name, data=values)
        stamp(handle, attrs)
    return path


class _Checkpoint:
    """Stand-in for a LoadedCheckpoint, whose loader is not yet implemented."""

    def __init__(self, path, sha256, data_cfg=None, cfg=None):
        self.path = path
        self.sha256 = sha256
        self.data_cfg = data_cfg or {}
        self.cfg = cfg or {}


class TestProvenanceBlock:
    """What the block contains."""

    def test_every_declared_key_is_present(self, tmp_path):
        """
        A block carries all of PROVENANCE_KEYS.

        The tuple is the contract every reader relies on; a key that is silently absent
        turns into a missing column in the methods table long after the run.
        """
        attrs = provenance(_spec(tmp_path))
        assert set(PROVENANCE_KEYS) <= set(attrs)

    def test_config_recorded_from_spec(self, tmp_path):
        """The block states the configuration it was built from, not a default."""
        attrs = provenance(_spec(tmp_path))
        assert attrs["config_module"] == "runs.search.config_o3a_HL"
        assert attrs["observing_run"] == "O3a"
        assert attrs["detectors"] == ("H1", "L1")
        assert attrs["stride_samples"] == 205
        assert attrs["seed"] == 20260809

    def test_extras_merged_and_win(self, tmp_path):
        """
        A caller can add what it alone knows, and correct what it knows better.

        The geometry mismatches a checkpoint check found, or the slide a shard belongs
        to, exist only at the call site; without this they would go unrecorded.
        """
        attrs = provenance(_spec(tmp_path), slide_id=7, observing_run="O3b")
        assert attrs["slide_id"] == 7
        assert attrs["observing_run"] == "O3b"

    def test_git_state_recorded_or_blank(self, tmp_path):
        """
        The code version is a commit or nothing.

        A product has to be traceable to the code that wrote it. Where the checkout has no
        repository the commit is unknowable, and a blank says so; anything else would
        attribute the product to code that may not have produced it.
        """
        attrs = provenance(_spec(tmp_path))
        assert attrs["git_hash"] == UNRECORDED or len(attrs["git_hash"]) == 40
        assert attrs["git_dirty"] in (True, False, UNRECORDED)


class TestRecordedSpecHash:
    """The field a resumed campaign trusts its previous products on."""

    def test_hash_is_specs_own(self, tmp_path):
        """
        provenance records ``spec.hash()`` unchanged.

        The resume check compares a product's recorded hash against the hash of the spec
        being run. A block that recorded anything else -- a digest of a subset of the
        spec, a rounded copy -- would compare unequal to every live spec and rebuild
        products that were already correct.
        """
        spec = _spec(tmp_path)
        assert provenance(spec)["spec_hash"] == spec.hash()

    def test_hash_survives_subprocess(self, tmp_path, repo_root):
        """
        The same spec yields the same recorded hash in a separate interpreter.

        The stage that writes a product and the stage that later reuses it are different
        jobs. If the hash the block carries depended on anything per-process, every
        resumed campaign would recompute everything, and this is the only check that
        catches it: an in-process comparison passes regardless.
        """
        script = textwrap.dedent(
            f"""
            from pathlib import Path
            from sage.search.manifest import provenance
            from sage.search.spec import (
                DataSpec, EngineSpec, GeometrySpec, SearchSpec
            )
            root = Path({str(tmp_path)!r})
            (root / "gwconfig.yaml").write_text(
                "priors:\\n  tc:\\n    name: uniform\\n    min: 11.0\\n    max: 11.2\\n"
            )
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
                engine=EngineSpec(
                    checkpoint=root / "best.pt",
                    gwconfig=root / "gwconfig.yaml",
                ),
                geometry=GeometrySpec(
                    tc_source="explicit", tc_lower_s=5.0, tc_upper_s=7.0
                ),
            )
            print(provenance(spec)["spec_hash"])
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
        assert len(digests) == 1, f"recorded hash varied across processes: {digests}"
        assert digests == {provenance(_spec(tmp_path))["spec_hash"]}

    def test_str_path_same_hash(self, tmp_path):
        """
        How a path was spelled must not reattribute a product.

        A configuration module that writes ``out_dir`` as a string and one that writes it
        as a ``Path`` describe the same campaign, and a block that distinguished them
        would invalidate every product on the next run.
        """
        as_path = provenance(_spec(tmp_path))
        as_string = provenance(_spec(tmp_path, out_dir=str(tmp_path / "campaign")))
        assert as_path["spec_hash"] == as_string["spec_hash"]

    def test_changed_data_changes_hash(
        self, tmp_path, synthetic_release
    ):
        """
        Rebuilding the strain underneath a campaign invalidates its products.

        Without this the previous release's triggers would be reused against the new one,
        and nothing downstream could see it.
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
        before = provenance(spec)["spec_hash"]

        sidecar = synthetic_release / "data_H1_O3a_segments.json"
        records = json.loads(sidecar.read_text())
        records[0]["noise_low_freq_cutoff"] = 20.0
        sidecar.write_text(json.dumps(records))

        assert provenance(spec)["spec_hash"] != before


class TestUnavailableFields:
    """What a block does when it cannot know something."""

    def test_unknown_geometry_is_blank(self, tmp_path):
        """
        A window length that is not known is recorded as missing.

        The coalescence-time band comes from the training run's own prior, so a spec
        deferring to one it cannot reach has not resolved its geometry at all. Filling in
        the production values then would produce a block that is plausible, wrong, and
        impossible to distinguish from a measured one.
        """
        spec = _spec(
            tmp_path,
            geometry=GeometrySpec(tc_source="checkpoint"),
            engine=EngineSpec(
                checkpoint=tmp_path / "best.pt",
                gwconfig=tmp_path / "absent" / "gwconfig.yaml",
            ),
        )
        attrs = provenance(spec)
        assert attrs["sample_rate"] == UNRECORDED
        assert attrs["window_samples"] == UNRECORDED

    def test_geometry_from_spec(self, tmp_path):
        """A spec that can state its own geometry has it recorded, in samples."""
        attrs = provenance(_spec(tmp_path))
        assert attrs["sample_rate"] == 2048.0
        assert attrs["window_samples"] == 32768

    def test_digest_blank_without_checkpoint(self, tmp_path):
        """
        No checkpoint means no digest, never a fabricated one.

        The digest is what proves a product came from the weights it names; a placeholder
        would let two products from different networks claim the same identity.
        """
        attrs = provenance(_spec(tmp_path))
        assert attrs["checkpoint_sha256"] == UNRECORDED
        assert attrs["checkpoint_path"].endswith("best.pt")

    def test_checkpoint_supplies_digest(self, tmp_path):
        """
        The network actually loaded is what gets recorded.

        The checkpoint carries both the weight digest and the window geometry the weights
        were trained under, which is the geometry the product was produced with even when
        the spec defers it.
        """
        ckpt = _Checkpoint(
            path=tmp_path / "loaded.pt",
            sha256="a" * 64,
            data_cfg={
                "sample_rate": 2048.0,
                "sample_length_in_s": 12.0,
                "padding_length_in_s": 2.0,
            },
        )
        attrs = provenance(
            _spec(tmp_path, geometry=GeometrySpec(tc_source="checkpoint")), ckpt
        )
        assert attrs["checkpoint_sha256"] == "a" * 64
        assert attrs["checkpoint_path"] == str(tmp_path / "loaded.pt")
        assert attrs["sample_rate"] == 2048.0
        assert attrs["window_samples"] == 32768


class TestStampAndVerify:
    """The stamp on a product and the check that reads it back."""

    def test_verify_round_trips(self, tmp_path):
        """
        The block a product carries is the block that was written.

        HDF5 hands back numpy scalars and object arrays, so a stamp read back without
        decoding compares unequal to its own source and every provenance check that
        compares blocks fails on types rather than on content.
        """
        attrs = provenance(_spec(tmp_path))
        path = _stamped(tmp_path / "triggers.h5", attrs)
        assert verify(path) == attrs

    def test_detectors_round_trip(self, tmp_path):
        """The network a product was produced with reads back as names, not bytes."""
        attrs = provenance(_spec(tmp_path))
        path = _stamped(tmp_path / "triggers.h5", attrs)
        assert verify(path)["detectors"] == ("H1", "L1")

    def test_expected_hash_accepted(self, tmp_path):
        """A product produced under the running configuration passes its check."""
        spec = _spec(tmp_path)
        path = _stamped(tmp_path / "triggers.h5", provenance(spec))
        assert verify(path, expect_spec_hash=spec.hash())["spec_hash"] == spec.hash()

    def test_wrong_hash_refused(self, tmp_path):
        """
        A product from another configuration is rejected, loudly.

        This is the whole point of the check. Reusing a product built under different
        settings is invisible in the result: the numbers are there, they are plausible,
        and they answer a question nobody asked.
        """
        spec = _spec(tmp_path)
        path = _stamped(tmp_path / "triggers.h5", provenance(spec))
        other = _spec(tmp_path, seed=1).hash()
        with pytest.raises(ValueError, match="spec"):
            verify(path, expect_spec_hash=other)

    def test_incomplete_block_refused(self, tmp_path):
        """
        A partial stamp never reaches disk.

        A product carrying some of its provenance reads as provenanced and is trusted,
        and the field that would have contradicted the reader is precisely the missing
        one. Refusing at write time keeps that product from existing.
        """
        attrs = dict(provenance(_spec(tmp_path)))
        attrs.pop("spec_hash")
        with atomic_h5(tmp_path / "triggers.h5") as handle:
            with pytest.raises(ValueError, match="spec_hash"):
                stamp(handle, attrs)

    def test_unstamped_product_refused(self, tmp_path):
        """
        A product with no provenance cannot be attributed and is refused.

        Returning an empty block instead would let a caller compare it against nothing
        and conclude the product was fine.
        """
        path = tmp_path / "unstamped.h5"
        write_h5(path, {"stat": [1.0, 2.0]})
        with pytest.raises(ValueError, match="provenance"):
            verify(path)

    def test_missing_product_reported(self, tmp_path):
        """A product that was never written is a missing file, not a bad stamp."""
        with pytest.raises(FileNotFoundError):
            verify(tmp_path / "absent.h5")

    def test_nested_value_refused(self, tmp_path):
        """
        The block stays flat.

        Encoding a nested value into an attr would hide it from every tool that reads the
        file, so it is refused rather than silently serialised, and refused before
        anything is written so the product is not left half stamped.
        """
        attrs = dict(provenance(_spec(tmp_path)))
        attrs["coverage"] = {"union_s": 1.0}
        with atomic_h5(tmp_path / "triggers.h5") as handle:
            with pytest.raises(TypeError):
                stamp(handle, attrs)
            assert "spec_hash" not in handle.attrs

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([[1.0, 2.0], [3.0]], id="ragged list"),
            pytest.param({"a", "b"}, id="set"),
            pytest.param(object(), id="arbitrary object"),
            pytest.param([1.0, "text"], id="mixed sequence"),
            pytest.param([{"nested": 1}], id="sequence of dicts"),
        ],
    )
    def test_unstampable_leaves_nothing(self, tmp_path, value):
        """
        A value the attrs cannot hold is refused before the first one is written.

        The guarantee is all-or-nothing, and it is worth having only if it covers every
        value h5py might reject rather than the one shape that happens to be checked. A
        block that fails part-way through leaves exactly the product the module says it
        refuses -- one that looks provenanced, and is therefore trusted, with the field
        that would have contradicted the reader among the ones never written.

        These arrive by the route ``provenance`` invites: ``**extra`` is merged last and
        is documented as the place for "the geometry mismatches a checkpoint check
        reported", so a caller's own structure lands after the whole declared block.
        """
        attrs = dict(provenance(_spec(tmp_path)))
        attrs["mismatches"] = value
        with atomic_h5(tmp_path / "triggers.h5") as handle:
            with pytest.raises(TypeError):
                stamp(handle, attrs)
            assert dict(handle.attrs) == {}

    def test_flat_sequence_stamped(self, tmp_path):
        """The allowlist admits what an attr genuinely holds, and is not a blanket no."""
        attrs = dict(provenance(_spec(tmp_path)))
        attrs["mismatches"] = ["sample_rate", "window_samples"]
        path = tmp_path / "triggers.h5"
        with atomic_h5(path) as handle:
            stamp(handle, attrs)
        assert verify(path)["mismatches"] == ("sample_rate", "window_samples")


class TestJournal:
    """The append-only campaign timeline."""

    def test_events_append(self, tmp_path):
        """
        The journal appends.

        A journal that truncated would hold only the last stage to finish, which is the
        one piece of the timeline that is already known.
        """
        path = tmp_path / "journal" / "stages.jsonl"
        journal(path, {"stage": "segments", "status": "ok"})
        journal(path, {"stage": "grid", "status": "ok"})
        lines = path.read_text().splitlines()
        assert [json.loads(line)["stage"] for line in lines] == ["segments", "grid"]

    def test_event_recorded_verbatim(self, tmp_path):
        """What a stage reported is what the line says."""
        path = tmp_path / "stages.jsonl"
        journal(path, {"stage": "far", "n_triggers": 12, "seconds": 3.5})
        record = json.loads(path.read_text().splitlines()[0])
        assert record["stage"] == "far"
        assert record["n_triggers"] == 12
        assert record["seconds"] == 3.5

    def test_every_line_timestamped(self, tmp_path):
        """A timeline without times cannot order the jobs that wrote it."""
        path = tmp_path / "stages.jsonl"
        journal(path, {"stage": "segments"})
        assert json.loads(path.read_text().splitlines()[0])["utc"]

    def test_supplied_time_kept(self, tmp_path):
        """
        A stage that reports when something happened is not overwritten.

        The time a stage records is the time of the event; the time the line was written
        is only a fallback for events that did not carry one.
        """
        path = tmp_path / "stages.jsonl"
        journal(path, {"stage": "segments", "utc": "2026-01-01T00:00:00Z"})
        record = json.loads(path.read_text().splitlines()[0])
        assert record["utc"] == "2026-01-01T00:00:00Z"

    def test_concurrent_journal_writers(self, tmp_path, repo_root):
        """
        Four processes writing at once produce four processes' worth of whole lines.

        Every stage journals, stages run simultaneously in separate jobs, and they share
        one journal. A writer that read, appended and rewrote would drop whatever landed
        between its read and its write; one that wrote unlocked could interleave two
        lines into one unparseable one. Both fail here and neither is visible in a
        single-process test.
        """
        path = tmp_path / "stages.jsonl"
        n_workers, per_worker = 4, 50
        script = textwrap.dedent(
            """
            import sys
            from sage.search.manifest import journal
            path, worker, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
            for index in range(count):
                journal(path, {"worker": worker, "index": index})
            """
        )
        runner = tmp_path / "write_journal.py"
        runner.write_text(script)
        processes = [
            subprocess.Popen(
                [sys.executable, str(runner), str(path), str(worker), str(per_worker)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for worker in range(n_workers)
        ]
        for process in processes:
            _, stderr = process.communicate(timeout=300)
            assert process.returncode == 0, stderr.decode()

        lines = path.read_text().splitlines()
        assert len(lines) == n_workers * per_worker
        written = {
            (record["worker"], record["index"])
            for record in (json.loads(line) for line in lines)
        }
        assert written == {
            (worker, index)
            for worker in range(n_workers)
            for index in range(per_worker)
        }


class TestRunManifest:
    """The campaign-level record of what ran and over how much time."""

    def test_concurrent_manifest_writers(self, tmp_path, repo_root):
        """
        Four processes recording at once leave four entries, and four live jobs.

        The class docstring's premise is that stages run as separate jobs, and the stage
        graph makes that literal: ``zerolag``, ``slides`` and ``injections`` depend only
        on ``grid``, so ``submit.sh`` puts them on the queue together. Each records into
        one manifest through a read-modify-write that replaces the file from the snapshot
        it opened, which without a lock loses whichever entry landed in between -- or
        collides on the shared temporary path and kills the job. The journal is already
        locked for exactly this; the manifest is written by the same jobs.
        """
        path = tmp_path / "manifest.h5"
        stages = ("segments", "grid", "zerolag", "slides")
        script = textwrap.dedent(
            """
            import sys
            from sage.search.manifest import RunManifest
            path, stage, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
            RunManifest(path=path).record_stage(stage, {"worker": index})
            """
        )
        runner = tmp_path / "write_manifest.py"
        runner.write_text(script)
        processes = [
            subprocess.Popen(
                [sys.executable, str(runner), str(path), stage, str(index)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for index, stage in enumerate(stages)
        ]
        for process in processes:
            _, stderr = process.communicate(timeout=300)
            assert process.returncode == 0, stderr.decode()

        summary = RunManifest(path=path).summary()
        assert set(summary["stages"]) == set(stages)
        assert {report["worker"] for report in summary["stages"].values()} == set(
            range(len(stages))
        )
        # Order is a real sequence, not four writers all claiming the same slot.
        assert len(set(summary["complete"])) == len(stages)

    def test_report_returned_verbatim(self, tmp_path):
        """A stage's own numbers reach the summary unaltered."""
        manifest = RunManifest(path=tmp_path / "manifest.h5")
        manifest.record_stage("far", {"n_triggers": 12, "ifar_max_yr": 3.5})
        assert manifest.summary()["stages"]["far"] == {
            "n_triggers": 12,
            "ifar_max_yr": 3.5,
        }

    def test_unrun_stage_absent(self, tmp_path):
        """
        Nothing is invented for a stage that did not run.

        A stage reported with a zero livetime, or an empty report, is indistinguishable in
        a table from one that ran and measured nothing. Absence is the only honest answer,
        and the campaign is meant to be resumable off exactly this.
        """
        manifest = RunManifest(path=tmp_path / "manifest.h5")
        manifest.record_stage("segments", {"livetime_s": 100.0})
        summary = manifest.summary()
        assert "far" not in summary["stages"]
        assert "far" not in summary["complete"]

    def test_completion_in_record_order(self, tmp_path):
        """
        The summary states which stages are complete, in the order they finished.

        The order is the campaign's own history; HDF5 group iteration is alphabetical and
        would report ``background`` before ``segments``, which is not what happened.
        """
        manifest = RunManifest(path=tmp_path / "manifest.h5")
        manifest.record_stage("segments", {"livetime_s": 100.0})
        manifest.record_stage("grid", {"n_windows": 7})
        manifest.record_stage("background", {"n_slides": 2})
        assert manifest.summary()["complete"] == ("segments", "grid", "background")

    def test_rerun_replaces_entry(self, tmp_path):
        """
        A re-run stage is recorded once, with its latest report.

        A second entry would leave the manifest describing products that are no longer on
        disk alongside the ones that are.
        """
        manifest = RunManifest(path=tmp_path / "manifest.h5")
        manifest.record_stage("far", {"n_triggers": 12})
        manifest.record_stage("far", {"n_triggers": 15})
        summary = manifest.summary()
        assert summary["complete"] == ("far",)
        assert summary["stages"]["far"] == {"n_triggers": 15}

    def test_livetime_per_run(self, tmp_path):
        """
        Each run's livetime stands on its own.

        Every rate the search quotes divides by one of these, and two runs' livetimes are
        not interchangeable.
        """
        manifest = RunManifest(path=tmp_path / "manifest.h5")
        manifest.record_livetime("O3a", {"union_s": 100.0, "hosted_s": 90.0})
        manifest.record_livetime("O3b", {"union_s": 200.0, "hosted_s": 180.0})
        summary = manifest.summary()
        assert summary["runs"] == ("O3a", "O3b")
        assert summary["livetime"]["O3a"]["hosted_s"] == 90.0
        assert summary["livetime"]["O3b"]["hosted_s"] == 180.0

    def test_coverage_round_trips(self, tmp_path):
        """
        The decomposition the segment layer emits is stored whole.

        ``CoverageReport`` is what ``record_livetime`` is given in the campaign, and the
        lost-time terms are what make the analysed time defensible rather than asserted.
        """
        from sage.search.segments import CoverageReport

        coverage = CoverageReport(
            union_s=100.0,
            hosted_s=88.0,
            n_windows=880,
            lost_window_fit_s=8.0,
            lost_boundary_holes_s=3.0,
            lost_phase_restart_s=1.0,
            n_holes=2,
        )
        manifest = RunManifest(path=tmp_path / "manifest.h5")
        manifest.record_livetime("O3a", coverage.as_dict())
        assert manifest.summary()["livetime"]["O3a"] == coverage.as_dict()

    def test_empty_manifest_reports_nothing(self, tmp_path):
        """
        A campaign that has not started summarises as empty rather than failing.

        The manifest is read to decide what to run next, including on the first call,
        when nothing has written it yet.
        """
        summary = RunManifest(path=tmp_path / "manifest.h5").summary()
        assert summary["stages"] == {}
        assert summary["complete"] == ()
        assert summary["livetime"] == {}

    def test_state_lives_in_the_file(self, tmp_path):
        """
        A later job sees what earlier jobs recorded.

        Stages run as separate jobs, so the process that records a stage is almost never
        the one that reads the manifest back. State held on the instance would make every
        job report only its own work.
        """
        path = tmp_path / "manifest.h5"
        RunManifest(path=path).record_stage("segments", {"livetime_s": 100.0})
        RunManifest(path=path).record_livetime("O3a", {"union_s": 100.0})
        summary = RunManifest(path=path).summary()
        assert summary["complete"] == ("segments",)
        assert summary["runs"] == ("O3a",)

    def test_nested_report_recorded(self, tmp_path):
        """
        Stage reports may nest, unlike the flat block stamped onto a product.

        Throughput and per-detector numbers arrive as nested mappings, and flattening them
        to fit HDF5 attrs would change what the stage recorded.
        """
        manifest = RunManifest(path=tmp_path / "manifest.h5")
        report = {"throughput": {"windows_per_s": 2.5, "blocks": [1, 2, 3]}}
        manifest.record_stage("zerolag", report)
        assert manifest.summary()["stages"]["zerolag"] == report


class TestTrainSearchDisjointness:
    """The searched run must be readable as disjoint from the trained-on runs."""

    @staticmethod
    def _ckpt(tmp_path, train_runs):
        return _Checkpoint(
            tmp_path / "best.pt", "a" * 64, cfg={"train_runs": list(train_runs)}
        )

    def test_train_runs_recorded(self, tmp_path):
        """
        Every product says which runs its network was trained on.

        The production configuration trains on O3b and searches O3a, so the background is
        drawn from data the network never saw. That is true by construction, and recording
        it makes it a property of the file rather than a claim made about the file.
        """
        attrs = provenance(_spec(tmp_path), ckpt=self._ckpt(tmp_path, ["O3b"]))

        assert attrs["train_runs"] == ("O3b",)
        assert attrs["observing_run"] == "O3a"

    def test_disjointness_readable_from_a_product(self, tmp_path):
        """The question is answered from the stamped block, not from a running job."""
        held_out = provenance(_spec(tmp_path), ckpt=self._ckpt(tmp_path, ["O3b"]))
        overlapping = provenance(
            _spec(tmp_path), ckpt=self._ckpt(tmp_path, ["O3a", "O3b"])
        )

        assert searched_run_is_held_out(held_out)
        assert not searched_run_is_held_out(overlapping)

    def test_unanswerable_is_not_a_pass(self, tmp_path):
        """
        A block that cannot answer the question reports False, not True.

        A checkpoint too old to record its training runs leaves the overlap unknown, and
        an unknown overlap is not a held-out one; reporting True would let a missing field
        read as a guarantee.
        """
        assert not searched_run_is_held_out(provenance(_spec(tmp_path)))
        assert not searched_run_is_held_out(
            provenance(_spec(tmp_path), ckpt=self._ckpt(tmp_path, []))
        )

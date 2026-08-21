#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_fingerprint.py
Description   : Fingerprints must move when the product does.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

A fingerprint decides whether everything downstream of a stage is rebuilt, so the failure
worth testing is the silent one: the product changed and the fingerprint did not. Every
case below is a change a summary-scalar fingerprint was measured to miss -- a lattice
shifted by one sample, a lag ladder read out in reverse, a loss decomposition
re-attributed between its terms, a FAR curve divided by a different livetime.

The complementary property matters as much and is tested beside each: an unchanged
product must keep its fingerprint, or the cascade fires on every re-run and a campaign
never resumes.

Synthetic release throughout.
"""

import dataclasses

import numpy as np
import pytest

from sage.search.fingerprint import combine, digest_h5, digest_values

pytest.importorskip("h5py")

from tests.test_search_drivers import _background, _campaign  # noqa: E402


class TestDigestValues:
    """The in-memory digest: what it must separate and what it must not."""

    def test_mapping_order_ignored(self):
        """Two spellings of one dict are one product."""
        left = digest_values({"a": np.arange(4), "b": 1.0})
        right = digest_values({"b": 1.0, "a": np.arange(4)})
        assert left == right

    def test_sequence_order_kept(self):
        """
        A list is ordered and the order is part of the product. A lag ladder is a list.
        """
        assert digest_values({"x": [1, 2, 3]}) != digest_values({"x": [3, 2, 1]})

    def test_types_separated(self):
        """
        ``1``, ``1.0``, ``True`` and ``"1"`` render alike and are different products.
        """
        digests = {
            digest_values({"x": value}) for value in (1, 1.0, True, "1", None)
        }
        assert len(digests) == 5

    def test_array_layout_ignored(self):
        """
        A transposed view holds the same bytes in a different order; the digest reads
        the array, not the buffer.
        """
        base = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert digest_values({"x": base.T}) != digest_values({"x": base})
        assert digest_values({"x": base.T}) == digest_values(
            {"x": np.ascontiguousarray(base.T)}
        )

    def test_float_bits(self):
        """``-0.0`` and ``0.0`` compare equal and are not the same product."""
        assert digest_values({"x": -0.0}) != digest_values({"x": 0.0})

    def test_unknown_type_refused(self):
        """
        ``repr`` embeds object addresses, so the fallback a digest cannot have is a
        silent one.
        """
        with pytest.raises(TypeError, match="no defined digest"):
            digest_values({"x": object()})


class TestDigestH5:
    """The persisted digest: contents, not bytes."""

    def test_rewrite_stable(self, tmp_path):
        """
        A dataset carrying ``track_times`` embeds a modification time, so a digest over
        file bytes moves on every re-write of identical data and cascades the campaign
        each time.
        """
        import time

        import h5py

        path = tmp_path / "x.h5"
        raw, digests = [], []
        for _ in range(2):
            with h5py.File(path, "w") as handle:
                handle.create_dataset("x", data=np.arange(32), track_times=True)
                handle.attrs["livetime_s"] = 12.5
            raw.append(path.read_bytes())
            digests.append(digest_h5(path))
            time.sleep(1.1)
        assert raw[0] != raw[1]
        assert digests[0] == digests[1]

    def test_layout_ignored(self, tmp_path):
        """
        Chunk shape and compression change a file's bytes and not its contents. A
        fingerprint that noticed them would rebuild a campaign over a storage setting.
        """
        import h5py

        path = tmp_path / "x.h5"
        raw, digests = [], []
        for options in ({}, {"chunks": (8,), "compression": "gzip"}):
            with h5py.File(path, "w") as handle:
                handle.create_dataset("x", data=np.arange(32), **options)
                handle.attrs["livetime_s"] = 12.5
            raw.append(path.read_bytes())
            digests.append(digest_h5(path))
        assert raw[0] != raw[1]
        assert digests[0] == digests[1]

    def test_attribute_change_seen(self, tmp_path):
        """A livetime lives in an attribute, and it is a denominator."""
        import h5py

        digests = []
        path = tmp_path / "x.h5"
        for livetime in (12.5, 12.6):
            with h5py.File(path, "w") as handle:
                handle.create_dataset("x", data=np.arange(32))
                handle.attrs["livetime_s"] = livetime
            digests.append(digest_h5(path))
        assert digests[0] != digests[1]

    def test_provenance_clock_ignored(self, tmp_path):
        """
        Every product carries a provenance block whose write time and git state move on
        their own. Digesting them cascades a campaign on every re-run of unchanged work,
        which is the same failure from the other side and a louder one: an operator whose
        pipeline rebuilds everything every time learns to pass --no-cascade.
        """
        import h5py

        digests = []
        path = tmp_path / "x.h5"
        for stamp in ("2026-08-20T10:00:00Z", "2026-08-20T11:00:00Z"):
            with h5py.File(path, "w") as handle:
                handle.create_dataset("x", data=np.arange(8))
                handle.attrs["created_utc"] = stamp
                handle.attrs["git_dirty"] = stamp.endswith("11:00:00Z")
                handle.attrs["spec_hash"] = "abc123"
            digests.append(digest_h5(path))
        assert digests[0] == digests[1]

    def test_checkpoint_identity_kept(self, tmp_path):
        """
        A network retrained in place changes ``checkpoint_sha256`` and nothing else, so
        excluding the provenance block wholesale would hide the one change that matters
        most.
        """
        import h5py

        digests = []
        path = tmp_path / "x.h5"
        for sha in ("aaa", "bbb"):
            with h5py.File(path, "w") as handle:
                handle.create_dataset("x", data=np.arange(8))
                handle.attrs["created_utc"] = "2026-08-20T10:00:00Z"
                handle.attrs["checkpoint_sha256"] = sha
            digests.append(digest_h5(path))
        assert digests[0] != digests[1]

    def test_missing_file_distinct(self, tmp_path):
        """
        A stage that wrote a different subset of its products has a different product.
        """
        import h5py

        present, absent = tmp_path / "a.h5", tmp_path / "b.h5"
        with h5py.File(present, "w") as handle:
            handle.create_dataset("x", data=np.arange(4))
        assert digest_h5([present, absent]) != digest_h5([present])


class TestProductsTracked:
    """Regression: each case is a change a scalar fingerprint was measured to miss."""

    def test_grid_lattice_phase(self, tmp_path, monkeypatch):
        """
        A lattice shifted by one sample holds the same window count over the same
        livetime and scores an entirely different set of stretches of strain.
        """
        import sage.search.grid as grid
        import sage.search.segments as segments

        spec = _campaign(tmp_path)
        segments.run(spec)
        before = grid.run(spec)["fingerprint"]
        assert grid.run(spec)["fingerprint"] == before

        original = grid.window_hosts

        def shifted(*args, **kwargs):
            result = original(*args, **kwargs)
            spans, rest = (result[0], result[1:]) if isinstance(result, tuple) else (
                result,
                (),
            )
            moved = [
                dataclasses.replace(s, first_local=s.first_local + 1) for s in spans
            ]
            return (moved, *rest) if rest else moved

        monkeypatch.setattr(grid, "window_hosts", shifted)
        assert grid.run(spec)["fingerprint"] != before

    def test_segments_loss_attribution(self, tmp_path, monkeypatch):
        """
        The decomposition says whether a deficit is a genuine gap or a lattice
        restarting its phase. Re-attributing between the terms leaves the total alone.
        """
        import sage.search.segments as segments

        spec = _campaign(tmp_path)
        before = segments.run(spec)["fingerprint"]
        assert segments.run(spec)["fingerprint"] == before

        original = segments.coverage_report

        def moved(*args, **kwargs):
            report = original(*args, **kwargs)
            return dataclasses.replace(
                report,
                lost_phase_restart_s=report.lost_phase_restart_s
                + report.lost_window_fit_s,
                lost_window_fit_s=0.0,
            )

        monkeypatch.setattr(segments, "coverage_report", moved)
        assert segments.run(spec)["fingerprint"] != before

    def test_slides_ladder_order(self, tmp_path, monkeypatch):
        """
        Reversing the ladder leaves the summed background livetime exactly where it was
        and gives every slide_id -- and so every shard named by one -- a different lag.
        """
        import sage.search.segments as segments
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        segments.run(spec)
        before = slides.run(spec)["fingerprint"]
        assert slides.run(spec)["fingerprint"] == before

        original = slides.stratified_lags
        monkeypatch.setattr(
            slides, "stratified_lags", lambda *a, **k: original(*a, **k)[::-1]
        )
        assert slides.run(spec)["fingerprint"] != before

    def test_far_livetime(self, tmp_path):
        """
        A FAR curve is counts over a livetime. The livetime reaches no fitted shape
        parameter, so a curve wrong by that ratio kept its fingerprint.
        """
        import sage.search.far as far
        import sage.search.segments as segments
        import sage.search.slides as slides

        spec = _campaign(tmp_path)
        segments.run(spec)
        slides.run(spec)
        background = _background(spec)
        before = far.run(spec)["fingerprint"]
        assert far.run(spec)["fingerprint"] == before

        halved = dataclasses.replace(
            background, livetime_s=background.livetime_s / 2.0
        )
        halved.save(spec.path("background", "bg_inclusive.h5"))
        assert far.run(spec)["fingerprint"] != before


class TestEngineShard:
    """The engine's product is its shard's contents, not its row count."""

    def test_statistics_tracked(self, tmp_path):
        """
        A changed checkpoint moves every ranking statistic while leaving the row count
        exactly where it was.
        """
        import h5py

        from sage.search.fingerprint import digest_h5

        path = tmp_path / "shard.h5"
        digests = []
        for scale in (1.0, 2.0):
            with h5py.File(path, "w") as handle:
                group = handle.create_group("triggers")
                group.create_dataset("stat", data=np.arange(8.0) * scale)
                group.create_dataset("gps", data=np.arange(8.0))
            digests.append(digest_h5(path))
        assert digests[0] != digests[1]

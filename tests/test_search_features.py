#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_features.py
Description   : The frontend feature cache and the depth at which it pays for itself.

Created on 2026-08-19

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The cache is what makes a deep background affordable: under a separable frontend each
detector's features are computed once per window and reused for every lag, so a slide
costs a backend pass rather than a full forward. It is only legal when
``checkpoint.assert_separable`` passes, and the tests here are about the bookkeeping --
that a lag-shifted gather returns the right windows, and that eviction never silently
drops one the widest remaining slide still needs.
"""

import pytest

torch = pytest.importorskip("torch")

from sage.search.features import CacheResidency, FrontendCache, crossover_slides  # noqa: E402


def _cache(n_detectors=2, shape=(4, 8)):
    return FrontendCache(n_detectors, shape, device="cpu", dtype="float32")


class TestStoreAndGather:
    """Features must come back attached to the windows they were computed from."""

    def test_round_trip(self):
        cache = _cache()
        values = torch.randn(3, 4, 8)
        cache.put(0, [10, 11, 12], values)

        assert torch.equal(cache.gather(0, [10, 11, 12]), values)

    def test_gather_follows_the_requested_order(self):
        """
        A lag-shifted gather asks for windows out of order, and must get them in the order
        it asked. Returning them sorted would pair each window's features with a different
        window's backend input, and every value involved is still a real feature.
        """
        cache = _cache()
        values = torch.randn(3, 4, 8)
        cache.put(0, [10, 11, 12], values)
        out = cache.gather(0, [12, 10])

        assert torch.equal(out[0], values[2])
        assert torch.equal(out[1], values[0])

    def test_detectors_are_independent(self):
        """Two detectors holding the same window ids must not share storage."""
        cache = _cache()
        first, second = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        cache.put(0, [5, 6], first)
        cache.put(1, [5, 6], second)

        assert torch.equal(cache.gather(0, [5, 6]), first)
        assert torch.equal(cache.gather(1, [5, 6]), second)

    def test_stored_copy_is_detached_from_the_batch(self):
        """
        Mutating the batch afterwards must not change what was cached.

        A live view would pin the whole batch in memory for as long as any one of its
        windows is reachable, and would keep it attached to an autograd graph the search
        has no use for.
        """
        cache = _cache()
        batch = torch.randn(2, 4, 8)
        cache.put(0, [1, 2], batch)
        before = cache.gather(0, [1]).clone()
        batch[0] += 99.0

        assert torch.equal(cache.gather(0, [1]), before)

    def test_missing_window_raises(self):
        """
        A gather outside the cached halo is a fault, not a gap to fill.

        Zero-filling would score as an ordinary quiet window and hide an undersized halo
        inside the background rather than reporting it.
        """
        cache = _cache()
        cache.put(0, [10, 11], torch.randn(2, 4, 8))
        with pytest.raises(KeyError, match="no cached features"):
            cache.gather(0, [9, 10])

    def test_mismatched_lengths_refused(self):
        """Ids and rows read side by side must describe the same windows."""
        cache = _cache()
        with pytest.raises(ValueError, match="window ids against"):
            cache.put(0, [1, 2, 3], torch.randn(2, 4, 8))

    def test_wrong_feature_shape_refused(self):
        """A feature of the wrong shape means the frontend is not the one configured."""
        cache = _cache()
        with pytest.raises(ValueError, match="expected"):
            cache.put(0, [1], torch.randn(1, 4, 9))

    def test_unknown_detector_refused(self):
        cache = _cache(n_detectors=2)
        with pytest.raises(IndexError):
            cache.put(2, [1], torch.randn(1, 4, 8))


class TestEviction:
    """Dropping only what no remaining lag can reach."""

    def test_evicts_strictly_before_the_boundary(self):
        """The boundary window itself is still reachable and must survive."""
        cache = _cache()
        cache.put(0, [10, 11, 12], torch.randn(3, 4, 8))
        cache.evict_before(11)

        assert cache.gather(0, [11, 12]).shape[0] == 2
        with pytest.raises(KeyError):
            cache.gather(0, [10])

    def test_eviction_spans_every_detector(self):
        """A window dropped for one detector is unusable for all of them."""
        cache = _cache()
        for detector in (0, 1):
            cache.put(detector, [1, 2], torch.randn(2, 4, 8))
        cache.evict_before(2)

        assert cache.residency().n_windows == 1
        for detector in (0, 1):
            with pytest.raises(KeyError):
                cache.gather(detector, [1])

    def test_residency_tracks_what_is_held(self):
        """
        The footprint is what decides whether a block fits, so it has to follow the store.

        float32 features of shape (4, 8) are 128 bytes each; two detectors holding three
        windows is 768.
        """
        cache = _cache()
        assert cache.residency().total_bytes == 0
        for detector in (0, 1):
            cache.put(detector, [1, 2, 3], torch.randn(3, 4, 8))
        residency = cache.residency()

        assert residency.bytes_per_window_per_detector == 4 * 8 * 4
        assert residency.n_windows == 3
        assert residency.total_bytes == 768

    def test_residency_arithmetic(self):
        """The dataclass multiplies out rather than being told a total."""
        assert CacheResidency(128, 10, 3).total_bytes == 3840


class TestCrossover:
    """When caching starts being cheaper than re-running the whole model."""

    def test_matches_the_closed_form(self):
        """
        Uncached ``n / f_full`` against cached ``1/f_front + (1+n)/f_back``.

        At 100, 200 and 400 the equality is ``4n = 3 + n``, so the crossover is exactly
        one slide. Checked against the equation rather than a recomputation of the code.
        """
        n = crossover_slides(f_full=100.0, f_front=200.0, f_back=400.0)
        assert n == pytest.approx(1.0)

        uncached = n / 100.0
        cached = 1.0 / 200.0 + (1.0 + n) / 400.0
        assert uncached == pytest.approx(cached)

    def test_never_pays_when_the_backend_is_no_faster(self):
        """
        ``inf`` rather than a negative number.

        If the backend alone is no quicker than the whole model then lifting the frontend
        out bought nothing, and a negative crossover would read as "always worth it".
        """
        assert crossover_slides(100.0, 200.0, 100.0) == float("inf")
        assert crossover_slides(100.0, 200.0, 50.0) == float("inf")

    def test_falls_as_the_backend_gets_cheaper(self):
        """A cheaper backend pays for the cache at shallower ladders."""
        shallow = crossover_slides(100.0, 200.0, 10000.0)
        deep = crossover_slides(100.0, 200.0, 110.0)

        assert shallow < deep

    def test_non_positive_rates_refused(self):
        for bad in ({"f_full": 0.0}, {"f_front": -1.0}, {"f_back": float("nan")}):
            kwargs = {"f_full": 100.0, "f_front": 200.0, "f_back": 400.0, **bad}
            with pytest.raises(ValueError, match="positive finite rate"):
                crossover_slides(**kwargs)

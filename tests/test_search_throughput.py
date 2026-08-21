#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_throughput.py
Description   : The measurement that sets the campaign's compute budget.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The measuring itself needs a GPU. What is checkable without one is the arithmetic the
measurement feeds -- and that is where a wrong number would be believed, because a cost
projection has nothing to be compared against until the campaign has already run.
"""

import pytest

from sage.diagnostics import diagnose_search_throughput as tp
from sage.search.features import crossover_slides


class TestProjection:
    """Cost with and without the frontend cache."""

    RATES = {"f_full": 5_000.0, "f_front": 12_000.0, "f_back": 20_000.0}

    def test_uncached_is_linear_in_slides(self):
        """No cache means one full pass per slide, and nothing shared between them."""
        one = tp.project(self.RATES, 1, 1_000_000)
        ten = tp.project(self.RATES, 10, 1_000_000)

        assert ten["uncached_gpu_h"] == pytest.approx(10 * one["uncached_gpu_h"])

    def test_cached_counts_the_foreground_pass(self):
        """
        A ladder of n slides costs n+1 backend passes: the zero-lag foreground goes
        through the backend like any other pairing. Dropping it understates the cost by
        one pass, which at 82 slides is small and at 2 is not.
        """
        plan = tp.project(self.RATES, 2, 1_000_000)
        expected = (
            1_000_000 / self.RATES["f_front"]
            + 3 * 1_000_000 / self.RATES["f_back"]
        ) / 3600.0

        assert plan["cached_gpu_h"] == pytest.approx(expected)

    def test_cache_verdict_follows_the_crossover(self):
        """The reported verdict has to be the one the crossover implies, not a rule."""
        crossover = crossover_slides(**self.RATES)
        below = tp.project(self.RATES, max(int(crossover), 0), 1_000_000)
        above = tp.project(self.RATES, int(crossover) + 5, 1_000_000)

        assert not below["cache_pays"]
        assert above["cache_pays"]

    def test_useless_cache_reported_as_such(self):
        """
        A backend no cheaper than the whole model means the frontend split bought
        nothing. That must read as "never pays", not as a negative crossover that would
        compare as smaller than any slide count and so as always worth it.
        """
        rates = {"f_full": 5_000.0, "f_front": 12_000.0, "f_back": 4_000.0}
        plan = tp.project(rates, 82, 1_000_000)

        assert plan["crossover_slides"] == float("inf")
        assert not plan["cache_pays"]

    def test_zerolag_is_one_full_pass(self):
        """The foreground is scored once through the whole network, cache or not."""
        plan = tp.project(self.RATES, 82, 1_000_000)

        assert plan["zerolag_gpu_h"] == pytest.approx(
            1_000_000 / self.RATES["f_full"] / 3600.0
        )


class TestWarmup:
    """The first pass is not the rate a campaign runs at."""

    def test_warmup_batches_excluded(self):
        """
        A first pass pays for kernel autotuning and pool growth. Counting it reports a
        rate the campaign never sees again, and always an optimistic error in the wrong
        direction -- it makes the budget look smaller than it is.
        """
        calls = []

        class _Batch:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

        batches = [_Batch(100) for _ in range(5)]
        scored, seconds = tp._timed(calls.append, batches, warmup=2, device_type="cpu")

        assert len(calls) == 5      # every batch is run
        assert scored == 300        # only three are counted
        assert seconds > 0.0

    def test_single_batch_has_no_warmup(self):
        """With one batch there is nothing to discard, and discarding it measures zero."""

        class _Batch:
            def __len__(self):
                return 10

        _, warmup = tp._collect(iter([_Batch()]), 5)
        assert warmup == 0

    def test_empty_reader_refused(self):
        """A lattice with no windows is a configuration error, not a rate of zero."""
        with pytest.raises(RuntimeError, match="no windows"):
            tp._collect(iter([]), 100)

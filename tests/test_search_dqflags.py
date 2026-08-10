#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_dqflags.py
Description   : Flag policy: what a search requires, and what it must never require.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One of these tests exists because of a measured near-miss. Continuous hardware injections
run for the whole observing run, so ``NO_CW_HW_INJ`` is satisfied for zero seconds:
requiring it, as one would if the injection flags were treated as a single family, deletes
every second of data. The resulting dataset is empty rather than wrong, which is the good
case; the bad case is a policy that removes most of a run and is never questioned.

The published flag set differs between runs, so a policy is resolved against the run it
will be applied to and fails loudly if a flag it needs is not published there.

Runs on synthetic flag definitions; needs no data, no GPU and no network.
"""

import pytest

from sage.search.dqflags import (
    CONTINUOUS_INJECTION_FLAGS,
    TRANSIENT_INJECTION_FLAGS,
    FlagBit,
    FlagPolicy,
    RunFlags,
    search_policy,
    training_policy,
)


def _run_flags(observing_run="O3a", extra_dq=(), extra_inj=()):
    """A run publishing the flags the third observing run actually publishes."""
    dq_names = ["DATA", "CBC_CAT1", "CBC_CAT2", "CBC_CAT3", *extra_dq]
    inj_names = [
        "NO_CBC_HW_INJ",
        "NO_BURST_HW_INJ",
        "NO_DETCHAR_HW_INJ",
        "NO_CW_HW_INJ",
        "NO_STOCH_HW_INJ",
        *extra_inj,
    ]
    return RunFlags(
        observing_run=observing_run,
        dataset=f"{observing_run}_16KHZ_R1",
        dq_bits=tuple(
            FlagBit(name, i, 1 << i, f"{name} description")
            for i, name in enumerate(dq_names)
        ),
        inj_bits=tuple(
            FlagBit(name, i, 1 << i, f"{name} description")
            for i, name in enumerate(inj_names)
        ),
    )


class TestRunFlags:
    """What a run publishes is queried, not assumed."""

    def test_names_covers_both_masks(self):
        flags = _run_flags()
        assert "DATA" in flags.names()
        assert "NO_CBC_HW_INJ" in flags.names()

    def test_has(self):
        flags = _run_flags()
        assert flags.has("CBC_CAT1")
        assert not flags.has("CBC_CAT9")

    def test_require_passes_when_published(self):
        _run_flags().require(["DATA", "CBC_CAT1"])

    def test_require_names_the_run_and_what_it_publishes(self):
        """
        A missing flag reports the run and the available set.

        The published set differs between runs, so 'flag not found' alone sends the
        reader to the wrong place.
        """
        with pytest.raises(ValueError) as excinfo:
            _run_flags("O4a").require(["CBC_CAT9"])
        message = str(excinfo.value)
        assert "O4a" in message and "CBC_CAT9" in message

    def test_injection_bits_are_identified_as_such(self):
        flags = _run_flags()
        by_name = {b.short_name: b for b in flags.inj_bits}
        assert by_name["NO_CBC_HW_INJ"].is_injection
        dq = {b.short_name: b for b in flags.dq_bits}
        assert not dq["CBC_CAT1"].is_injection


class TestContinuousInjectionGuard:
    """The guard that stops a policy silently emptying a dataset."""

    def test_continuous_flags_are_listed_separately(self):
        """Continuous and transient injections are different families."""
        assert set(CONTINUOUS_INJECTION_FLAGS).isdisjoint(TRANSIENT_INJECTION_FLAGS)
        assert "NO_CW_HW_INJ" in CONTINUOUS_INJECTION_FLAGS
        assert "NO_CBC_HW_INJ" in TRANSIENT_INJECTION_FLAGS

    def test_requiring_a_continuous_flag_is_refused(self):
        """
        Measured over 600 ks of each run, NO_CW_HW_INJ holds for zero seconds.

        Requiring it removes every second of data, so the policy refuses rather than
        producing an empty release.
        """
        policy = FlagPolicy(injection_flags=("NO_CBC_HW_INJ", "NO_CW_HW_INJ"))
        with pytest.raises(ValueError, match="continuous"):
            policy.validate(_run_flags())

    def test_requiring_a_continuous_flag_via_extra_is_also_refused(self):
        """The guard covers the escape hatch too."""
        policy = FlagPolicy(extra=("NO_STOCH_HW_INJ",))
        with pytest.raises(ValueError, match="continuous"):
            policy.validate(_run_flags())

    def test_transient_flags_are_permitted(self):
        FlagPolicy().validate(_run_flags())

    def test_unpublished_flag_is_refused(self):
        policy = FlagPolicy(categories=("CBC_CAT1", "CBC_CAT9"))
        with pytest.raises(ValueError, match="CBC_CAT9"):
            policy.validate(_run_flags())


class TestPolicies:
    """The two policies differ in exactly the ways that matter."""

    def test_search_policy_requires_category_one(self):
        assert "CBC_CAT1" in search_policy("O3a").categories

    def test_search_policy_excludes_transient_injections(self):
        assert set(search_policy("O3a").injection_flags) == set(TRANSIENT_INJECTION_FLAGS)

    def test_search_policy_follows_how_each_run_was_analysed(self):
        """
        Category two was applied in the third observing run and dropped in the fourth.

        Defaulting to how the run was actually analysed keeps a comparison against the
        published results like for like.
        """
        assert "CBC_CAT2" in search_policy("O3a").categories
        assert "CBC_CAT2" in search_policy("O3b").categories
        assert "CBC_CAT2" not in search_policy("O4a").categories

    def test_category_two_can_be_overridden_explicitly(self):
        assert "CBC_CAT2" not in search_policy("O3a", apply_cat2=False).categories
        assert "CBC_CAT2" in search_policy("O4a", apply_cat2=True).categories

    def test_training_policy_requires_data_only(self):
        """
        The training releases select on data presence alone.

        Recorded so an audit can state precisely how a training dataset differs from a
        search one rather than inferring it.
        """
        policy = training_policy()
        assert policy.require_data
        assert policy.categories == ()
        assert policy.injection_flags == ()

    def test_the_two_policies_differ(self):
        assert search_policy("O3a") != training_policy()

    def test_required_flags_are_detector_prefixed(self):
        names = search_policy("O3a").required_flags("H1", _run_flags())
        assert "H1_DATA" in names
        assert "H1_CBC_CAT1" in names
        assert "H1_NO_CBC_HW_INJ" in names

    def test_required_flags_are_validated_against_the_run(self):
        policy = FlagPolicy(categories=("CBC_CAT9",))
        with pytest.raises(ValueError):
            policy.required_flags("H1", _run_flags())

    def test_describe_states_what_is_required(self):
        text = search_policy("O3a").describe()
        assert "CBC_CAT1" in text
        assert "NO_CBC_HW_INJ" in text

    def test_describe_mentions_what_is_deliberately_not_required(self):
        """The methods section should say that continuous injections are kept."""
        assert "continuous" in search_policy("O3a").describe().lower()

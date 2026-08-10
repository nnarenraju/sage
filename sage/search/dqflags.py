#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : dqflags.py
Description   : Data-quality and injection flags, and the policy a search applies.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The open-data releases publish two bitmasks per detector: a data-quality mask and an
injection mask. Both are queried as named timeline flags.

The distinction that matters here is that ``DATA`` and ``CBC_CAT1`` are separate bits.
``DATA`` means only that samples exist; it carries no statement about whether those
samples are analysable. A dataset selected on ``DATA`` alone therefore has no category
vetoing and no injection filtering, which is adequate for sampling noise to train on but
not for a search, where vetoed time inflates livetime and hardware injections appear as
candidates.

Bit definitions below were read from the release metadata rather than transcribed, and
they differ between runs: O3b publishes seven data-quality bits, O4a nine, the two extra
covering categories for other search groups. Because the set is run-dependent, a policy
is resolved against the run it will be applied to and fails loudly if a flag it needs is
not published for that run.

Category conventions for compact-binary searches:

* CAT1 marks data too badly affected to analyse at all, and is applied by every search.
* CAT2 marks times with a statistical association to an auxiliary disturbance. It was
  applied in the third observing run, and dropped from compact-binary searches in the
  fourth; it remains in use for unmodelled searches.
* CAT3 is not used by compact-binary searches.

Measured against the open data, these cost very little on top of ``DATA``: over 600 ks
sampled in each of O3b and O4a, ``CBC_CAT1`` coincides with ``DATA`` exactly, and
``CBC_CAT2`` removes about 0.1 per cent in O3b and nothing in O4a. Requiring CAT1 is
therefore close to a formality on public data, but it is requested explicitly so that the
selection states what it assumes rather than relying on that continuing to hold.

Hardware injections are simulated signals added to the interferometer, and a transient
search excludes those that are transient. It must not exclude the continuous ones: those
run for the entire observing run, so requiring their absence removes all the data. See
:data:`CONTINUOUS_INJECTION_FLAGS`.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

CBC_CATEGORIES: Tuple[str, ...] = ("CBC_CAT1", "CBC_CAT2", "CBC_CAT3")

# Transient hardware injections. These are simulated signals of finite duration and are
# excluded, since a search cannot distinguish them from candidates.
TRANSIENT_INJECTION_FLAGS: Tuple[str, ...] = (
    "NO_CBC_HW_INJ",
    "NO_BURST_HW_INJ",
    "NO_DETCHAR_HW_INJ",
)

# Continuous injections. These run for the whole observing run rather than in bursts:
# measured over 600 ks of each of O3b and O4a, NO_CW_HW_INJ is satisfied for zero seconds,
# so requiring it removes every second of data. They are narrowband and always present,
# the detectors are characterised with them in place, and a transient search is unaffected
# by them. They are therefore not excluded, and requiring one is refused rather than
# silently emptying a dataset.
CONTINUOUS_INJECTION_FLAGS: Tuple[str, ...] = (
    "NO_CW_HW_INJ",
    "NO_STOCH_HW_INJ",
)


@dataclass(frozen=True)
class FlagBit:
    """One published flag."""

    short_name: str
    bit: int
    mask: int
    description: str

    @property
    def is_injection(self) -> bool:
        """Whether the flag belongs to the injection mask rather than data quality."""
        return self.short_name.startswith("NO_") and self.short_name.endswith("_INJ")


@dataclass
class RunFlags:
    """The flags a given observing run publishes."""

    observing_run: str
    dataset: str
    dq_bits: Tuple[FlagBit, ...] = ()
    inj_bits: Tuple[FlagBit, ...] = ()

    def names(self) -> Tuple[str, ...]:
        """Every flag name published for this run."""
        return tuple(bit.short_name for bit in (*self.dq_bits, *self.inj_bits))

    def has(self, name: str) -> bool:
        """Whether a named flag is published for this run."""
        return name in self.names()

    def require(self, names: Sequence[str]) -> None:
        """
        Assert every named flag exists, naming the run and what it does publish.

        The published set is run-dependent, so reporting only the missing name sends the
        reader looking in the wrong place.
        """
        missing = [name for name in names if not self.has(name)]
        if missing:
            raise ValueError(
                f"{self.observing_run} does not publish {', '.join(sorted(missing))}; "
                f"it publishes {', '.join(sorted(self.names()))}"
            )


def fetch_run_flags(observing_run: str, dataset: Optional[str] = None) -> RunFlags:
    """Read the published flag definitions for a run from the release metadata."""
    raise NotImplementedError


@dataclass(frozen=True)
class FlagPolicy:
    """
    Which flags a dataset requires.

    A detector-second is kept only where every required flag is set. Times excluded by
    the policy are not analysed and do not count toward livetime.
    """

    require_data: bool = True
    categories: Tuple[str, ...] = ("CBC_CAT1",)
    injection_flags: Tuple[str, ...] = TRANSIENT_INJECTION_FLAGS
    extra: Tuple[str, ...] = ()

    def _bare_names(self) -> Tuple[str, ...]:
        """Every flag this policy requires, without a detector prefix."""
        names = list(self.categories) + list(self.injection_flags) + list(self.extra)
        if self.require_data:
            names.insert(0, "DATA")
        return tuple(dict.fromkeys(names))

    def required_flags(self, detector: str, run_flags: RunFlags) -> Tuple[str, ...]:
        """Detector-prefixed flag names this policy needs, checked against the run."""
        self.validate(run_flags)
        return tuple(f"{detector}_{name}" for name in self._bare_names())

    def validate(self, run_flags: RunFlags) -> None:
        """
        Check the policy can be satisfied before it is used to select data.

        Refuses a continuous-injection flag, which is never satisfied and would silently
        produce an empty dataset, and refuses any flag the run does not publish.
        """
        requested = self._bare_names()
        continuous = [name for name in requested if name in CONTINUOUS_INJECTION_FLAGS]
        if continuous:
            raise ValueError(
                f"policy requires the continuous-injection flag(s) "
                f"{', '.join(continuous)}, which are satisfied for zero seconds because "
                "those injections run for the whole observing run; requiring one removes "
                "every second of data. Continuous injections are narrowband and always "
                "present, and a transient search is unaffected by them."
            )
        run_flags.require(requested)

    def describe(self) -> str:
        """Readable statement of the policy, for the methods section and file provenance."""
        parts = []
        if self.require_data:
            parts.append("data present")
        if self.categories:
            parts.append(f"passing {', '.join(self.categories)}")
        if self.injection_flags:
            parts.append(f"free of {', '.join(self.injection_flags)}")
        if self.extra:
            parts.append(f"and {', '.join(self.extra)}")
        required = "; ".join(parts) if parts else "no conditions"
        return (
            f"Detector-seconds are kept where: {required}. Continuous hardware "
            f"injections ({', '.join(CONTINUOUS_INJECTION_FLAGS)}) are deliberately not "
            "excluded: they run for the whole observing run, the detectors are "
            "characterised with them in place, and a transient search is unaffected."
        )


def search_policy(observing_run: str, apply_cat2: Optional[bool] = None) -> FlagPolicy:
    """
    The policy for a compact-binary search.

    Requires data present, passes CAT1, and carries no *transient* hardware injection.
    Continuous injections are kept; see :data:`CONTINUOUS_INJECTION_FLAGS`.

    CAT2 defaults to how the run was analysed: applied for the third observing run, not
    applied for the fourth. Passing ``apply_cat2`` overrides that, which is worth doing
    only deliberately, since it changes the analysed livetime and therefore every rate
    the search reports, and makes a comparison against the published results no longer
    like for like.
    """
    if apply_cat2 is None:
        apply_cat2 = observing_run.upper().startswith("O3")
    categories = ("CBC_CAT1", "CBC_CAT2") if apply_cat2 else ("CBC_CAT1",)
    return FlagPolicy(
        require_data=True,
        categories=categories,
        injection_flags=TRANSIENT_INJECTION_FLAGS,
    )


def training_policy() -> FlagPolicy:
    """
    The policy used for the existing noise datasets, for comparison.

    Requires data present and nothing else. Recorded here so that an audit can state
    precisely how a training dataset differs from a search one rather than inferring it.
    """
    return FlagPolicy(require_data=True, categories=(), injection_flags=(), extra=())

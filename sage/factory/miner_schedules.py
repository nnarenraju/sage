#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Threshold schedules for SageCurriculumTraining hard-noise miners.

All schedules are callable objects::

    schedule(epoch) -> (explore_threshold, refine_threshold)

Pass one to ``SageCurriculumTraining(threshold_schedule=...)``.
Passing ``None`` (the default) keeps each miner's threshold constant.

Available schedules
-------------------
LinearThresholdSchedule   — constant ramp from start to end values
CosineThresholdSchedule   — cosine ramp (slow start, fast middle, slow end)
StepThresholdSchedule     — explicit jumps at milestone epochs
"""

import math


class LinearThresholdSchedule:
    """
    Linearly ramp both thresholds from start values to end values over
    the post-warmup training period.

    Parameters
    ----------
    total_epochs : int
    warmup_epochs : int
        Epochs before mining starts.  Thresholds stay at start values until
        then.
    explore_start, explore_end : float
        Threshold range for the GPS explorer (CMAMEMiner).
    refine_start, refine_end : float
        Threshold range for the pattern refiner (CMAMEGAMiner).

    Example
    -------
    >>> sched = LinearThresholdSchedule(total_epochs=80, warmup_epochs=10)
    >>> sched(10)   # (3.0, 5.0) — start of ramp
    (3.0, 5.0)
    >>> sched(79)   # (6.0, 8.0) — end of ramp
    (6.0, 8.0)
    """

    def __init__(
        self,
        total_epochs:  int,
        warmup_epochs: int,
        explore_start: float = 3.0,
        explore_end:   float = 6.0,
        refine_start:  float = 5.0,
        refine_end:    float = 8.0,
    ):
        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs
        self.explore_start = explore_start
        self.explore_end   = explore_end
        self.refine_start  = refine_start
        self.refine_end    = refine_end
        self._post = max(total_epochs - warmup_epochs - 1, 1)

    def __call__(self, epoch: int):
        t = min(max(epoch - self.warmup_epochs, 0) / self._post, 1.0)
        return (
            self.explore_start + t * (self.explore_end - self.explore_start),
            self.refine_start  + t * (self.refine_end  - self.refine_start),
        )

    def __repr__(self):
        return (
            f"LinearThresholdSchedule("
            f"explore {self.explore_start}→{self.explore_end}, "
            f"refine {self.refine_start}→{self.refine_end}, "
            f"epochs {self.warmup_epochs}–{self.total_epochs})"
        )


class CosineThresholdSchedule:
    """
    Cosine ramp: slow increase at the start, fast in the middle, slow at the
    end.  Useful when you want a gentle entry into hard mining and a plateau
    near the maximum threshold toward the end of training.

    Parameters are identical to LinearThresholdSchedule.

    Example
    -------
    >>> sched = CosineThresholdSchedule(total_epochs=80, warmup_epochs=10)
    >>> sched(10)   # (3.0, 5.0) — start
    (3.0, 5.0)
    >>> sched(44)   # midpoint — close to halfway between start and end
    (4.5, 6.5)
    >>> sched(79)   # (6.0, 8.0) — end
    (6.0, 8.0)
    """

    def __init__(
        self,
        total_epochs:  int,
        warmup_epochs: int,
        explore_start: float = 3.0,
        explore_end:   float = 6.0,
        refine_start:  float = 5.0,
        refine_end:    float = 8.0,
    ):
        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs
        self.explore_start = explore_start
        self.explore_end   = explore_end
        self.refine_start  = refine_start
        self.refine_end    = refine_end
        self._post = max(total_epochs - warmup_epochs - 1, 1)

    def __call__(self, epoch: int):
        raw_t = min(max(epoch - self.warmup_epochs, 0) / self._post, 1.0)
        t = (1.0 - math.cos(math.pi * raw_t)) / 2.0
        return (
            self.explore_start + t * (self.explore_end - self.explore_start),
            self.refine_start  + t * (self.refine_end  - self.refine_start),
        )

    def __repr__(self):
        return (
            f"CosineThresholdSchedule("
            f"explore {self.explore_start}→{self.explore_end}, "
            f"refine {self.refine_start}→{self.refine_end}, "
            f"epochs {self.warmup_epochs}–{self.total_epochs})"
        )


class StepThresholdSchedule:
    """
    Jump thresholds at explicit milestone epochs.

    The active thresholds are those from the latest step whose epoch index
    is ≤ the current epoch.  Before the first step fires, ``default_explore``
    and ``default_refine`` are used.

    Parameters
    ----------
    steps : list of (int, float, float)
        Each tuple is ``(epoch, explore_thresh, refine_thresh)``.  The list
        does not need to be sorted.
    default_explore : float
        Explore threshold before the first step fires.
    default_refine : float
        Refine threshold before the first step fires.

    Example
    -------
    >>> sched = StepThresholdSchedule(
    ...     steps=[
    ...         (10, 3.0, 5.0),
    ...         (25, 4.5, 6.5),
    ...         (50, 6.0, 8.0),
    ...     ]
    ... )
    >>> sched(5)    # (3.0, 5.0) — default, no step fired yet
    (3.0, 5.0)
    >>> sched(10)   # (3.0, 5.0) — first step fires
    (3.0, 5.0)
    >>> sched(30)   # (4.5, 6.5) — second step fired
    (4.5, 6.5)
    >>> sched(60)   # (6.0, 8.0) — third step fired
    (6.0, 8.0)
    """

    def __init__(
        self,
        steps,
        default_explore: float = 3.0,
        default_refine:  float = 5.0,
    ):
        self._steps          = sorted(steps, key=lambda x: x[0])
        self.default_explore = default_explore
        self.default_refine  = default_refine

    def __call__(self, epoch: int):
        explore = self.default_explore
        refine  = self.default_refine
        for step_epoch, step_explore, step_refine in self._steps:
            if epoch >= step_epoch:
                explore = step_explore
                refine  = step_refine
            else:
                break
        return explore, refine

    def __repr__(self):
        return f"StepThresholdSchedule(steps={self._steps})"

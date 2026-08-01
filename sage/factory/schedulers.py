#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : schedulers.py
Description     : Short description of the file

Created on 2026-03-06 16:29:32

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


class ManageScheduler:
    """
    Unified adapter for PyTorch learning-rate schedulers with multiple
    step-trigger modes.

    Different scheduler types require ``step()`` to be called at different
    times (per batch, per epoch, or after a metric update).  This wrapper
    standardises the interface by routing :meth:`batch_step` and
    :meth:`epoch_step` calls to the underlying scheduler according to the
    configured ``mode``.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The underlying scheduler instance.
    mode : str
        When to advance the scheduler:

        * ``"batch"`` — call ``scheduler.step()`` on every batch.
        * ``"fractional"`` — call ``scheduler.step(epoch + batch/total)``
          for schedulers that accept a fractional epoch argument.
        * ``"epoch"`` — call ``scheduler.step()`` once per epoch (in
          :meth:`epoch_step`).
        * ``"metric"`` — call ``scheduler.step(metric)`` once per epoch
          (e.g. :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`).
    """

    def __init__(self, scheduler, mode="batch"):
        self.scheduler = scheduler
        self.mode = mode

    def batch_step(self, nepoch=None, nbatch=None, num_batches=None):
        """
        Advance the scheduler at the end of a batch.

        Only takes effect when ``mode`` is ``"batch"`` or ``"fractional"``.

        Parameters
        ----------
        nepoch : int
            Current epoch index (required for ``"fractional"`` mode).
        nbatch : int
            Current batch index within the epoch (required for ``"fractional"``).
        num_batches : int
            Total number of batches per epoch (required for ``"fractional"``).
        """

        if self.mode == "batch":
            self.scheduler.step()

        elif self.mode == "fractional":
            self.scheduler.step(nepoch + nbatch / num_batches)

    def epoch_step(self, metric=None):
        """
        Advance the scheduler at the end of an epoch.

        Only takes effect when ``mode`` is ``"epoch"`` or ``"metric"``.

        Parameters
        ----------
        metric : float or None
            Validation metric value (required for ``"metric"`` mode, e.g.
            validation loss passed to :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`).
        """

        if self.mode == "epoch":
            self.scheduler.step()

        elif self.mode == "metric":
            self.scheduler.step(metric)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : schedulers.py
Description     : Short description of the file

Created on 2026-03-06 16:29:32

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


class ManageScheduler:

    def __init__(self, scheduler, mode="batch"):
        self.scheduler = scheduler
        self.mode = mode

    def batch_step(self, nepoch=None, nbatch=None, num_batches=None):

        if self.mode == "batch":
            self.scheduler.step()

        elif self.mode == "fractional":
            self.scheduler.step(nepoch + nbatch / num_batches)

    def epoch_step(self, metric=None):

        if self.mode == "epoch":
            self.scheduler.step()

        elif self.mode == "metric":
            self.scheduler.step(metric)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : crop.py
Description   : Short description of the file

Created on 2026-01-19 16:35:49

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


class Crop(TransformWrapperPerChannel):
    # Crop the signal if required
    # If not whitening, we use this to emulate the remove_corrupted = True option
    def __init__(
        self,
        always_apply=True,
        croplen=0.0,
        emulate_rmcorrupt=False,
        double_sided=True,
        side="left",
    ):
        super().__init__(always_apply)
        self.emulate_rmcorrupt = emulate_rmcorrupt
        self.croplen = croplen
        self.double_sided = double_sided
        self.side = side

    def get_cropped(self, y, data_cfg):
        if self.emulate_rmcorrupt:
            self.croplen = int(round(data_cfg.whiten_padding * data_cfg.sample_rate))
        if self.double_sided:
            cropped = y[int(self.croplen / 2) : int(len(y) - self.croplen / 2)]
        else:
            if self.side == "left":
                cropped = y[int(self.croplen) :]
            elif self.side == "right":
                cropped = y[: int(len(y) - self.croplen)]
            else:
                raise ValueError("Crop Transform: side not recognised!")
        return cropped

    def apply(self, y: np.ndarray, channel: int, special: dict):
        return self.get_cropped(y, special["data_cfg"])

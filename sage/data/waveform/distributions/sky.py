#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sky-position prior uniform over the sphere.

Filename        : sky.py
Description     : Short description of the file

Created on 2026-02-16 10:53:06

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

import torch

from .angular import UniformAngle, CosAngle


class UniformSky:
    """
    Sky position uniform over the sphere.

    :class:`CosAngle` for the declination and :class:`UniformAngle` for
    the right ascension.

    Parameters
    ----------
    polar_name : str
        Key of the declination in the draw.
    azimuthal_name : str
        Key of the right ascension in the draw.
    polar_bounds : tuple[float, float]
        Declination bounds in radians.
    azimuthal_bounds : tuple[float, float]
        Right-ascension bounds in radians.
    """

    def __init__(
        self,
        polar_name="dec",
        azimuthal_name="ra",
        polar_bounds=(-torch.pi / 2, torch.pi / 2),
        azimuthal_bounds=(0.0, 2 * torch.pi),
    ):
        self.polar_name = polar_name
        self.azimuthal_name = azimuthal_name

        self.polar_sampler = CosAngle(*polar_bounds)
        self.azimuth_sampler = UniformAngle(*azimuthal_bounds)

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Declination and right ascension, keyed by their names.

        Parameters
        ----------
        shape : tuple of int
            Shape of the draw.
        device : str or torch.device or None
            Device of the draw.
        dtype : torch.dtype
            Dtype of the draw.
        generator : torch.Generator or None
            Generator of the draw.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{polar_name: dec, azimuthal_name: ra}``, each of shape
            ``shape``.
        """
        dec = self.polar_sampler.sample(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        ra = self.azimuth_sampler.sample(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        return {
            self.polar_name: dec,
            self.azimuthal_name: ra,
        }

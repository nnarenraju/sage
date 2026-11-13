#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Angle priors: uniform, sin and cos angles, and a solid angle uniform
over the sphere.

Filename        : angular.py
Description     : Short description of the file

Created on 2026-02-16 10:45:51

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

from .uniform import Uniform


class UniformAngle:
    """
    Angle uniform on ``[lower, upper)``.

    Parameters
    ----------
    lower : float
        Lower bound in radians.
    upper : float
        Upper bound in radians.
    cyclic_domain : bool
        Whether to wrap the samples into ``[lower, upper)``.
    """

    def __init__(self, lower=0.0, upper=2 * torch.pi, cyclic_domain=True):
        self.lower = lower
        self.upper = upper
        self.cyclic = cyclic_domain
        self.range = upper - lower

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Angles drawn uniformly from ``[lower, upper)``.

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
        torch.Tensor
            Angles in radians, of shape ``shape``.
        """
        x = (
            torch.rand(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * self.range
            + self.lower
        )
        if self.cyclic:
            x = self.wrap(x)
        return x

    def wrap(self, x):
        """
        ``x`` wrapped into ``[lower, upper)``.

        Parameters
        ----------
        x : torch.Tensor
            Angles in radians.

        Returns
        -------
        torch.Tensor
            The wrapped angles.
        """
        return (x - self.lower) % self.range + self.lower


class SinAngle(UniformAngle):
    """
    Polar angle with PDF proportional to sin(theta) on ``[low, high]``.

    The isotropic prior of an inclination-like angle, drawn by the
    inverse CDF uniform in ``cos(theta)``.

    Parameters
    ----------
    low : float
        Lower bound in radians (default ``0``).
    high : float
        Upper bound in radians (default ``pi``).
    """

    PI = torch.pi

    def __init__(self, low=0.0, high=torch.pi):
        self.low = float(low)
        self.high = float(high)

        self.cos_low = torch.cos(torch.tensor(self.high))
        self.cos_high = torch.cos(torch.tensor(self.low))

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Polar angles drawn from the sin-angle prior.

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
        torch.Tensor
            Angles in radians, of shape ``shape``.
        """
        # uniform in cos(theta)
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        cos_theta = u * (self.cos_high - self.cos_low) + self.cos_low

        # inverse CDF
        theta = torch.arccos(cos_theta)

        return theta


class CosAngle(SinAngle):
    """
    Angle with PDF proportional to ``cos(theta)`` on ``[low, high]``.

    The isotropic prior of a declination-like angle, drawn by the
    inverse CDF uniform in ``sin(theta)``.

    Parameters
    ----------
    low : float
        Lower bound in radians (default ``-pi/2``).
    high : float
        Upper bound in radians (default ``pi/2``).
    """

    HALF_PI = torch.pi / 2

    def __init__(self, low=-torch.pi / 2, high=torch.pi / 2):
        self.low = float(low)
        self.high = float(high)

        self.sin_low = torch.sin(torch.tensor(self.low))
        self.sin_high = torch.sin(torch.tensor(self.high))

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Declination-like angles drawn from the cos-angle prior.

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
        torch.Tensor
            Angles in radians, of shape ``shape``.
        """
        # uniform in sin(theta)
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        sin_theta = u * (self.sin_high - self.sin_low) + self.sin_low

        # inverse CDF
        theta = torch.arcsin(sin_theta)

        return theta


class UniformSolidAngle:
    """
    Pair of angles uniform over the sphere.

    :class:`SinAngle` for the polar angle and :class:`UniformAngle` for
    the azimuthal angle.

    Parameters
    ----------
    polar_name : str
        Key of the polar angle in the draw.
    azimuthal_name : str
        Key of the azimuthal angle in the draw.
    polar_bounds : tuple[float, float]
        ``(low, high)`` of the polar angle in radians.
    azimuthal_bounds : tuple[float, float]
        ``(low, high)`` of the azimuthal angle in radians.
    """

    def __init__(
        self,
        polar_name="theta",
        azimuthal_name="phi",
        polar_bounds=(0.0, torch.pi),
        azimuthal_bounds=(0.0, 2 * torch.pi),
    ):
        self.polar_name = polar_name
        self.azimuthal_name = azimuthal_name

        self.polar_sampler = SinAngle(*polar_bounds)
        self.azimuth_sampler = UniformAngle(*azimuthal_bounds)

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Polar and azimuthal angles, keyed by their names.

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
            ``{polar_name: theta, azimuthal_name: phi}``, each of shape
            ``shape``.
        """
        theta = self.polar_sampler.sample(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        phi = self.azimuth_sampler.sample(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        return {
            self.polar_name: theta,
            self.azimuthal_name: phi,
        }

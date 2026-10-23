#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Packages
import torch

# LOCAL
from .uniform import Uniform


class UniformAngle:
    """GPU-friendly uniform distribution with optional arbitrary bounds and cyclic domain."""

    def __init__(self, lower=0.0, upper=2 * torch.pi, cyclic_domain=True):
        """
        Args:
            lower (float): lower bound (radians)
            upper (float): upper bound (radians)
            cyclic_domain (bool): whether to wrap samples into [lower, upper)
        """
        self.lower = lower
        self.upper = upper
        self.cyclic = cyclic_domain
        self.range = upper - lower

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """Draw *shape* samples uniformly from ``[lower, upper)``."""
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
        """Wrap values into [lower, upper)"""
        return (x - self.lower) % self.range + self.lower


class SinAngle(UniformAngle):
    """
    Isotropic polar-angle sampler for inclination-like parameters.

    Draws samples from the distribution whose PDF is proportional to
    ``sin(theta)`` over ``[low, high]``.  This is the correct prior for an
    angle that is uniform over a sphere (e.g. binary inclination, source sky
    polar angle).  Sampling is done via the inverse-CDF method in
    ``cos(theta)`` space.

    Parameters
    ----------
    low : float
        Lower bound in radians (default ``0``).
    high : float
        Upper bound in radians (default ``pi``).
    """

    PI = torch.pi

    def __init__(self, low=0.0, high=torch.pi):
        # domain [0, pi]
        self.low = float(low)
        self.high = float(high)

        self.cos_low = torch.cos(torch.tensor(self.high))
        self.cos_high = torch.cos(torch.tensor(self.low))

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """Draw *shape* polar angles with the sin-angle prior (isotropic sphere)."""
        # uniform in cos(theta)
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        cos_theta = u * (self.cos_high - self.cos_low) + self.cos_low

        # inverse CDF
        theta = torch.arccos(cos_theta)

        return theta


class CosAngle(SinAngle):
    """
    Isotropic azimuthal-like angle sampler for declination-type parameters.

    Draws samples whose PDF is proportional to ``cos(theta)`` over
    ``[-pi/2, pi/2]`` — the correct isotropic prior for declination or
    elevation angles.  Sampling uses the inverse-CDF method in ``sin(theta)``
    space, which is the dual of :class:`SinAngle`.

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
        """Draw *shape* declination angles with the cos-angle prior (isotropic sphere)."""
        # uniform in sin(theta)
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        sin_theta = u * (self.sin_high - self.sin_low) + self.sin_low

        # inverse CDF
        theta = torch.arcsin(sin_theta)

        return theta


class UniformSolidAngle:
    """
    Joint sampler for a pair of angles that is uniform over the full sphere.

    Combines :class:`SinAngle` for the polar angle (inclination / sky
    colatitude) and :class:`UniformAngle` for the azimuthal angle (right
    ascension / polarisation) to produce an isotropic orientation prior.
    The ``sample`` method returns a dict keyed by the names given at
    construction, making it easy to merge into a broader parameter dict.

    Parameters
    ----------
    polar_name : str
        Key for the polar-angle output (default ``"theta"``).
    azimuthal_name : str
        Key for the azimuthal-angle output (default ``"phi"``).
    polar_bounds : tuple[float, float]
        ``(low, high)`` in radians for the polar angle (default ``(0, pi)``).
    azimuthal_bounds : tuple[float, float]
        ``(low, high)`` in radians for the azimuthal angle
        (default ``(0, 2*pi)``).
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

        # reuse your fast samplers
        self.polar_sampler = SinAngle(*polar_bounds)
        self.azimuth_sampler = UniformAngle(*azimuthal_bounds)

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Draw *shape* angle pairs and return them as a dict.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{polar_name: theta, azimuthal_name: phi}`` each of shape *shape*.
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : cosmology.py
Description     : Cosmology-aware luminosity-distance distributions.

Created on 2026-06-24

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__status__        = inProgress
"""

# Packages
import numpy as np
import torch


class UniformComovingVolume:
    """
    Luminosity-distance sampler uniform in COMOVING volume.

    Draws sources with constant number density per unit comoving volume
    between luminosity-distance bounds ``[low, high]`` (Mpc) and returns their
    luminosity distance.  Unlike
    :class:`~sage.data.waveform.distributions.powerlaw.UniformRadius`
    (uniform in *Euclidean* volume, ``p(r) ~ r^2`` in luminosity distance),
    this accounts for the cosmological volume element ``dV_c/dz`` via
    ``astropy`` and is the standard isotropic-homogeneous astrophysical
    source-distribution prior.

    A monotonic comoving-volume -> luminosity-distance lookup table is built
    once at construction with ``astropy``; sampling then inverts it with torch
    ops, so the draw stays GPU- and ``torch.Generator``-aware exactly like the
    other Sage distributions.

    Parameters
    ----------
    low, high : float
        Luminosity-distance bounds in Mpc.
    cosmology : str
        Name of a realised ``astropy.cosmology`` instance (default
        ``"Planck18"``).
    n_grid : int
        Number of redshift grid points in the lookup table (default 4096).
    """

    name = "uniform_comoving_volume"

    def __init__(self, low, high, cosmology="Planck18", n_grid=4096):
        import astropy.cosmology as ac
        from astropy.cosmology import z_at_value
        from astropy import units as u

        cosmo = getattr(ac, cosmology)

        z_lo = float(z_at_value(cosmo.luminosity_distance, low * u.Mpc))
        z_hi = float(z_at_value(cosmo.luminosity_distance, high * u.Mpc))

        zg = np.linspace(z_lo, z_hi, n_grid)
        Vg = cosmo.comoving_volume(zg).to(u.Mpc ** 3).value      # monotonic in z
        dLg = cosmo.luminosity_distance(zg).to(u.Mpc).value

        self.low = float(low)
        self.high = float(high)
        self._Vg = torch.as_tensor(Vg, dtype=torch.float64)
        self._dLg = torch.as_tensor(dLg, dtype=torch.float64)
        self._Vmin = float(Vg[0])
        self._Vmax = float(Vg[-1])

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """Draw luminosity distances (Mpc) uniform in comoving volume."""
        u = torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
        V = self._Vmin + (self._Vmax - self._Vmin) * u          # uniform in V_comoving

        Vg = self._Vg.to(V.device)
        dLg = self._dLg.to(V.device)

        idx = torch.searchsorted(Vg, V).clamp(1, Vg.numel() - 1)
        V0, V1 = Vg[idx - 1], Vg[idx]
        d0, d1 = dLg[idx - 1], dLg[idx]
        w = (V - V0) / (V1 - V0)
        dL = d0 + w * (d1 - d0)                                  # invert V -> d_L

        return dL.to(device=device, dtype=dtype)

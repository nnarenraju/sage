#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Luminosity-distance prior uniform in comoving volume.

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

import warnings

import numpy as np
import torch


class UniformComovingVolume:
    """
    Luminosity distance drawn uniform in comoving volume.

    A comoving-volume to luminosity-distance table on a redshift grid is
    built with ``astropy``; a draw uniform in comoving volume is mapped
    to distance by linear interpolation in that table.

    Parameters
    ----------
    low, high : float
        Luminosity-distance bounds in Mpc.
    cosmology : str
        Name of a realised ``astropy.cosmology`` instance.
    n_grid : int
        Number of redshift grid points in the table.
    """

    name = "uniform_comoving_volume"

    def __init__(self, low, high, cosmology="Planck18", n_grid=4096):
        # Deferred: astropy is heavy and only this prior needs it
        import astropy.cosmology as ac
        from astropy.cosmology import z_at_value
        from astropy import units as u

        warnings.warn(
            "uniform_comoving_volume is not a PyCBC distribution (PyCBC draws uniform comoving_volume with "
            "Planck15 and transforms it to distance); its distance inversion interpolates a table "
            "(1e-6 relative), so draws may pass max by ~1e-6 Mpc"
        )
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
        """
        Luminosity distances in Mpc, uniform in comoving volume.

        Parameters
        ----------
        shape : tuple of int
            Shape of the draw.
        device : str or torch.device or None
            Device of the draw.
        dtype : torch.dtype
            Dtype of the returned distances; the draw is float64.
        generator : torch.Generator or None
            Generator of the draw.

        Returns
        -------
        torch.Tensor
            Distances of shape ``shape``.
        """
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : project.py
Description     : Short description of the file

Created on 2026-01-23 16:17:38

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

# Packages
import torch

# LOCAL
from sage.core.constants import C


class Detector:
    """
    Minimal detector container for FD projection.
    All tensors should live on the same device as waveforms.
    """

    def __init__(
        self,
        detector_name,
        reference_time=1126259462.423,  # GPS seconds
    ):
        # Detector tensor
        self.response = response
        # Detector position (ECEF)
        self.loc = location
        self.longitude = longitude
        self.latitude = latitude

        # Reference times
        self.gmst_reference = None
        # GPS time of GW150914 by default
        self.reference_time = reference_time

    def antenna_pattern(
        self,
        right_ascension,
        declination,
        polarization,
        reference_time,
    ):
        """Return the detector response.

        Parameters
        ----------
        right_ascension: float or numpy.ndarray
            The right ascension of the source
        declination: float or numpy.ndarray
            The declination of the source
        polarization: float or numpy.ndarray
            The polarization angle of the source
        polarization_type: string flag: Tensor, Vector or Scalar
            The gravitational wave polarizations. Default: 'Tensor'

        Returns
        -------
        fplus(default) or fx or fb : float or numpy.ndarray
            The plus or vector-x or breathing polarization factor for this sky location / orientation
        fcross(default) or fy or fl : float or numpy.ndarray
            The cross or vector-y or longitudnal polarization factor for this sky location / orientation
        """

        gha = self.gmst_estimate(reference_time) - right_ascension

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(declination)
        sindec = torch.sin(declination)
        cospsi = torch.cos(polarization)
        sinpsi = torch.sin(polarization)

        resp = self.response
        ttype = torch.float64

        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 = sinpsi * cosdec

        x = np.array([x0, x1, x2], dtype=object)
        dx = resp.dot(x)

        y0 = sinpsi * singha - cospsi * cosgha * sindec
        y1 = sinpsi * cosgha + cospsi * singha * sindec
        y2 = cospsi * cosdec

        y = np.array([y0, y1, y2], dtype=object)
        dy = resp.dot(y)

        if hasattr(dx, "shape"):
            fplus = (x * dx - y * dy).sum(axis=0).astype(ttype)
            fcross = (x * dy + y * dx).sum(axis=0).astype(ttype)
        else:
            fplus = (x * dx - y * dy).sum()
            fcross = (x * dy + y * dx).sum()

        return fplus, fcross

    def constant_project(self, hp, hc, ra, dec, polarization):
        """Return the strain of a waveform as measured by the detector.
        Apply the time shift for the given detector relative to the assumed
        geocentric frame and apply the antenna patterns to the plus and cross
        polarizations.

        Parameters
        ----------
        hp: pycbc.types.TimeSeries
            Plus polarization of the GW
        hc: pycbc.types.TimeSeries
            Cross polarization of the GW
        ra: float
            Right ascension of source location
        dec: float
            Declination of source location
        polarization: float
            Polarization angle of the source
        """
        # 'constant' assume fixed orientation relative to source over the
        # duration of the signal, accurate for short duration signals
        fp, fc = antenna_pattern(ra, dec, polarization, self.reference_time)
        dt = time_delay_from_earth_center(ra, dec, self.reference_time)
        ts = fp * hp + fc * hc
        ts.start_time = float(ts.start_time) + dt
        return ts

    def gmst_estimate(self, gps_time):
        if self.reference_time is None:
            return gmst_accurate(gps_time)

        if self.gmst_reference is None:
            self.set_gmst_reference()
        dphase = (gps_time - self.reference_time) / self.sday * (2.0 * np.pi)
        gmst = (self.gmst_reference + dphase) % (2.0 * np.pi)
        return gmst

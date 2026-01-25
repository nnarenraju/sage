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
from sage.core.constants import PI, C
from sage.core.math import rotation_matrix


class Detector:
    """
    Minimal detector container for FD projection.
    All tensors should live on the same device as waveforms.
    """

    def __init__(self, detector_names, device="cuda"):
        # CUDA device
        self.device = device
        # Detector
        self.detector_names = detector_names
        # Detector tensor
        # TODO: Set detector data as buffer once
        # self.response shape = (batch_size, num_dets, response)
        self.response = response
        # Detector position (ECEF)
        location = location
        # Earth center and relative positions of dets
        earth_center = torch.stack([0, 0, 0], device=self.device)
        self.dx = earth_center - self.location

    def get_detector_response(
        self,
        longitude,
        latitude,
        yangle=0,
        xangle=None,
        xaltitude=0,
        yaltitude=0,
    ):
        """Add a new detector on the earth

        Parameters
        ----------
        longitude: float
            Longitude in radians using geodetic coordinates of the detector
        latitude: float
            Latitude in radians using geodetic coordinates of the detector
        yangle: float
            Azimuthal angle of the y-arm (angle drawn from pointing north)
        xangle: float
            Azimuthal angle of the x-arm (angle drawn from point north). If not set
            we assume a right angle detector following the right-hand rule.
        xaltitude: float
            The altitude angle of the x-arm measured from the local horizon.
        yaltitude: float
            The altitude angle of the y-arm measured from the local horizon.

        """
        # Baseline response of a single arm pointed in the -X direction
        resp = torch.stack([[-1, 0, 0], [0, 0, 0], [0, 0, 0]], device=self.device)
        # Latitude and longitude provided in radians
        # {x,y,z} -> {0,1,2}
        rm2 = rotation_matrix(-longitude, 2)
        rm1 = rotation_matrix(-1.0 * (PI / 2.0 - latitude), 1)

        # Calculate response in earth centered coordinates
        # by rotation of response in coordinates aligned
        # with the detector arms
        resps = []
        # Only computed once; so for loop is fine
        for angle, azi in [(yangle, yaltitude), (xangle, xaltitude)]:
            # {x,y,z} -> {0,1,2}
            rm0 = rotation_matrix(angle, 2)
            rmN = rotation_matrix(-azi, 1)
            rm = rm2 @ rm1 @ rm0 @ rmN
            # apply rotation
            resps.append(rm @ resp @ rm.T / 2.0)

        full_response = resps[0] - resps[1]
        return full_response

    def random_gmst_estimate(self, batch_shape):
        # Random GMST in radians to compute the antenna patterns
        # Reference times to GMST requires table reads and is expensive
        # Instead we simply randomise GMST in [0, 2PI)
        return 2 * PI * torch.rand(batch_shape, device=self.device)

    def antenna_pattern(
        self, right_ascension, declination, polarization, gmst_estimate
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

        Returns
        -------
        fplus(default) or fx or fb : float or numpy.ndarray
            The plus or vector-x or breathing polarization factor for this sky location / orientation
        fcross(default) or fy or fl : float or numpy.ndarray
            The cross or vector-y or longitudnal polarization factor for this sky location / orientation
        """

        gha = gmst_estimate - right_ascension

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(declination)
        sindec = torch.sin(declination)
        cospsi = torch.cos(polarization)
        sinpsi = torch.sin(polarization)

        # Basis vectors
        x = torch.stack(
            [
                -cospsi * singha - sinpsi * cosgha * sindec,
                -cospsi * cosgha + sinpsi * singha * sindec,
                sinpsi * cosdec,
            ],
            dim=-1,
        )

        y = torch.stack(
            [
                sinpsi * singha - cospsi * cosgha * sindec,
                sinpsi * cosgha + cospsi * singha * sindec,
                cospsi * cosdec,
            ],
            dim=-1,
        )

        # x & y are the same for all dets
        # self.response should vary for each
        dx = x @ self.response.T
        dy = y @ self.response.T

        fplus = torch.sum(x * dx - y * dy, dim=-1)
        fcross = torch.sum(x * dy + y * dx, dim=-1)

        return fplus, fcross

    def time_delay_from_earth_center(self, right_ascension, declination, gmst_estimate):
        """Return the time delay from the given location to detector for
        a signal with the given sky location
        In other words return `t1 - t2` where `t1` is the
        arrival time in this detector and `t2` is the arrival time in the
        other location.

        Parameters
        ----------
        right_ascension : float
            The right ascension (in rad) of the signal.
        declination : float
            The declination (in rad) of the signal.

        Returns
        -------
        float
            The arrival time difference between the detectors.
        """
        ra_angle = gmst_estimate - right_ascension
        cosd = torch.cos(declination)

        e0 = cosd * torch.cos(ra_angle)
        e1 = cosd * -torch.sin(ra_angle)
        e2 = torch.sin(declination)

        ehat = torch.stack([e0, e1, e2], dim=-1)
        return ehat @ self.dx.T / C

    def constant_project(self, hp, hc, freqs, ra, dec, polarization):
        """Return the strain of a waveform as measured by all detectors.
        Apply the time shift for all given detectors relative to the assumed
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
        # Get GMST estimates for entire batch
        # Batch shape should be (batch_size, num_dets, seq_len)
        gmst_estimate = self.random_gmst_estimate(batch_shape=hp.size()[0])
        # 'constant' assume fixed orientation relative to source over the
        # duration of the signal, accurate for short duration signals
        fp, fc = self.antenna_pattern(ra, dec, polarization, gmst_estimate)
        # Get time delay for all dets given the sky location
        dt = self.time_delay_from_earth_center(ra, dec, gmst_estimate)
        # Get hf from hp and hc given detector response
        hf = fp[..., None] * hp + fc[..., None] * hc
        # Apply time shift relative to detectors
        hf *= torch.exp(-2j * PI * freqs * dt)
        return hf

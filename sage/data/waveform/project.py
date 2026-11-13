#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batched frequency-domain projection of h+ and hx onto detectors, with
a random GMST per sample.

Filename        : project.py
Description     : Short description of the file

Created on 2026-01-23 16:17:38

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
import numpy as np

from astropy import coordinates, units

from sage.core.constants import PI, C
from sage.core.math import rotation_matrix
from sage.core.hardcode import _DETMETADATA
from sage.data.waveform import waveform_utils
from sage.core.geometry import Window


class ConstantProjection(torch.nn.Module):
    """
    Frequency-domain projection of batched waveforms onto detectors.

    Applies each detector's antenna pattern and time delay from the
    geocentre. The GMST is drawn uniformly per sample, not taken from a
    GPS time; the antenna pattern is constant over the signal.

    Parameters
    ----------
    detectors : list of str
        Detectors, in channel order.
    window : Window
        Window geometry; sets the frequency grid.
    device : str or torch.device
        Device of the cached tensors.
    dtype : torch.dtype
        Dtype of the cached tensors.
    generator : torch.Generator
        Generator of the GMST draws, on ``device``.
    """

    def __init__(self, detectors: list, window: Window, device, dtype: torch.dtype, generator: torch.Generator):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.generator = generator
        self.detnames = list(detectors)

        freqs = waveform_utils.get_freqs(
            f_l=0.0,
            f_u=window.sample_rate_in_hz / 2.0,
            sample_length_in_s=window.padded_length_in_s,
            device=device,
            dtype=dtype,
        )
        self.freqs = freqs

        # Response tensor of each detector, (D, 3, 3)
        self.response = torch.empty(
            (len(self.detnames), 3, 3),
            device=self.device,
            dtype=self.dtype,
        )

        # Earth centre minus each detector's position, (D, 3)
        earth_center = torch.tensor([0, 0, 0], device=self.device, dtype=self.dtype)
        self.dx = torch.empty(
            (len(self.detnames), 3),
            device=self.device,
            dtype=self.dtype,
        )

        # Baseline response of a single arm pointed in the -X direction
        self.resp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])

        # Hardcoded detector metadata, from LAL
        for ndet, detname in enumerate(self.detnames):
            self.response[ndet] = self.get_detector_response(
                _DETMETADATA[detname]["longitude"],
                _DETMETADATA[detname]["latitude"],
                _DETMETADATA[detname]["yangle"],
                _DETMETADATA[detname]["xangle"],
                _DETMETADATA[detname]["xaltitude"],
                _DETMETADATA[detname]["yaltitude"],
            )

            loc = self.get_relative_position(detname)
            self.dx[ndet] = earth_center - loc

    def get_relative_position(self, detname):
        """
        Earth-centred, Earth-fixed position of a detector.

        From the detector's geodetic coordinates, with astropy.

        Parameters
        ----------
        detname : str
            Detector name (e.g. ``"H1"``, ``"L1"``).

        Returns
        -------
        torch.Tensor, shape ``(3,)``
            ``[x, y, z]`` in metres.
        """
        loc = coordinates.EarthLocation.from_geodetic(
            _DETMETADATA[detname]["longitude"] * units.rad,
            _DETMETADATA[detname]["latitude"] * units.rad,
            _DETMETADATA[detname]["height"] * units.meter,
        )

        return torch.tensor(
            [loc.x.value, loc.y.value, loc.z.value],
            device=self.device,
        )

    def get_detector_response(
        self,
        longitude,
        latitude,
        yangle=0,
        xangle=None,
        xaltitude=0,
        yaltitude=0,
    ):
        """
        Response tensor of a detector, as PyCBC's add_detector_on_earth.

        Parameters
        ----------
        longitude : float
            Geodetic longitude of the detector in radians.
        latitude : float
            Geodetic latitude of the detector in radians.
        yangle : float
            Azimuth of the y-arm from north, in radians.
        xangle : float
            Azimuth of the x-arm from north, in radians.
        xaltitude : float
            Altitude of the x-arm from the local horizon, in radians.
        yaltitude : float
            Altitude of the y-arm from the local horizon, in radians.

        Returns
        -------
        torch.Tensor, shape ``(3, 3)``
            The response tensor, in Earth-centred coordinates.
        """
        # {x,y,z} -> {0,1,2}
        rm2 = rotation_matrix(-longitude, 2)
        rm1 = rotation_matrix(-1.0 * (PI / 2.0 - latitude), 1)

        # Calculate response in earth centered coordinates
        # by rotation of response in coordinates aligned
        # with the detector arms
        resps = []
        for angle, azi in [(yangle, yaltitude), (xangle, xaltitude)]:
            # {x,y,z} -> {0,1,2}
            rm0 = rotation_matrix(angle, 2)
            rmN = rotation_matrix(-azi, 1)
            rm = rm2 @ rm1 @ rm0 @ rmN
            # apply rotation
            resps.append(rm @ self.resp @ rm.T / 2.0)

        return torch.tensor(resps[0] - resps[1], device=self.device, dtype=self.dtype)

    def random_gmst_estimate(self, B: int) -> torch.Tensor:
        """
        GMST values drawn uniformly from ``[0, 2 pi)``.

        Parameters
        ----------
        B : int
            Number of values.

        Returns
        -------
        torch.Tensor, shape ``(B,)``
            GMST values in radians.
        """
        return 2 * PI * torch.rand(B, device=self.device, generator=self.generator)

    def antenna_pattern(
        self, right_ascension, declination, polarization, gmst_estimate
    ):
        """
        Plus and cross antenna patterns of every detector.

        Parameters
        ----------
        right_ascension : torch.Tensor, shape ``(B,)``
            Right ascension of the source in radians.
        declination : torch.Tensor, shape ``(B,)``
            Declination of the source in radians.
        polarization : torch.Tensor, shape ``(B,)``
            Polarisation angle of the source in radians.
        gmst_estimate : torch.Tensor, shape ``(B,)``
            GMST in radians.

        Returns
        -------
        fplus, fcross : torch.Tensor, shape ``(B, D)``
            Plus and cross antenna patterns.
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

        # Shared basis vectors against each detector's response
        dx = torch.einsum("bi,dij->bdj", x, self.response)
        dy = torch.einsum("bi,dij->bdj", y, self.response)

        fplus = torch.sum(x.unsqueeze(-2) * dx - y.unsqueeze(-2) * dy, dim=-1)
        fcross = torch.sum(x.unsqueeze(-2) * dy + y.unsqueeze(-2) * dx, dim=-1)

        return fplus, fcross

    def time_delay_from_earth_center(
        self,
        right_ascension,
        declination,
        gmst_estimate,
    ):
        """
        Arrival time at each detector minus that at the Earth's centre.

        Parameters
        ----------
        right_ascension : torch.Tensor, shape ``(B,)``
            Right ascension of the signal in radians.
        declination : torch.Tensor, shape ``(B,)``
            Declination of the signal in radians.
        gmst_estimate : torch.Tensor, shape ``(B,)``
            GMST in radians.

        Returns
        -------
        torch.Tensor, shape ``(B, D)``
            Time delays in seconds.
        """
        ra_angle = gmst_estimate - right_ascension
        cosd = torch.cos(declination)

        e0 = cosd * torch.cos(ra_angle)
        e1 = cosd * -torch.sin(ra_angle)
        e2 = torch.sin(declination)

        ehat = torch.stack([e0, e1, e2], dim=-1)

        # dx.dot(ehat) per sample and detector, as PyCBC's
        # Detector.time_delay_from_location
        # (pycbc/detector/ground.py:449)
        return torch.einsum("bi,di->bd", ehat, self.dx) / C

    def forward(self, hp, hc, ra, dec, polarization, return_delay=False):
        """
        Strain of each waveform in every detector.

        Applies the antenna patterns to h+ and hx and the time shift of
        each detector from the geocentre, at a random GMST per sample.

        Parameters
        ----------
        hp : torch.Tensor, shape ``(B, F)``
            Plus polarisation in the frequency domain.
        hc : torch.Tensor, shape ``(B, F)``
            Cross polarisation in the frequency domain.
        ra : torch.Tensor, shape ``(B,)``
            Right ascension of the source in radians.
        dec : torch.Tensor, shape ``(B,)``
            Declination of the source in radians.
        polarization : torch.Tensor, shape ``(B,)``
            Polarisation angle of the source in radians.
        return_delay : bool
            Also return the applied time delays.

        Returns
        -------
        hf : torch.Tensor, shape ``(B, D, F)``
            Detector-frame strain.
        dt : torch.Tensor, shape ``(B, D)``
            With ``return_delay``: the delays in seconds, so that
            ``tc_det = tc_geocentric + dt``.
        """
        freqs = self.freqs
        gmst_estimate = self.random_gmst_estimate(ra.shape[0])
        # Orientation fixed over the signal's duration
        fp, fc = self.antenna_pattern(ra, dec, polarization, gmst_estimate)
        dt = self.time_delay_from_earth_center(ra, dec, gmst_estimate)
        hf = fp[..., None] * hp[:, None, :] + fc[..., None] * hc[:, None, :]
        phase = torch.polar(
            torch.ones(1, 1, freqs.shape[-1], device=self.device, dtype=self.dtype),
            -2 * PI * freqs[None, None, :] * dt[:, :, None],
        )
        # hf and phase are (B, D, F)
        hf *= phase

        if return_delay:
            return hf, dt
        return hf

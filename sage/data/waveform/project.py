#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Packages
import torch
import numpy as np

from astropy import coordinates, units

# LOCAL
from sage.core.constants import PI, C
from sage.core.math import rotation_matrix
from sage.core.hardcode import _DETMETADATA
from sage.data.waveform import waveform_utils
from sage.core.config import get_cfg, get_data_cfg


class ConstantProjection(torch.nn.Module):
    """
    GPU-native frequency-domain detector projection for batched GW waveforms.

    Projects a batch of plus/cross polarisation waveforms ``(hp, hc)`` onto
    each detector's antenna pattern and applies the sky-position-dependent
    time-delay phase shift to produce detector-frame strain ``h(f)``.

    The module pre-computes and caches:

    - **Detector response tensors** ``(D, 3, 3)`` built from LAL-derived arm
      azimuths and altitudes via Euler rotation matrices.
    - **Detector ECEF position vectors** ``(D, 3)`` for time-delay calculation.
    - **Frequency array** ``(F,)`` on the target device.

    The Greenwich Mean Sidereal Time (GMST) is randomised per batch rather
    than looked up from GPS time — this is an approximation suitable for
    training where the source position is already drawn isotropically.

    Parameters
    ----------
    None — all configuration is read from :func:`~sage.core.config.get_cfg`
    and :func:`~sage.core.config.get_data_cfg` at construction time.

    Inputs to :meth:`forward`
    -------------------------
    hp, hc : torch.Tensor, shape ``(B, F)``, complex
        Plus and cross polarisations in the frequency domain.
    ra, dec, polarization : torch.Tensor, shape ``(B,)``
        Right ascension (rad), declination (rad), and polarisation angle (rad).

    Returns
    -------
    hf : torch.Tensor, shape ``(B, D, F)``, complex
        Detector-frame frequency-domain strain for each detector.
    """

    def __init__(self):

        super().__init__()

        # Setup configs
        cfg = get_cfg()
        data_cfg = get_data_cfg()

        # CUDA device
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.batch_size = int(cfg.batch_size * cfg.class_balance)
        # Detector
        self.detnames = cfg.detectors

        # Frequencies
        freqs, _ = waveform_utils.get_freqs(
            f_l=0.0,
            f_u=data_cfg.sample_rate / 2.0,
            sample_length_in_s=data_cfg.padded_length_in_s,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        self.freqs = freqs

        # Detector tensor
        # self.response shape is (batch_size, num_dets, response)
        self.response = torch.empty(
            (len(self.detnames), 3, 3),
            device=self.device,
            dtype=self.dtype,
        )

        # Get relative position of DET from Earth center
        earth_center = torch.tensor([0, 0, 0], device=self.device, dtype=self.dtype)
        self.dx = torch.empty(
            (len(self.detnames), 3),
            device=self.device,
            dtype=self.dtype,
        )

        # Baseline response of a single arm pointed in the -X direction
        self.resp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])

        # Get hardcoded detector metadata (obtained from LAL)
        for ndet, detname in enumerate(self.detnames):
            self.response[ndet] = self.get_detector_response(
                _DETMETADATA[detname]["longitude"],
                _DETMETADATA[detname]["latitude"],
                _DETMETADATA[detname]["yangle"],
                _DETMETADATA[detname]["xangle"],
                _DETMETADATA[detname]["xaltitude"],
                _DETMETADATA[detname]["yaltitude"],
            )

            # Relative position of DET of Earth center
            loc = self.get_relative_position(detname)
            self.dx[ndet] = earth_center - loc

    def get_relative_position(self, detname):
        """
        Return the ECEF position vector of *detname* as a ``torch.Tensor``.

        Parameters
        ----------
        detname : str
            Detector name (e.g. ``"H1"``, ``"L1"``).

        Returns
        -------
        torch.Tensor, shape ``(3,)``
            Earth-centred, Earth-fixed ``[x, y, z]`` coordinates in metres.
        """
        # Get relative position of DET from Earth center
        # Detector position (ECEF)
        # TODO: Convert to PyTorch too at some point
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
            resps.append(rm @ self.resp @ rm.T / 2.0)

        return torch.tensor(resps[0] - resps[1], device=self.device, dtype=self.dtype)

    def random_gmst_estimate(self, B=None):
        """
        Return a batch of uniformly random GMST values in ``[0, 2π)``.

        Exact GPS-to-GMST conversion requires expensive table look-ups.
        Randomising GMST instead effectively marginalises over Earth's
        rotation, which is equivalent to drawing uniformly from all
        possible observation times.

        Parameters
        ----------
        B : int or None
            Batch size to use.  When ``None``, falls back to
            ``self.batch_size`` (the value fixed at construction time from
            ``cfg.batch_size * cfg.class_balance``).  Pass an explicit value
            when calling from a context where the runtime batch size may
            differ from the training batch size (e.g. validation), because
            using the fixed ``self.batch_size`` would produce a shape mismatch.
            NOTE: torch.compile with static shapes requires a fixed batch size;
            if you re-enable compilation, revert to the ``self.batch_size``
            path and ensure the caller always uses the same B.

        Returns
        -------
        torch.Tensor, shape ``(B,)``
            GMST values in radians.
        """
        # Random GMST in radians to compute the antenna patterns
        # Reference times to GMST requires table reads and is expensive
        # Instead we simply randomise GMST in [0, 2PI)
        #
        # NOTE (kept for torch.compile compatibility reference):
        # The original implementation used self.batch_size (a fixed integer
        # baked in at construction time) so that torch.compile sees a static
        # shape and does not retrace.  The dynamic version below uses the
        # caller-supplied B; this triggers retracing if B changes between
        # calls, which breaks torch.compile with static-shape assumptions.
        # Old line (static, compile-safe but wrong at validation time):
        #   return 2 * PI * torch.rand(self.batch_size, device=self.device)
        _B = self.batch_size if B is None else B
        return 2 * PI * torch.rand(_B, device=self.device)

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
            The plus or vector-x or breathing polarization factor
            for this sky location / orientation
        fcross(default) or fy or fl : float or numpy.ndarray
            The cross or vector-y or longitudnal polarization factor
            for this sky location / orientation
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

        # "bi,di->bd": dot product over spatial index i for each batch sample b
        # and detector d.  The previous "bi,dj->bd" was wrong — it computed
        # (sum_i ehat[b,i]) * (sum_j dx[d,j]), a product of scalar sums rather
        # than a proper 3D dot product.  PyCBC reference: dx.dot(ehat) in
        # Detector.time_delay_from_location (pycbc/detector/ground.py:449).
        return torch.einsum("bi,di->bd", ehat, self.dx) / C

    def forward(self, hp, hc, ra, dec, polarization, freqs=None, return_delay=False):
        """Return the strain of a waveform as measured by all detectors.
        Apply the time shift for all given detectors relative to the assumed
        geocentric frame and apply the antenna patterns to the plus and cross
        polarizations.

        When ``return_delay=True`` also return the per-detector time delay
        ``dt`` of shape ``(B, D)`` (seconds, relative to the geocentre) that was
        applied. This is the quantity that converts a geocentric coalescence
        time into the per-detector arrival time: ``tc_det = tc_geocentric + dt``.

        Parameters
        ----------
        hp: torch.Tensor, shape (B, F)
            Plus polarization of the GW
        hc: torch.Tensor, shape (B, F)
            Cross polarization of the GW
        ra: float
            Right ascension of source location
        dec: float
            Declination of source location
        polarization: float
            Polarization angle of the source
        freqs: torch.Tensor, shape (F,) or None
            Frequency array in Hz.  When None the module's stored full
            uniform grid (self.freqs) is used.  Pass the coarse multibanding
            frequency array here when hp/hc are already in the coarse
            representation so the time-delay phase is computed at the right
            frequencies.
        """
        if freqs is None:
            freqs = self.freqs
        # Get GMST estimates for entire batch.
        # Pass the actual runtime batch size so validation (where B may differ
        # from self.batch_size) does not produce a shape mismatch.
        gmst_estimate = self.random_gmst_estimate(B=ra.shape[0])
        # 'constant' assume fixed orientation relative to source over the
        # duration of the signal, accurate for short duration signals
        fp, fc = self.antenna_pattern(ra, dec, polarization, gmst_estimate)
        # Get time delay for all dets given the sky location
        dt = self.time_delay_from_earth_center(ra, dec, gmst_estimate)
        # Get hf from hp and hc given detector response
        hf = fp[..., None] * hp[:, None, :] + fc[..., None] * hc[:, None, :]
        # Apply time shift relative to detectors
        phase = torch.polar(
            torch.ones(1, 1, freqs.shape[-1], device=self.device, dtype=self.dtype),
            -2 * PI * freqs[None, None, :] * dt[:, :, None],
        )
        # hf is (B, N_ifo, seq_len) and phase is (B, 1, seq_len)
        hf *= phase

        if return_delay:
            return hf, dt
        return hf

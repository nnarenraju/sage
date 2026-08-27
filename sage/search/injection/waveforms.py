#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : waveforms.py
Description   : Waveform generation and detector projection for injections.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Waveforms are generated to match the settings under which the published injection set
was defined, so that a recovered injection means the same thing here as in the
reference analyses. Sage's own GPU approximants are used where they are verified
against the reference implementation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WaveformSettings:
    """Generation settings pinned to the injection release."""

    approximant: str = "IMRPhenomXPHM"
    f_ref: float = 16.0
    f_lower: float = 10.0
    f_final: float = 8192.0
    sample_rate: float = 2048.0
    multibanding: bool = False


class InjectionGenerator:
    """Generate polarisations for a batch of injections."""

    def __init__(self, settings: WaveformSettings, device: str = "cuda") -> None:
        raise NotImplementedError

    def generate(self, params: Dict[str, np.ndarray]):
        """Return frequency-domain plus and cross polarisations."""
        raise NotImplementedError

    def optimal_snr(self, params: Dict[str, np.ndarray], asds) -> np.ndarray:
        """Network optimal SNR against the reference spectra."""
        raise NotImplementedError


class ExactProjection:
    """Project polarisations onto detectors with full time-delay and antenna response."""

    def __init__(self, detectors: Sequence[str]) -> None:
        raise NotImplementedError

    def project(
        self,
        hp,
        hc,
        ra: np.ndarray,
        dec: np.ndarray,
        psi: np.ndarray,
        gps: np.ndarray,
    ):
        """Return per-detector strain, including light-travel delays."""
        raise NotImplementedError

    def time_delay_s(self, detector: str, ra: np.ndarray, dec: np.ndarray, gps: np.ndarray) -> np.ndarray:
        """Geocentre-to-detector arrival-time delay."""
        raise NotImplementedError


class TabulatedSampler:
    """
    A parameter sampler that reads pre-drawn rows instead of sampling a prior.

    Everything except the values comes from the real
    :class:`~sage.data.waveform.sampler.DistributionSampler` the network was trained
    under: the column order, the standardisation buffers, the bounds and the constraint
    machinery. Only :meth:`__call__` is replaced, to return the next rows of a table.

    That is what makes the injections comparable with the training distribution. A
    separately built sampler could order its columns differently or standardise against
    different buffers, and either would put the point estimates on a different scale
    without changing anything that fails.

    Attribute writes are forwarded to the wrapped sampler, because the approximant sets
    ``param_sampler.req_idx`` on it during construction and the slice it implies has to be
    the one the wrapped object reports.
    """

    def __init__(self, base, table) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_cursor", 0)

    def __call__(self, n: int):
        """
        The next ``n`` rows, wrapping is refused rather than silently repeating.

        Returned in the wrapped sampler's dtype and on its device. The population is drawn
        in float64 and staged to disk, which is right for the draw, but the approximant
        multiplies the parameter batch against coefficient tables registered in the
        sampler's dtype -- so handing it float64 raised ``expected mat1 and mat2 to have
        the same dtype`` in the first matmul of ``IMRPhenomD.get_coeffs``. Training
        samples in that dtype, so casting here is what the network was fitted against.
        """
        cursor = object.__getattribute__(self, "_cursor")
        table = object.__getattribute__(self, "_table")
        base = object.__getattribute__(self, "_base")
        if cursor + int(n) > table.shape[0]:
            raise IndexError(
                f"the campaign asked for {n} injections from row {cursor} of a table "
                f"holding {table.shape[0]}. Wrapping would score the same signals twice "
                "and put them into p(x | signal) twice"
            )
        object.__setattr__(self, "_cursor", cursor + int(n))
        rows = table[cursor : cursor + int(n)]
        return rows.to(device=base.device, dtype=base.dtype)

    def seek(self, row: int) -> None:
        """Rewind or advance to a row, for resuming a batch-committed campaign."""
        object.__setattr__(self, "_cursor", int(row))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name, value) -> None:
        setattr(object.__getattribute__(self, "_base"), name, value)


def build_injection_table(base_sampler, intrinsic, seed: int = 0):
    """
    Assemble the full parameter table: population intrinsics, prior extrinsics.

    ``intrinsic`` is ``(N, 7)`` in :data:`~sage.search.injection.population.
    INTRINSIC_COLUMNS` order -- ``m1, q, z, chi_1, chi_2, costilt_1, costilt_2`` -- as
    drawn from the GWTC-3 Power-Law + Peak model.

    The extrinsic parameters are drawn from the *training run's own prior*, and the
    intrinsic columns are then overwritten. That is sgwc-1's construction with the PyCBC
    ``.ini`` replaced by the ``gwconfig.yaml`` expressing the same prior: right ascension,
    declination, inclination, coalescence phase, polarisation and the in-window
    coalescence time are all prior draws there too.

    Spin components are built the way sgwc-1 builds them: the population supplies a
    magnitude and a cosine tilt per component, the azimuth is uniform on ``[0, 2pi)``, and
    the Cartesian components follow. Luminosity distance comes from redshift under
    **Planck15**, which is the cosmology GWTC-3 quotes the population in -- using a
    different one would place a population defined in one cosmology at distances from
    another.

    **The table is detector frame**, which is PyCBC's convention and therefore this one.
    The population model states source-frame masses; the columns written here are
    ``m_source * (1 + z)``, because that is what a waveform generator is handed. The two
    were previously mixed -- the redshift was spent on the distance and withheld from the
    masses -- which placed a source-frame binary at its correct luminosity distance while
    keeping it too light for that distance by a median factor of 2.06. See SB-50.

    Returns
    -------
    torch.Tensor
        ``(N, num_params)`` in the sampler's own column order, ready to be handed to
        :class:`TabulatedSampler`.
    """
    import torch
    from astropy import units as u
    from astropy.cosmology import Planck15
    from astropy.cosmology import units as cu

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.ndim != 2 or intrinsic.shape[1] != 7:
        raise ValueError(
            f"intrinsic parameters have shape {intrinsic.shape}; expected (N, 7) in the "
            "order (m1, q, z, chi_1, chi_2, costilt_1, costilt_2)"
        )
    n = intrinsic.shape[0]
    table = base_sampler(n).clone()
    index = base_sampler.param_index

    m1_source, q, z = intrinsic[:, 0], intrinsic[:, 1], intrinsic[:, 2]
    chi1, chi2 = intrinsic[:, 3], intrinsic[:, 4]
    cos1, cos2 = intrinsic[:, 5], intrinsic[:, 6]

    # Source frame to detector frame. The Power-Law + Peak model is a statement about
    # source-frame masses; a waveform generator is handed detector-frame ones, and the
    # two differ by (1 + z) -- which on this population is a median factor of 2.06, so it
    # is not a correction that can be left implicit.
    #
    # PyCBC's convention, which this follows: `mass1`/`mass2` are unqualified and are what
    # the generator receives, while `srcmass1`/`srcmass2`/`srcmchirp` are separate,
    # explicitly source-frame parameters filed under "derived parameters (these are not
    # used for waveform generation)" (pycbc/waveform/parameters.py:167-174, 203, 216-231).
    # The relation is `pycbc.mchirp_area.src_mass_from_z_det_mass`, `msrc = mdet / (1 + z)`
    # (mchirp_area.py:134), and `transforms.LambdaFromTOVFile` states it directly: "the
    # mass values to be transformed are assumed to be detector frame masses ... a distance
    # should be provided along with the mass for transformation to the source frame mass".
    #
    # A PyCBC injection file stores detector-frame masses and a luminosity distance and
    # nothing else -- `population/scale_injections.py:13` enumerates exactly what is
    # saved, and neither a redshift nor a source-frame mass is in it. The redshift is
    # recovered downstream by inverting the distance (`cosmology.redshift`), so the table
    # written here carries the same information under the same convention.
    m1 = m1_source * (1.0 + z)

    rng = np.random.default_rng(int(seed))
    phi1 = rng.uniform(0.0, 2.0 * np.pi, n)
    phi2 = rng.uniform(0.0, 2.0 * np.pi, n)
    sin1 = np.sqrt(np.clip(1.0 - cos1**2, 0.0, None))
    sin2 = np.sqrt(np.clip(1.0 - cos2**2, 0.0, None))

    distance = (
        (z * cu.redshift)
        .to(u.Mpc, cu.redshift_distance(Planck15, kind="luminosity"))
        .value
    )

    m2 = q * m1
    values = {
        "mass1": m1,
        "mass2": m2,
        "spin1x": chi1 * sin1 * np.cos(phi1),
        "spin1y": chi1 * sin1 * np.sin(phi1),
        "spin1z": chi1 * cos1,
        "spin2x": chi2 * sin2 * np.cos(phi2),
        "spin2y": chi2 * sin2 * np.sin(phi2),
        "spin2z": chi2 * cos2,
        "distance": distance,
    }

    # The prior's *derived* columns, rewritten from the values just set. The sampler
    # carries both representations -- Cartesian spins and the polar triple it sampled
    # them from, masses and the (mchirp, q) it drew them from -- and only the Cartesian
    # ones reach the waveform. Left alone, the rest would still hold the training-prior
    # draw: a table whose `mchirp` column disagrees with the `mass1` and `mass2` beside
    # it, which is the column any recovered-versus-injected comparison reads.
    derived = {
        "mchirp": (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2,
        "q": q,
        "spin1_a": chi1,
        "spin2_a": chi2,
        "spin1_polar": np.arccos(np.clip(cos1, -1.0, 1.0)),
        "spin2_polar": np.arccos(np.clip(cos2, -1.0, 1.0)),
        "spin1_azimuthal": phi1,
        "spin2_azimuthal": phi2,
    }
    if "chirp_distance" in index:
        # PyCBC's definition, against the fiducial 1.4+1.4 chirp mass.
        fiducial = (1.4 * 1.4) ** 0.6 / (2.8) ** 0.2
        derived["chirp_distance"] = distance * (fiducial / derived["mchirp"]) ** (5.0 / 6.0)
    values.update({k: v for k, v in derived.items() if k in index})

    missing = sorted(name for name in values if name not in index)
    if missing:
        raise KeyError(
            f"the training prior names no {missing}; the injection table is written into "
            f"the sampler's own columns and it holds {sorted(index)}"
        )
    for name, column in values.items():
        table[:, index[name]] = torch.as_tensor(
            column, dtype=table.dtype, device=table.device
        )
    return table


def in_training_prior(table, base_sampler, mass_lo: float, mass_hi: float):
    """
    Mask of injections whose chirp mass lies inside the trained region.

    sgwc-1 keeps exactly this subset (``injection_study.ipynb`` cells 29-32): the chirp
    mass of an equal-mass binary at each end of the training mass prior bounds what the
    network was ever shown, and an injection outside it measures how the network responds
    to a signal it was not trained on rather than how it responds to a signal.

    Returned rather than applied, so the caller records how many were dropped. On sgwc-1's
    draw that was a large fraction, and a silently shortened injection set would make the
    signal density describe a different population from the one drawn.
    """
    index = base_sampler.param_index
    m1 = np.asarray(table[:, index["mass1"]].cpu(), dtype=np.float64)
    m2 = np.asarray(table[:, index["mass2"]].cpu(), dtype=np.float64)
    mchirp = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    lo = (mass_lo * mass_lo) ** 0.6 / (mass_lo + mass_lo) ** 0.2
    hi = (mass_hi * mass_hi) ** 0.6 / (mass_hi + mass_hi) ** 0.2
    return (mchirp >= lo) & (mchirp <= hi)

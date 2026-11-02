"""
Spectral lines in strain: Thomson's multitaper harmonic F-test, and
their excision from a fiducial ASD.

Thomson, D. J. (1982), Proc. IEEE 70(9), 1055-1096. Tapers and
eigencoefficients from ``multitaper`` (Prieto, G. A. (2022), Seismol.
Res. Lett. 93(3), 1922-1929).
"""

import numpy as np
from multitaper import MTSpec
from multitaper import utils as mt_utils
from scipy import ndimage, stats


def ftest(spec: MTSpec) -> np.ndarray:
    """
    Thomson's F statistic at every frequency of a multitaper spectrum.

    ``multitaper.utils.ftest`` as array operations (that function loops
    over the frequencies in Python). Under pure noise the statistic
    follows F(2, 2K - 2).

    Parameters
    ----------
    spec : multitaper.MTSpec
        Spectrum of one stretch.

    Returns
    -------
    numpy.ndarray, shape ``(nfft,)``
        F statistic.
    """
    tapers = spec.vn.shape[1]
    vn0 = np.sum(spec.vn, axis=0)
    mu = (spec.yk @ vn0) / np.sum(np.abs(vn0) ** 2)
    misfit = np.sum(np.abs(spec.yk - mu[:, None] * vn0) ** 2, axis=1)
    return (tapers - 1) * np.abs(mu) ** 2 * np.sum(vn0**2) / misfit


def line_detections(
    sample, npts: int, stretches: int, nw: float, tapers: int, sample_rate: float
) -> np.ndarray:
    """
    Number of stretches in which each frequency holds a line.

    A frequency holds a line in a stretch when its F statistic exceeds
    the ``1 - 1/F`` quantile of F(2, 2K - 2) for F frequencies: about
    one false line per stretch.

    Parameters
    ----------
    sample : callable
        ``sample(npts)`` returns one stretch of strain.
    npts : int
        Samples per stretch.
    stretches : int
        Stretches drawn.
    nw : float
        Time-bandwidth product NW.
    tapers : int
        Number of Slepian tapers K.
    sample_rate : float
        Sample rate of the strain (Hz).

    Returns
    -------
    numpy.ndarray of int, shape ``(npts // 2 + 1,)``
        Detections per frequency.
    """
    bins = npts // 2 + 1
    vn, lamb = mt_utils.dpss(npts, nw, tapers)
    threshold = stats.f.isf(1 / bins, 2, 2 * tapers - 2)
    detections = np.zeros(bins, dtype=np.int64)

    for _ in range(stretches):
        spec = MTSpec(
            sample(npts),
            nw=nw,
            kspec=tapers,
            dt=1 / sample_rate,
            nfft=npts,
            iadapt=1,
            vn=vn,
            lamb=lamb,
        )
        detections += ftest(spec)[:bins] > threshold
    return detections


def line_bins(detections: np.ndarray, stretches: int, alpha: float) -> np.ndarray:
    """
    Frequencies detected in more stretches than chance allows.

    Binomial test at level ``alpha``: under pure noise each stretch
    detects a frequency with probability ``alpha``.

    Parameters
    ----------
    detections : numpy.ndarray of int
        Detections per frequency (:func:`line_detections`).
    stretches : int
        Stretches the detections come from.
    alpha : float
        Detection level per stretch and level of the test.

    Returns
    -------
    numpy.ndarray of int
        Indices of the line bins.
    """
    return np.flatnonzero(stats.binom.sf(detections - 1, stretches, alpha) < alpha)


def excision_weight(lines: np.ndarray, bins: int, half: int) -> np.ndarray:
    """
    Whitening weight that excises lines without sharp spectral edges.

    Zero within ``half`` bins of each line, rising to one over the next
    ``half`` bins with a raised-cosine (Hann) edge (window method;
    Harris, F. J. (1978), Proc. IEEE 66(1), 51-83).

    Parameters
    ----------
    lines : numpy.ndarray of int
        Line bins.
    bins : int
        Frequency bins.
    half : int
        Half-width of the excised band and width of its edge, in bins.

    Returns
    -------
    numpy.ndarray, shape ``(bins,)``
        Weight in ``[0, 1]``.
    """
    excised = np.zeros(bins, dtype=bool)
    for line in lines:
        excised[max(line - half, 0) : line + half + 1] = True
    if not excised.any():
        return np.ones(bins)

    distance = ndimage.distance_transform_edt(~excised)  # to the nearest excised bin
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(distance / (half + 1), 0, 1))

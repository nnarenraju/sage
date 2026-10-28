"""General-purpose waveforms: LAL's frequency-domain output, optionally projected onto detectors."""

import math
import secrets
import warnings
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from sage.config import loader
from sage.config.prior import check_prior, load_prior_file
from sage.data.waveform import waveform_utils
from sage.data.waveform.approximants.IMRPhenomXAS_NRTidalv3 import IMRPhenomXAS_NRTidalv3
from sage.data.waveform.sampler import prior_sampler

APPROXIMANTS = {"IMRPhenomXAS_NRTidalv3": IMRPhenomXAS_NRTidalv3}
# PyCBC get_fd_waveform defaults (pycbc.waveform.parameters); an absent lambda is LAL's 0.
DEFAULTS = {
    "spin1x": 0.0, "spin1y": 0.0, "spin1z": 0.0, "spin2x": 0.0, "spin2y": 0.0, "spin2z": 0.0,
    "distance": 1.0, "coa_phase": 0.0, "inclination": 0.0, "lambda1": 0.0, "lambda2": 0.0,
}
REQUIRED = ("mass1", "mass2")
EXTRINSIC = ("tc", "ra", "dec", "polarization")
PRESETS = "waveform_priors"


def c_round(x: float) -> float:
    """C's round() for x >= 0: halves away from zero."""
    r = math.floor(x)
    return r + 1.0 if x - r >= 0.5 else float(r)


def chirp_duration(mass1, mass2, spin1z, spin2z, f_min, f_max, delta_f) -> float:
    """
    LAL's segment length for a chirp, from the rule of its conditioned FD generator.

    Source: generate_conditioned_fd_waveform_from_fd, LALSimInspiralGeneratorConditioning.c 366-433.

    Parameters
    ----------
    mass1, mass2 : float
        Masses in solar masses.
    spin1z, spin2z : float
        Aligned spins.
    f_min, f_max, delta_f : float
        LAL's f_min, f_max and deltaF, in Hz.

    Returns
    -------
    float
        Duration in seconds; a deltaF above its inverse is too coarse.
    """
    m1, m2 = mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI
    n = int(c_round(f_max / delta_f))
    f_nyquist = f_max if not n & (n - 1) else math.ldexp(1.0, math.frexp(n)[1]) * delta_f
    delta_t = 0.5 / f_nyquist
    fisco = 1.0 / (math.pow(9.0, 1.5) * lal.PI * (m1 + m2) * lal.MTSUN_SI / lal.MSUN_SI)
    if f_min > fisco:
        f_min = fisco
    tchirp = lalsim.SimInspiralChirpTimeBound(f_min, m1, m2, spin1z, spin2z)
    s = lalsim.SimInspiralFinalBlackHoleSpinBound(spin1z, spin2z)
    tmerge = lalsim.SimInspiralMergeTimeBound(m1, m2) + lalsim.SimInspiralRingdownTimeBound(m1 + m2, s)
    textra = 3.0 / f_min
    fstart = lalsim.SimInspiralChirpStartFrequencyBound((1.0 + 0.1) * tchirp, m1, m2)
    tchirp = lalsim.SimInspiralChirpTimeBound(fstart, m1, m2, spin1z, spin2z)
    chirplen = c_round((tchirp + tmerge + 2.0 * textra) / delta_t)
    return math.ldexp(1.0, math.frexp(chirplen)[1]) * delta_t


def project(hp, hc, rows, detectors, delta_f, epoch):
    """
    Detector strain at GPS time, as PyCBC's FDomainDetFrameGenerator, with LAL's response and delay.

    Each detector's strain is (F+ h+ + Fx hx) exp(-2 pi i f (t_det - epoch)), with
    t_det = tc + XLALTimeDelayFromEarthCenter(tc) and F+, Fx from XLALComputeDetAMResponse
    at XLALGreenwichMeanSiderealTime(t_det).

    Parameters
    ----------
    hp, hc : torch.Tensor, shape (n, F)
        Polarisations on LAL's grid from DC.
    rows : list of dict
        Parameters of each signal.
    detectors : list of str
        LAL detector prefixes.
    delta_f : float
        Frequency spacing in Hz.
    epoch : None, float or sequence
        GPS start of the segment; None puts the merger at the end, as LAL does.

    Returns
    -------
    torch.Tensor, shape (n, D, F)
    """
    epochs = [epoch] * len(rows) if epoch is None or np.ndim(epoch) == 0 else list(epoch)
    if len(epochs) != len(rows):
        raise ValueError(f"epoch: give one value or one per signal ({len(rows)}), not {len(epochs)}")
    duration = 1.0 / delta_f
    f = torch.arange(hp.shape[1], device=hp.device, dtype=torch.float64) * delta_f
    out = torch.empty((hp.shape[0], len(detectors), hp.shape[1]), dtype=hp.dtype, device=hp.device)
    for i, (row, start) in enumerate(zip(rows, epochs)):
        offset = duration if start is None else row["tc"] - float(start)
        if not 0.0 < offset <= duration:
            raise ValueError(f"signal {i}: tc = {row['tc']} is outside the segment from epoch = {start} "
                             f"to epoch + 1/delta_f")
        for d, name in enumerate(detectors):
            det = lal.cached_detector_by_prefix[name]
            delay = lal.TimeDelayFromEarthCenter(det.location, row["ra"], row["dec"], row["tc"])
            gmst = lal.GreenwichMeanSiderealTime(row["tc"] + delay)
            fplus, fcross = lal.ComputeDetAMResponse(det.response, row["ra"], row["dec"], row["polarization"], gmst)
            shift = torch.polar(torch.ones_like(f), -2.0 * math.pi * f * (offset + delay))
            out[i, d] = (fplus * hp[i] + fcross * hc[i]) * shift
    return out


def generate_fd_waveform(parameters, *, approximant, delta_f, detectors=None, sample_rate=4096.0,
                         f_lower=20.0, f_ref=20.0, epoch=None, device="cpu", numpy=False):
    """
    LAL's frequency-domain waveforms for given parameters, optionally projected onto detectors.

    Examples
    --------
    One signal's polarisations::

        from sage.data.waveform import generate_fd_waveform
        hp, hc = generate_fd_waveform({"mass1": 1.4, "mass2": 1.3, "lambda1": 400, "lambda2": 500},
                                      approximant="IMRPhenomXAS_NRTidalv3", delta_f=1 / 256)

    Several signals in H1, L1 and V1 at a GPS time::

        h = generate_fd_waveform(rows, approximant="IMRPhenomXAS_NRTidalv3", delta_f=1 / 256,
                                 detectors=["H1", "L1", "V1"])   # (len(rows), 3, F)

    Parameters
    ----------
    parameters : dict or iterable of dict
        PyCBC names (``waveform_utils.WAVEFORM_PARAMETERS``). ``mass1`` and ``mass2`` are
        required; spins, ``distance`` (Mpc), ``inclination``, ``coa_phase`` and the lambdas
        take PyCBC's ``get_fd_waveform`` defaults. With detectors, ``tc`` (geocentric GPS
        time), ``ra``, ``dec`` and ``polarization`` are required.
    approximant : str
        One of :data:`APPROXIMANTS`.
    delta_f : float
        Frequency spacing in Hz; a spacing too coarse for a signal warns, as LAL does.
    detectors : list of str or None
        LAL detector prefixes, e.g. ``["H1", "L1"]``. None returns h+ and hx, and tc, ra, dec
        and polarization then have no effect.
    sample_rate : float
        Sets LAL's f_max = sample_rate / 2.
    f_lower, f_ref : float
        LAL's f_min and fRef in Hz; fRef = 0 means f_min.
    epoch : None, float or sequence
        GPS start of the segment, with detectors; None puts the merger at the end, as LAL does.
    device : str or torch.device
        Where to generate.
    numpy : bool
        Return NumPy arrays instead of torch tensors.

    Returns
    -------
    (n, D, F) complex strain with detectors, else (hplus, hcross) of shape (n, F), on LAL's
    grid of NextPow2(f_max / delta_f) + 1 bins from DC.

    Raises
    ------
    ValueError
        On an unknown approximant, detector or parameter, a missing parameter, or a
        parameter LAL rejects.
    """
    rows = [dict(parameters)] if isinstance(parameters, dict) else [dict(row) for row in parameters]
    if not rows:
        raise ValueError("parameters: give at least one signal")
    if approximant not in APPROXIMANTS:
        raise ValueError(f"approximant {approximant!r}; {loader.suggestion(approximant, list(APPROXIMANTS))}")
    names = waveform_utils.WAVEFORM_PARAMETERS
    for i, row in enumerate(rows):
        unknown = sorted(set(row) - set(names))
        if unknown:
            raise ValueError(f"signal {i}: unknown parameters {unknown}; known: {', '.join(names)}")
        missing = [p for p in REQUIRED + (EXTRINSIC if detectors else ()) if p not in row]
        if missing:
            raise ValueError(f"signal {i}: missing {', '.join(missing)}"
                             + (" (required with detectors)" if set(missing) & set(EXTRINSIC) else ""))
        for name, value in DEFAULTS.items():
            row.setdefault(name, value)
        row.update({name: float(value) for name, value in row.items()})
    # Without detectors the output is h+ and hx; tc, ra, dec and polarization have no effect then.
    if detectors is not None:
        if not detectors:
            raise ValueError("detectors: give at least one, or None for h+ and hx")
        unknown = [d for d in detectors if d not in lal.cached_detector_by_prefix]
        if unknown:
            raise ValueError(f"detectors {unknown}: not LAL detectors; known: {', '.join(lal.cached_detector_by_prefix)}")

    f_max = sample_rate / 2.0
    for i, row in enumerate(rows):
        duration = chirp_duration(row["mass1"], row["mass2"], row["spin1z"], row["spin2z"], f_lower, f_max, delta_f)
        if delta_f > 1.0 / duration:
            warnings.warn(f"signal {i}: Specified frequency interval of {delta_f:g} Hz is too large for a chirp "
                          f"of duration {duration:g} s")

    theta = torch.zeros((len(rows), len(names)), dtype=torch.float64, device=device)
    for i, row in enumerate(rows):
        for name, value in row.items():
            theta[i, names.index(name)] = value
    model = APPROXIMANTS[approximant].from_band(delta_f, f_lower, f_max, f_ref, len(rows), device)
    hp, hc = model.get_hphc(theta, reproduce_lal=True)
    out = (hp, hc) if detectors is None else project(hp, hc, rows, list(detectors), delta_f, epoch)
    if numpy:
        return tuple(h.cpu().numpy() for h in out) if detectors is None else out.cpu().numpy()
    return out


def load_waveform_prior(prior) -> dict:
    """A preset of ``sage/presets/waveform_priors`` by name, or a prior YAML file, checked."""
    if str(prior).endswith(".yaml"):
        return load_prior_file(Path(prior))
    return check_prior(loader.read_preset(PRESETS, prior))


def sample_fd_waveform(num_waveforms, prior, *, approximant, delta_f, detectors=None, seed=None,
                       sample_rate=4096.0, f_lower=20.0, f_ref=20.0, epoch=None, device="cpu", numpy=False):
    """
    Draw signals from a prior and make their waveforms with :func:`generate_fd_waveform`.

    Examples
    --------
    ::

        from sage.data.waveform import sample_fd_waveform
        h, params = sample_fd_waveform(1000, "bns_standard", approximant="IMRPhenomXAS_NRTidalv3",
                                       delta_f=1 / 256, detectors=["H1", "L1"], seed=0)

    Parameters
    ----------
    num_waveforms : int
        Signals to draw.
    prior : str or Path
        A preset name (``sage/presets/waveform_priors``) or a prior YAML file; ``tc`` is a
        geocentric GPS time.
    seed : int or None
        Seed of the prior draws; None draws fresh entropy.
    approximant, delta_f, detectors, sample_rate, f_lower, f_ref, epoch, device, numpy
        As for :func:`generate_fd_waveform`.

    Returns
    -------
    waveforms
        As :func:`generate_fd_waveform` returns them.
    parameters : dict
        Every drawn and derived prior parameter, shape (num_waveforms,).
    """
    sampler = prior_sampler(load_waveform_prior(prior), secrets.randbits(63) if seed is None else seed,
                            "cpu", torch.float64)
    draws = sampler(num_waveforms)
    columns = {name: draws[:, k] for name, k in sampler.param_index.items()}
    names = [name for name in waveform_utils.WAVEFORM_PARAMETERS if name in columns]
    rows = [{name: float(columns[name][i]) for name in names} for i in range(num_waveforms)]
    waveforms = generate_fd_waveform(rows, approximant=approximant, delta_f=delta_f, detectors=detectors,
                                     sample_rate=sample_rate, f_lower=f_lower, f_ref=f_ref, epoch=epoch,
                                     device=device, numpy=numpy)
    parameters = {name: column.numpy() if numpy else column.to(device) for name, column in columns.items()}
    return waveforms, parameters

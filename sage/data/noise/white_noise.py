#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gaussian noise, white or coloured by any ASD, in the time or frequency
domain: reproducible time-addressable series, tensors and a batch
noise sampler.

Filename      : white_noise.py
Description   : Gaussian noise generation — white or coloured by an arbitrary
                ASD, in the time domain or directly in the frequency domain.

Created on 2026-01-19 16:18:49

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.2
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL


Conventions
-----------
One-sided PSD ``S(f)`` (1/Hz) and ``ASD(f) = sqrt(S(f))``, as in LAL and
PyCBC. Unit-PSD white noise (``unit_psd=True``) has ``sigma = sqrt(fs / 2)``
in the time domain, as PyCBC's ``pycbc.noise.reproduceable.normal``; so
colouring is a multiply by the ASD. With ``X = rfft(x, norm="forward")``,
``E|X_k|^2 = S(f_k) * delta_f / 2`` for every bin, DC and Nyquist real.
"""

import difflib
import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from sage.core.geometry import Window
from sage.core.hardcode import PYCBC_FILTER_DURATION, PYCBC_LOW_FREQUENCY_CUTOFF


__all__ = [
    "BLOCK_SAMPLES",
    "sample_synthetic_noise",
    "available_asds",
    "check_detector_asds",
    "white_series",
    "coloured_series",
    "white_noise_td",
    "white_noise_fd",
    "coloured_noise_td",
    "colour_fd",
    "resolve_asd",
    "feather",
    "WhiteNoiseGenerator",
    "WhiteGaussianNoiseSampler",
]


# PyCBC's block size, so that legacy_pycbc=True streams match
# pycbc.noise.reproduceable.normal; every generated stream depends on it
BLOCK_SAMPLES = 1638400

# Block indices can be negative, SeedSequence spawn keys cannot; the
# offset covers |block| < 2**32
_BLOCK_KEY_OFFSET = 1 << 32

# Resolved ASDs keyed by spec and grid, so repeated draws skip the
# model lookup, interpolation and truncation; entries are read-only
_ASD_CACHE = {}
_ASD_CACHE_MAX = 32


def _fast_fft_len(n: int) -> int:
    """
    Round ``n`` up to the next 5-smooth length (factors 2, 3 and 5).

    FFTs of 5-smooth lengths are fast; the added samples are padding
    that is cropped. Lengths up to 8 are returned unchanged.

    Parameters
    ----------
    n : int
        Length.

    Returns
    -------
    int
        5-smooth length >= ``n``.
    """
    if n <= 8:
        return n
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _resolve_asd_cached(spec, **kwargs):
    """
    :func:`resolve_asd` memoised on ``(spec, grid, options)``.

    Unhashable specs (arrays, ``(freqs, values)`` pairs) are resolved
    without the cache. The cached tensor is shared: callers must not
    modify it.

    Parameters
    ----------
    spec : AsdSpec
        ASD specification.
    **kwargs
        Keyword arguments of :func:`resolve_asd`; ``freqs`` is required.

    Returns
    -------
    torch.Tensor or None
        The resolved ASD.
    """
    freqs = kwargs["freqs"]
    try:
        key = (spec, freqs.shape[-1], float(freqs[-1]) if freqs.numel() else 0.0,
               kwargs.get("sample_rate"), kwargs.get("is_psd"),
               kwargs.get("low_frequency_cutoff"), kwargs.get("filter_duration"),
               kwargs.get("n_detectors"), str(kwargs.get("device")),
               kwargs.get("dtype"))
        hash(key)
    except TypeError:
        return resolve_asd(spec, **kwargs)

    hit = _ASD_CACHE.get(key)
    if hit is None:
        hit = resolve_asd(spec, **kwargs)
        if len(_ASD_CACHE) >= _ASD_CACHE_MAX:
            _ASD_CACHE.pop(next(iter(_ASD_CACHE)))
        _ASD_CACHE[key] = hit
    return hit




# Interferometer of each detector label: sets its default curve and is
# checked against the curve it is given
_DETECTOR_FAMILY = {
    "H1": "LIGO", "H2": "LIGO", "L1": "LIGO", "I1": "LIGO", "A1": "LIGO",
    "V1": "Virgo", "K1": "KAGRA", "G1": "GEO",
    "E1": "ET", "E2": "ET", "E3": "ET", "C1": "CE",
}


# Design-sensitivity curve of each interferometer, the default curve
_DETECTOR_DEFAULT_ASD = {
    "LIGO":  "aLIGODesignSensitivityP1200087",
    "Virgo": "AdVDesignSensitivityP1200087",
    "KAGRA": "KAGRADesignSensitivityT1600593",
    "ET":    "EinsteinTelescopeP1600143",
    "CE":    "CosmicExplorerP1600143",
}

# Plain names accepted for the default curves
_ASD_ALIASES = {
    "ligo": "LIGO", "aligo": "LIGO", "h1": "LIGO", "l1": "LIGO",
    "virgo": "Virgo", "advirgo": "Virgo", "adv": "Virgo", "v1": "Virgo",
    "kagra": "KAGRA", "k1": "KAGRA",
    "et": "ET", "einstein": "ET", "einstein telescope": "ET",
    "ce": "CE", "cosmic explorer": "CE", "cosmicexplorer": "CE",
}


def _model_family(name):
    """
    Interferometer an analytic model name describes, or ``None``.

    More specific families are tested first: some LALSimulation names of
    other detectors' curves start with ``aLIGO`` (``aLIGOAdVO4T1800545``
    is an AdVirgo curve).

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    str or None
        ``"LIGO"``, ``"Virgo"``, ``"KAGRA"``, ``"ET"``, ``"CE"`` or
        ``"GEO"``.
    """
    n = name.lower()
    if "adv" in n or "virgo" in n:
        return "Virgo"
    if "kagra" in n:
        return "KAGRA"
    if "einsteintelescope" in n:
        return "ET"
    if "cosmicexplorer" in n:
        return "CE"
    if "ligo" in n:
        return "LIGO"
    if n.startswith("geo"):
        return "GEO"
    return None


def check_detector_asds(asd: dict, detectors: list) -> list:
    """
    Check a per-detector choice of analytic curves.

    Parameters
    ----------
    asd : dict of str to str
        Detector to analytic model name (see :func:`available_asds`).
    detectors : list of str
        Detectors, in channel order.

    Returns
    -------
    list of str
        Model names in channel order.

    Raises
    ------
    ValueError
        If a detector is missing or extra, a name is unknown, or a curve
        belongs to another interferometer.
    """
    missing = [d for d in detectors if d not in asd]
    extra = [d for d in asd if d not in detectors]
    if missing or extra:
        raise ValueError(f"asd needs one curve per detector: missing {missing}, extra {extra}")

    names = available_asds()
    for detector in detectors:
        curve = asd[detector]
        if curve not in names:
            close = difflib.get_close_matches(str(curve), names, n=1)
            hint = f"did you mean {close[0]!r}?" if close else "see available_asds()"
            raise ValueError(f"asd {detector}: unknown curve {curve!r}; {hint}")
        want, got = _DETECTOR_FAMILY.get(detector), _model_family(curve)
        if want is None or want != got:
            raise ValueError(
                f"asd {detector}: {curve!r} is a {got or 'unknown'} curve, "
                f"{detector} is {want or 'unknown'}"
            )
    return [asd[detector] for detector in detectors]


def _expand_asd_spec(asd, detectors):
    """
    Replace ``"auto"``, plain names and an unset ASD by model names.

    An unset ASD of named, known detectors becomes their design curves;
    any other spec is returned unchanged.

    Parameters
    ----------
    asd : AsdSpec or list
        ASD specification, or one per detector.
    detectors : int or sequence or None
        Detector count or names.

    Returns
    -------
    AsdSpec or list
        The expanded specification.

    Raises
    ------
    ValueError
        If ``asd="auto"`` without named detectors of known families.
    """
    def _one(spec):
        """Model name of a plain name; any other spec unchanged."""
        if not isinstance(spec, str):
            return spec
        key = spec.strip().lower()
        if key in ("flat", "white", "none"):
            return spec
        family = _ASD_ALIASES.get(key)
        return _DETECTOR_DEFAULT_ASD[family] if family else spec

    if isinstance(asd, list):
        return [_one(s) for s in asd]

    names = detectors if isinstance(detectors, (list, tuple)) else None
    if names is not None and not all(isinstance(d, str) for d in names):
        names = None
    families = ([_DETECTOR_FAMILY.get(d.upper()) for d in names]
                if names is not None else None)
    all_known = (families is not None
                 and all(f in _DETECTOR_DEFAULT_ASD for f in families))

    if isinstance(asd, str) and asd.strip().lower() == "auto":
        if names is None:
            raise ValueError(
                'asd="auto" picks each detector\'s own curve from its name, so it '
                'needs named detectors — e.g. detectors=["H1", "L1", "V1"].  Name '
                "them, or ask for a specific model instead."
            )
        if not all_known:
            unknown = [d for d, f in zip(names, families)
                       if f not in _DETECTOR_DEFAULT_ASD]
            known = ", ".join(sorted(_DETECTOR_DEFAULT_ASD))
            raise ValueError(
                f'asd="auto" does not recognise {unknown}.  Known families: '
                f"{known}.  Name a model explicitly for those detectors."
            )
        return [_DETECTOR_DEFAULT_ASD[f] for f in families]

    if asd is None and all_known:
        # Known named detectors without an ASD get their design curves
        return [_DETECTOR_DEFAULT_ASD[f] for f in families]

    return _one(asd)


def _check_detector_asd_match(asd, detectors):
    """
    Raise if a named detector is given another interferometer's curve.

    Applies only when both the label and the model name are recognised;
    detector counts, arrays and unknown labels pass.

    Parameters
    ----------
    asd : AsdSpec or list
        ASD specification, or one per detector.
    detectors : int or sequence or None
        Detector count or names.

    Raises
    ------
    ValueError
        If a detector and its curve belong to different interferometers.
    """
    if not isinstance(detectors, (list, tuple)):
        return
    names = [d for d in detectors if isinstance(d, str)]
    if len(names) != len(detectors):
        return

    specs = asd if isinstance(asd, list) else [asd] * len(names)
    if len(specs) != len(names):
        return

    for det, spec in zip(names, specs):
        if not isinstance(spec, str):
            continue
        want = _DETECTOR_FAMILY.get(det.upper())
        got = _model_family(spec)
        if want and got and want != got:
            raise ValueError(
                f"detector {det!r} is a {want} interferometer but was given the "
                f"{got} curve {spec!r}.  Pass the model that matches the "
                f"detector; or, if the pairing is deliberate, drop the names and "
                f"give a count (detectors={len(names)}) or pass the ASD list on "
                f"its own and let it set the detector axis."
            )


def _default_cutoff(spec, low_frequency_cutoff):
    """
    Low-frequency cutoff for ``spec`` when none is given.

    Analytic model names get PyCBC's ``colored_noise`` default; other
    specs get none.

    Parameters
    ----------
    spec : AsdSpec
        ASD specification.
    low_frequency_cutoff : float or None
        Cutoff given by the caller; returned if not ``None``.

    Returns
    -------
    float or None
        The cutoff (Hz).
    """
    if low_frequency_cutoff is not None:
        return low_frequency_cutoff
    if not isinstance(spec, str) or spec.lower() in ("flat", "white", "none"):
        return None
    return PYCBC_LOW_FREQUENCY_CUTOFF


def sample_synthetic_noise(duration, asd=None, *, sample_rate, batch=None,
                           detectors=None, domain="td", seed=None,
                           unit_psd=True, low_frequency_cutoff=None,
                           filter_duration=PYCBC_FILTER_DURATION, numpy=False,
                           device=None, threads=1, dtype=torch.float32):
    """
    Draw synthetic Gaussian noise of a given duration.

    White noise, or noise coloured by ``asd`` with PyCBC's
    ``colored_noise`` recipe; returned as a time series or as its
    ``rfft(norm="forward")``.

    Parameters
    ----------
    duration : float
        Length in seconds.
    asd : AsdSpec or list, optional
        Colouring: ``"auto"``, a plain name (``"LIGO"``, ``"Virgo"``,
        ...), a model name (:func:`available_asds`) or any
        :func:`resolve_asd` spec; a list gives one per detector. Unset,
        named known detectors get their design curves, else white noise.
    sample_rate : float
        Samples per second.
    batch : int or None
        Number of samples; ``None`` omits the axis.
    detectors : int or sequence or None
        Number of detectors or their names; ``None`` omits the axis
        unless ``asd`` is a list. Only the length shapes the output.
    domain : {"td", "fd"}
        Time series, or its ``rfft(norm="forward")`` with
        ``N // 2 + 1`` bins.
    seed : int or None
        Seed; ``None`` draws fresh entropy.
    unit_psd : bool
        For ``asd=None``: one-sided PSD 1 (``sigma = sqrt(fs / 2)``) if
        True, unit variance if False.
    low_frequency_cutoff : float or None
        Frequency (Hz) below which the ASD is zero; ``None`` is PyCBC's
        default for model names and no cutoff otherwise.
    filter_duration : float
        Colouring-filter length and edge padding (s).
    numpy : bool
        Return a NumPy array instead of a tensor.
    device : str or torch.device or None
        Device; ``None`` is PyTorch's default.
    threads : int
        Threads of the CPU random draw; the noise does not depend on it.
    dtype : torch.dtype
        Real dtype; ``domain="fd"`` returns the matching complex type.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Shape ``(batch, detectors, N)`` (leading axes as requested), or
        ``N // 2 + 1`` complex bins for ``domain="fd"``, with
        ``N = round(duration * sample_rate)``.

    Raises
    ------
    ValueError
        If the duration, domain, number of ASDs or a detector's curve
        is invalid.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if domain not in ("td", "fd"):
        raise ValueError(f"domain must be 'td' or 'fd', got {domain!r}")

    n_time = int(round(duration * sample_rate))
    if n_time < 1:
        raise ValueError(
            f"duration {duration} s at {sample_rate} Hz rounds to {n_time} samples"
        )

    # "auto", plain names and an unset ASD become model names
    asd = _expand_asd_spec(asd, detectors)

    n_det = None
    if detectors is not None:
        n_det = detectors if isinstance(detectors, int) else len(detectors)
        _check_detector_asd_match(asd, detectors)
    elif isinstance(asd, list):
        # One ASD per detector sets the detector axis
        n_det = len(asd)

    lead = []
    if batch is not None:
        lead.append(int(batch))
    if n_det is not None:
        lead.append(n_det)
    shape = tuple(lead) + (n_time,)

    # One spec per detector, each resolved on the padded grid and
    # stacked to (D, F); a tuple is a (freqs, values) pair, not a list
    if isinstance(asd, list):
        if n_det is None or len(asd) != n_det:
            raise ValueError(
                f"got {len(asd)} per-detector ASDs but detectors="
                f"{detectors!r}; pass one ASD per detector, or a single ASD "
                f"(as an array, not a list) to share across detectors"
            )
        pad_seconds = filter_duration
        n_full = _fast_fft_len(n_time + 2 * int(round(pad_seconds * sample_rate)))
        freqs = torch.fft.rfftfreq(n_full, d=1.0 / sample_rate, device=device,
                                   dtype=torch.float64)
        asd = torch.stack([
            _resolve_asd_cached(spec, freqs=freqs, sample_rate=sample_rate,
                                is_psd=False,
                                low_frequency_cutoff=_default_cutoff(
                                    spec, low_frequency_cutoff),
                                filter_duration=filter_duration,
                                device=device, dtype=dtype)
            for spec in asd
        ])
        # Resolved and cut off already; only the padding is kept
        low_frequency_cutoff, filter_duration = None, None
    else:
        pad_seconds = filter_duration

    low_frequency_cutoff = _default_cutoff(asd, low_frequency_cutoff)

    gen = _make_generator(_derive_seed(seed) if seed is not None else None, device)

    if asd is None:
        if domain == "fd":
            out = white_noise_fd(shape, sample_rate, generator=gen,
                                 unit_psd=unit_psd, device=device,
                                 threads=threads, dtype=dtype)
        else:
            out = white_noise_td(shape, sample_rate, generator=gen,
                                 unit_psd=unit_psd, device=device,
                                 threads=threads, dtype=dtype)
    else:
        out = coloured_noise_td(
            asd, shape, sample_rate, generator=gen,
            low_frequency_cutoff=low_frequency_cutoff,
            filter_duration=filter_duration, pad_seconds=pad_seconds,
            device=device, threads=threads, dtype=dtype,
        )
        if domain == "fd":
            out = torch.fft.rfft(out, dim=-1, norm="forward")

    return out.cpu().numpy() if numpy else out


def available_asds(search=None):
    """
    Analytic ASD model names known to PyCBC (LALSimulation curves).

    Parameters
    ----------
    search : str or None
        Case-insensitive substring filter; ``None`` returns every name.

    Returns
    -------
    list of str
        Sorted model names.
    """
    from pycbc.psd import get_psd_model_list

    names = sorted(get_psd_model_list())
    if search is not None:
        names = [n for n in names if search.lower() in n.lower()]
    return names




def _spawn_rng(seed, *keys) -> np.random.Generator:
    """
    Independent NumPy generator addressed by ``(seed, *keys)``.

    Built from ``SeedSequence`` spawn keys, so distinct keys give
    independent streams.

    Parameters
    ----------
    seed : int or None
        Root entropy; ``None`` draws fresh OS entropy.
    *keys : int
        Non-negative stream coordinates, e.g. ``(block,)``.

    Returns
    -------
    numpy.random.Generator
        The generator.
    """
    return np.random.default_rng(
        np.random.SeedSequence(entropy=seed, spawn_key=tuple(int(k) for k in keys))
    )


def _derive_seed(seed, *keys) -> int:
    """
    Reproducible non-negative 63-bit seed from ``(seed, *keys)``.

    Parameters
    ----------
    seed : int or None
        Root entropy.
    *keys : int
        Non-negative stream coordinates.

    Returns
    -------
    int
        The seed.
    """
    ss = np.random.SeedSequence(entropy=seed, spawn_key=tuple(int(k) for k in keys))
    return int(ss.generate_state(2, dtype=np.uint64)[0] >> 1)  # keep it positive


def _legacy_block(seed_offset: int, block_index: int, sample_rate: float,
                  block_samples: int) -> np.ndarray:
    """
    One white block identical to ``pycbc.noise.reproduceable.block``.

    Parameters
    ----------
    seed_offset : int
        PyCBC's seed offset of the stream.
    block_index : int
        Block index.
    sample_rate : float
        Samples per second; the variance is ``sample_rate / 2``.
    block_samples : int
        Block length.

    Returns
    -------
    numpy.ndarray, shape ``(block_samples,)``
        The block.
    """
    rng = np.random.RandomState((seed_offset + block_index) % 2**32)
    return rng.normal(size=block_samples, scale=(sample_rate / 2.0) ** 0.5)




def _td_sigma(sample_rate: float, unit_psd: bool) -> float:
    """
    Time-domain standard deviation of white noise.

    Parameters
    ----------
    sample_rate : float
        Samples per second.
    unit_psd : bool
        One-sided PSD 1 (``sqrt(fs / 2)``) if True, else 1.

    Returns
    -------
    float
        The standard deviation.
    """
    return math.sqrt(sample_rate / 2.0) if unit_psd else 1.0




def white_series(start, end, sample_rate, seed=0, *, unit_psd=True,
                 legacy_pycbc=False, block_samples=BLOCK_SAMPLES):
    """
    Block-addressable white Gaussian noise over a time interval.

    Sample ``t`` comes from block ``floor(t / block_dur)``, each block
    seeded by its index, so overlapping requests agree exactly and any
    interval can be drawn alone (PyCBC ``reproduceable.normal``).

    Parameters
    ----------
    start, end : float
        Interval bounds in seconds; may be negative.
    sample_rate : float
        Samples per second; fixed for continuity across calls.
    seed : int or None
        Root seed.
    unit_psd : bool
        ``sigma = sqrt(fs / 2)`` (one-sided PSD 1) if True, else 1.
    legacy_pycbc : bool
        Reproduce ``pycbc.noise.reproduceable.normal`` bit for bit;
        exact for sample-aligned bounds (offsets are rounded here and
        floored by PyCBC).
    block_samples : int
        Block length.

    Returns
    -------
    numpy.ndarray, shape ``(round((end - start) * sample_rate),)``
        The noise.

    Raises
    ------
    ValueError
        If ``end <= start``.
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be greater than start ({start})")

    block_dur = block_samples / sample_rate
    first = int(np.floor(start / block_dur))
    # Block range from the requested sample indices, so the slice never
    # runs past the assembled blocks
    i0 = int(round((start - first * block_dur) * sample_rate))
    i1 = i0 + int(round((end - start) * sample_rate))
    last = first + (i1 - 1) // block_samples

    if legacy_pycbc:
        seed_offset = np.random.RandomState(seed).randint(-2**50, 2**50)
        blocks = [
            _legacy_block(seed_offset, i, sample_rate, block_samples)
            for i in range(first, last + 1)
        ]
    else:
        sigma = _td_sigma(sample_rate, unit_psd)
        blocks = [
            _spawn_rng(seed, i + _BLOCK_KEY_OFFSET).normal(
                scale=sigma, size=block_samples
            )
            for i in range(first, last + 1)
        ]

    data = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]

    out = data[i0:i1]

    if legacy_pycbc and not unit_psd:
        out = out / _td_sigma(sample_rate, True)

    return out


def coloured_series(asd, start, end, sample_rate, seed=0, *,
                    low_frequency_cutoff=None, filter_duration=PYCBC_FILTER_DURATION,
                    is_psd=False, scale=1.0, legacy_pycbc=False,
                    block_samples=BLOCK_SAMPLES):
    """
    Block-addressable Gaussian noise coloured by an ASD.

    PyCBC's ``colored_noise``: :func:`white_series` noise padded by
    ``filter_duration`` at each end is coloured in the frequency domain
    by the truncated ASD, then the padding is cropped.

    Parameters
    ----------
    asd : AsdSpec
        Any :func:`resolve_asd` spec.
    start, end : float
        Interval bounds in seconds.
    sample_rate : float
        Samples per second.
    seed : int or None
        Root seed, as in :func:`white_series`.
    low_frequency_cutoff : float or None
        Frequency (Hz) below which the ASD is zero; ``None`` as in
        :func:`sample_synthetic_noise`.
    filter_duration : float
        Colouring-filter length and edge padding (s).
    is_psd : bool
        ``asd`` is a PSD.
    scale : float
        Factor multiplying the noise.
    legacy_pycbc : bool
        PyCBC-identical white noise (:func:`white_series`).
    block_samples : int
        Block length of :func:`white_series`.

    Returns
    -------
    numpy.ndarray, shape ``(round((end - start) * sample_rate),)``
        The noise.
    """
    white = white_series(
        start - filter_duration, end + filter_duration,
        sample_rate=sample_rate, seed=seed, unit_psd=True,
        legacy_pycbc=legacy_pycbc, block_samples=block_samples,
    )

    n = len(white)
    x = torch.from_numpy(white).to(dtype=torch.float64)
    X = torch.fft.rfft(x, norm="forward")

    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate, dtype=torch.float64)
    asd_t = resolve_asd(
        asd, freqs=freqs, sample_rate=sample_rate, is_psd=is_psd,
        low_frequency_cutoff=_default_cutoff(asd, low_frequency_cutoff),
        filter_duration=filter_duration, dtype=torch.float64,
    )

    n_pad = int(round(filter_duration * sample_rate))
    n_out = int(round((end - start) * sample_rate))

    # None, "flat", "white" and "none" resolve to no colouring
    if asd_t is None:
        return (x[n_pad:n_pad + n_out] * scale).numpy()

    coloured = torch.fft.irfft(X * asd_t * scale, n=n, norm="forward")
    return coloured[n_pad:n_pad + n_out].numpy()


def feather(old_tail: torch.Tensor, new_head: torch.Tensor) -> torch.Tensor:
    """
    Power-complementary crossfade of two independent noise segments.

    LAL's ``XLALSimNoise`` feathering, ``out[j] = cos(pi j / 2L) old[j]
    + sin(pi j / 2L) new[j]``, which keeps the PSD of either input; it
    stitches coloured segments into a non-periodic stream.

    Parameters
    ----------
    old_tail, new_head : torch.Tensor
        Equal-shape overlap of the outgoing and incoming segments,
        blended along the last axis.

    Returns
    -------
    torch.Tensor
        The blended overlap.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    if old_tail.shape != new_head.shape:
        raise ValueError(
            f"feather() needs equal shapes, got {tuple(old_tail.shape)} "
            f"and {tuple(new_head.shape)}"
        )
    length = old_tail.shape[-1]
    j = torch.arange(length, device=old_tail.device, dtype=old_tail.dtype)
    phase = math.pi * j / (2.0 * length)
    return torch.cos(phase) * old_tail + torch.sin(phase) * new_head




AsdSpec = Union[
    None, float, str, Sequence, np.ndarray, torch.Tensor,
    Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]],
    Callable[[torch.Tensor], torch.Tensor],
]


def resolve_asd(spec: AsdSpec, *, freqs: torch.Tensor, sample_rate: float,
                is_psd: bool = False, low_frequency_cutoff: Optional[float] = None,
                filter_duration: Optional[float] = None,
                n_detectors: Optional[int] = None,
                device=None, dtype=torch.float32) -> Optional[torch.Tensor]:
    """
    Resolve an ASD specification onto a frequency grid.

    A PSD built from a float32 array warns: at strain scale it has
    already flushed to zero.

    Parameters
    ----------
    spec : AsdSpec
        ``None``, ``"flat"``, ``"white"`` or ``"none"`` (no
        colouring); a scalar (constant); an array ``(F,)`` or
        ``(D, F)`` on the grid, used as is; a ``(freqs, values)``
        pair, interpolated in log; a callable ``fn(freqs)``; or a model
        name, evaluated as a PSD by ``pycbc.psd.from_string``.
    freqs : torch.Tensor, shape ``(F,)``
        Target grid (Hz), uniform from 0.
    sample_rate : float
        Samples per second.
    is_psd : bool
        ``spec`` is a PSD, so its square root is taken; set for model
        names.
    low_frequency_cutoff : float or None
        Frequency (Hz) below which the ASD is zero; also the model's
        lower frequency.
    filter_duration : float or None
        If given, the PSD is inverse-spectrum truncated to a filter of
        this many seconds (PyCBC ``colored_noise``).
    n_detectors : int or None
        Broadcast a 1-D result to ``(n_detectors, F)``.
    device : str or torch.device or None
        Device of the result.
    dtype : torch.dtype
        Dtype of the result.

    Returns
    -------
    torch.Tensor or None
        ASD of shape ``(F,)`` or ``(D, F)``; ``None`` for no colouring.

    Raises
    ------
    ValueError
        If an array does not have ``F`` bins.
    """
    if spec is None:
        return None
    if isinstance(spec, str) and spec.lower() in ("flat", "white", "none"):
        return None

    # PSD intermediates are float64; a float32 PSD at strain scale
    # (~1e-46) has already flushed to zero
    if is_psd:
        raw = spec[1] if (isinstance(spec, tuple) and len(spec) == 2) else spec
        raw_dtype = str(getattr(raw, "dtype", ""))
        if "float" in raw_dtype and "float64" not in raw_dtype:
            warnings.warn(
                f"PSD supplied as {raw_dtype}: at strain scale a PSD (~1e-46) is "
                f"below float32's smallest normal (1.2e-38) and has already "
                f"flushed to zero.  Pass an ASD (is_psd=False) or a float64 PSD.",
                stacklevel=2,
            )

    freqs = freqs.to(device=device, dtype=torch.float64)
    n_freq = freqs.numel()
    delta_f = float(freqs[1] - freqs[0]) if n_freq > 1 else float(sample_rate)

    # PSD or ASD on the target grid
    if isinstance(spec, str):
        import pycbc.psd as _pypsd
        flow = float(low_frequency_cutoff or 0.0)
        series = _pypsd.from_string(spec, n_freq, delta_f, flow)
        values = torch.from_numpy(np.asarray(series.numpy(), dtype=np.float64))
        values = values.to(device=device)
        is_psd = True

    elif callable(spec) and not isinstance(spec, (np.ndarray, torch.Tensor)):
        values = torch.as_tensor(spec(freqs), device=device, dtype=torch.float64)

    elif isinstance(spec, tuple) and len(spec) == 2:
        values = _interp_log(_as_f64(spec[0], device), _as_f64(spec[1], device), freqs)

    elif np.ndim(spec) == 0:  # scalars and 0-d arrays
        values = torch.full((n_freq,), float(spec), dtype=torch.float64, device=device)

    else:
        values = _as_f64(spec, device)
        if values.shape[-1] != n_freq:
            raise ValueError(
                f"ASD array has {values.shape[-1]} bins but the target grid has "
                f"{n_freq}.  Pass a (freqs, values) pair if you want it "
                f"interpolated rather than used as is."
            )

    # Inverse-spectrum truncation, on the PSD
    if filter_duration is not None:
        psd = values if is_psd else values ** 2
        psd = _truncate_psd(psd, sample_rate, delta_f, filter_duration,
                            low_frequency_cutoff)
        values, is_psd = psd, True

    asd = torch.sqrt(values.clamp_min(0.0)) if is_psd else values

    if low_frequency_cutoff is not None:
        asd = torch.where(freqs < float(low_frequency_cutoff),
                          torch.zeros_like(asd), asd)

    if n_detectors is not None and asd.ndim == 1:
        asd = asd.expand(n_detectors, -1)

    return asd.to(dtype=dtype)


def _as_f64(x, device) -> torch.Tensor:
    """
    Array-like as a float64 tensor on ``device``.

    Parameters
    ----------
    x : array-like or torch.Tensor
        Values.
    device : str or torch.device or None
        Device.

    Returns
    -------
    torch.Tensor
        float64 tensor.
    """
    if isinstance(x, torch.Tensor):
        # Via torch: np.asarray() on a CUDA tensor raises
        return x.detach().to(device=device, dtype=torch.float64)
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)


def _interp_log(src_f: torch.Tensor, src_v: torch.Tensor,
                dst_f: torch.Tensor) -> torch.Tensor:
    """
    Interpolate an ASD onto ``dst_f``, clamped at the edges.

    Log-linear where both neighbours are positive, linear otherwise, so
    zero bands stay zero; exact at the source nodes.

    Parameters
    ----------
    src_f : torch.Tensor, shape ``(F_src,)``
        Source frequencies.
    src_v : torch.Tensor, shape ``(..., F_src)``
        Source ASD.
    dst_f : torch.Tensor, shape ``(F,)``
        Target frequencies.

    Returns
    -------
    torch.Tensor, shape ``(..., F)``
        ASD on ``dst_f``.
    """
    # Index the last axis, so a (D, F) ASD works
    order = torch.argsort(src_f)
    src_f, src_v = src_f[order], src_v[..., order]

    idx = torch.searchsorted(src_f, dst_f).clamp(1, src_f.numel() - 1)
    f0, f1 = src_f[idx - 1], src_f[idx]
    v0, v1 = src_v[..., idx - 1], src_v[..., idx]

    tiny = torch.finfo(src_f.dtype).tiny
    width = (f1 - f0).clamp_min(tiny)          # guards duplicated frequencies
    frac = ((dst_f - f0) / width).clamp(0.0, 1.0)

    linear = v0 + frac * (v1 - v0)
    log0, log1 = torch.log(v0.clamp_min(tiny)), torch.log(v1.clamp_min(tiny))
    log_linear = torch.exp(log0 + frac * (log1 - log0))
    return torch.where((v0 > 0) & (v1 > 0), log_linear, linear)


def _truncate_psd(psd: torch.Tensor, sample_rate: float, delta_f: float,
                  filter_duration: float,
                  low_frequency_cutoff: Optional[float]) -> torch.Tensor:
    """
    Inverse-spectrum-truncate a PSD, as PyCBC's ``colored_noise`` does.

    The inverted PSD is truncated and inverted back; zero bins are set
    to the maximum before and back to zero after.

    Parameters
    ----------
    psd : torch.Tensor, shape ``(F,)`` or ``(..., F)``
        PSD; each row is truncated separately.
    sample_rate : float
        Samples per second.
    delta_f : float
        Frequency resolution (Hz).
    filter_duration : float
        Filter length (s).
    low_frequency_cutoff : float or None
        Low-frequency cutoff of the truncation (Hz).

    Returns
    -------
    torch.Tensor
        Truncated PSD, float64, same shape.
    """
    from sage.dsp.inverse_spectrum_truncation import inverse_spectrum_truncation_single

    fil_len = int(round(filter_duration * sample_rate))
    zero_bins = psd <= 0
    safe = torch.where(zero_bins, psd.max(), psd).to(dtype=torch.float64)

    def _one(row):
        """Truncate one PSD row."""
        inverted = inverse_spectrum_truncation_single(
            1.0 / row,
            fil_len,
            low_frequency_cutoff=low_frequency_cutoff,
            delta_f=delta_f,
            trunc_method="hann",
        )
        return 1.0 / inverted.clamp_min(torch.finfo(torch.float64).tiny)

    if safe.ndim == 1:
        out = _one(safe)
    else:  # each row separately
        out = torch.stack([_one(row) for row in safe.reshape(-1, safe.shape[-1])])
        out = out.reshape(safe.shape)

    return torch.where(zero_bins, torch.zeros_like(out), out)




_RNG_POOLS = {}  # thread pools of _chunked_randn, by thread count


def _rng_chunk_count(lead: int) -> int:
    """
    Number of independent RNG chunks of a batch of ``lead`` rows.

    Depends on the batch shape only, so the noise does not depend on the
    thread count or the machine.

    Parameters
    ----------
    lead : int
        Length of the leading axis.

    Returns
    -------
    int
        1 below 8 rows, else ``min(32, lead)``.
    """
    if lead < 8:
        return 1
    return min(32, lead)


def _chunked_randn(shape, *, generator, device, dtype, threads):
    """
    Standard-normal tensor, drawn in chunks over a thread pool on CPU.

    Chunk boundaries and seeds are set by the shape, so ``threads``
    changes only the speed. CUDA draws in one call.

    Parameters
    ----------
    shape : tuple of int
        Output shape; the first axis is chunked.
    generator : torch.Generator or None
        Seeds the chunks; ``None`` draws fresh entropy.
    device : str or torch.device or None
        Device.
    dtype : torch.dtype
        Dtype.
    threads : int
        Worker threads; 1 or less draws serially.

    Returns
    -------
    torch.Tensor
        Standard-normal samples.
    """
    shape = tuple(shape)
    on_cpu = device is None or torch.device(device).type == "cpu"
    lead = shape[0] if len(shape) > 1 else 0
    n_chunks = _rng_chunk_count(lead) if on_cpu else 1

    if n_chunks <= 1:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    out = torch.empty(shape, device=device, dtype=dtype)
    edges = [lead * i // n_chunks for i in range(n_chunks + 1)]

    # Each chunk gets its own generator, seeded before the threads
    # start; the lock-guarded global RNG would serialise the pool
    if generator is not None:
        base = generator.initial_seed()
        seeds = [_derive_seed(base, i) for i in range(n_chunks)]
    else:  # unseeded: fresh entropy per chunk
        seeds = list(np.random.SeedSequence().generate_state(n_chunks,
                                                             dtype=np.uint64))

    jobs = [(seeds[i], edges[i], edges[i + 1]) for i in range(n_chunks)
            if edges[i + 1] > edges[i]]

    def _fill(job):
        """Fill rows ``lo:hi`` of ``out`` from the chunk's generator."""
        chunk_seed, lo, hi = job
        g = _make_generator(int(chunk_seed) >> 1, device)
        torch.randn((hi - lo,) + shape[1:], generator=g, out=out[lo:hi])

    if threads and threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = _RNG_POOLS.get(threads)
        if pool is None:
            pool = _RNG_POOLS[threads] = ThreadPoolExecutor(
                max_workers=threads, thread_name_prefix="sage-noise-rng"
            )
        list(pool.map(_fill, jobs))
    else:
        for job in jobs:
            _fill(job)
    return out


def _make_generator(seed, device=None):
    """
    Seeded ``torch.Generator`` on ``device``.

    Parameters
    ----------
    seed : int or None
        Seed, taken modulo ``2**63 - 1``.
    device : str or torch.device or None
        Device; ``None`` is CPU.

    Returns
    -------
    torch.Generator or None
        The generator, or ``None`` if ``seed`` is ``None``.
    """
    if seed is None:
        return None
    gen = torch.Generator(device=device if device is not None else "cpu")
    gen.manual_seed(int(seed) % (2**63 - 1))
    return gen


def white_noise_td(shape, sample_rate: float, *, seed=None, generator=None,
                   unit_psd: bool = True, device=None, threads: int = 1,
                   dtype=torch.float32) -> torch.Tensor:
    """
    White Gaussian noise in the time domain.

    Parameters
    ----------
    shape : tuple of int
        Output shape; the last axis is time.
    sample_rate : float
        Samples per second; sets the unit-PSD variance.
    seed : int or None
        Seed; ignored when ``generator`` is given.
    generator : torch.Generator or None
        Seeded generator on ``device``.
    unit_psd : bool
        ``sigma = sqrt(fs / 2)`` (one-sided PSD 1) if True, else 1.
    device : str or torch.device or None
        Device.
    threads : int
        Threads of the CPU draw (:func:`_chunked_randn`).
    dtype : torch.dtype
        Dtype.

    Returns
    -------
    torch.Tensor, shape ``shape``
        The noise.
    """
    gen = generator if generator is not None else _make_generator(seed, device)
    noise = _chunked_randn(shape, generator=gen, device=device, dtype=dtype,
                           threads=threads)
    sigma = _td_sigma(sample_rate, unit_psd)
    # In place: the largest allocation of the call
    return noise if sigma == 1.0 else noise.mul_(sigma)


def white_noise_fd(shape, sample_rate: float, *, seed=None, generator=None,
                   unit_psd: bool = True, device=None, threads: int = 1,
                   dtype=torch.float32) -> torch.Tensor:
    """
    White Gaussian noise drawn directly in the frequency domain.

    Same distribution as ``rfft(white_noise_td(...), norm="forward")``
    without the transform. DC, and Nyquist for even ``N``, are real with
    the power of the other bins.

    Parameters
    ----------
    shape : tuple of int
        Time-domain shape; the last axis is the number of samples ``N``.
    sample_rate : float
        Samples per second.
    seed : int or None
        Seed; ignored when ``generator`` is given.
    generator : torch.Generator or None
        Seeded generator on ``device``.
    unit_psd : bool
        As in :func:`white_noise_td`.
    device : str or torch.device or None
        Device.
    threads : int
        Threads of the CPU draw (:func:`_chunked_randn`).
    dtype : torch.dtype
        Real dtype; the output is the matching complex type.

    Returns
    -------
    torch.Tensor, shape ``shape[:-1] + (N // 2 + 1,)``
        Complex noise.
    """
    lead = tuple(shape[:-1])
    n_time = int(shape[-1])
    n_freq = n_time // 2 + 1
    gen = generator if generator is not None else _make_generator(seed, device)

    sigma_t = _td_sigma(sample_rate, unit_psd)
    sigma_fd = sigma_t / math.sqrt(2.0 * n_time)

    real = _chunked_randn(lead + (n_freq,), generator=gen, device=device,
                          dtype=dtype, threads=threads)
    imag = _chunked_randn(lead + (n_freq,), generator=gen, device=device,
                          dtype=dtype, threads=threads)
    noise = torch.complex(real, imag) * sigma_fd

    # DC, and Nyquist for even N, are real with the interior power
    edge_sigma = sigma_t / math.sqrt(n_time)
    dc = torch.randn(lead, generator=gen, device=device, dtype=dtype) * edge_sigma
    noise[..., 0] = torch.complex(dc, torch.zeros_like(dc))
    if n_time % 2 == 0:
        nyq = torch.randn(lead, generator=gen, device=device, dtype=dtype) * edge_sigma
        noise[..., -1] = torch.complex(nyq, torch.zeros_like(nyq))

    return noise


def colour_fd(noise_fd: torch.Tensor, asd: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Colour unit-PSD frequency-domain noise by an ASD: a multiply.

    Parameters
    ----------
    noise_fd : torch.Tensor, shape ``(..., F)``
        Unit-PSD white noise, ``rfft(norm="forward")``, complex.
    asd : torch.Tensor or None, shape ``(F,)`` or ``(D, F)``
        ASD; ``None`` returns ``noise_fd``.

    Returns
    -------
    torch.Tensor
        Coloured noise, shape of ``noise_fd``.
    """
    if asd is None:
        return noise_fd
    # Cast only if the device or dtype differs
    if asd.device != noise_fd.device or asd.dtype != noise_fd.real.dtype:
        asd = asd.to(device=noise_fd.device, dtype=noise_fd.real.dtype)
    return noise_fd * asd


def coloured_noise_td(asd, shape, sample_rate: float, *, seed=None,
                      generator=None, low_frequency_cutoff=None, is_psd=False,
                      filter_duration=PYCBC_FILTER_DURATION, pad_seconds=None,
                      device=None,
                      threads: int = 1, dtype=torch.float32) -> torch.Tensor:
    """
    Coloured Gaussian noise in the time domain.

    PyCBC's ``colored_noise`` recipe: white noise padded by
    ``pad_seconds`` at each end is coloured in the frequency domain,
    then the padding is cropped. ``asd=None`` gives unit-PSD white
    noise.

    Parameters
    ----------
    asd : AsdSpec
        Any :func:`resolve_asd` spec.
    shape : tuple of int
        Output shape; the last axis is time.
    sample_rate : float
        Samples per second.
    seed, generator, device, threads, dtype
        As in :func:`white_noise_td`.
    low_frequency_cutoff, is_psd
        Passed to :func:`resolve_asd`.
    filter_duration : float or None
        Inverse-spectrum-truncation length (s), passed to
        :func:`resolve_asd`; the default padding. ``None`` skips the
        truncation and needs ``pad_seconds``.
    pad_seconds : float or None
        Edge padding (s); ``None`` is ``filter_duration``.

    Returns
    -------
    torch.Tensor, shape ``shape``
        The noise.

    Raises
    ------
    ValueError
        If both ``pad_seconds`` and ``filter_duration`` are ``None``.
    """
    shape = tuple(shape)
    n_time = shape[-1]

    if asd is None:
        return white_noise_td(shape, sample_rate, seed=seed, generator=generator,
                              unit_psd=True, device=device, threads=threads,
                              dtype=dtype)

    if pad_seconds is None:
        if filter_duration is None:
            raise ValueError("pad_seconds is required when filter_duration is None")
        pad_seconds = filter_duration
    n_pad = int(round(pad_seconds * sample_rate))
    # 5-smooth transform length; the extra samples are cropped
    n_full = _fast_fft_len(n_time + 2 * n_pad)

    white = white_noise_td(shape[:-1] + (n_full,), sample_rate, seed=seed,
                           generator=generator, unit_psd=True, device=device,
                           threads=threads, dtype=dtype)
    spectrum = torch.fft.rfft(white, dim=-1, norm="forward")

    freqs = torch.fft.rfftfreq(n_full, d=1.0 / sample_rate, device=device,
                               dtype=torch.float64)
    asd_t = _resolve_asd_cached(asd, freqs=freqs, sample_rate=sample_rate,
                                is_psd=is_psd,
                                low_frequency_cutoff=low_frequency_cutoff,
                                filter_duration=filter_duration, device=device,
                                dtype=dtype)

    coloured = torch.fft.irfft(colour_fd(spectrum, asd_t), n=n_full, dim=-1,
                               norm="forward")
    return coloured[..., n_pad:n_pad + n_time].contiguous()




class WhiteNoiseGenerator:
    """
    Per-sample Gaussian noise of each detector, for CPU data loaders.

    Detector ``d`` of a sample draws from a stream addressed by
    ``(sample_seed, d)``.

    Parameters
    ----------
    n_detectors : int
        Detectors per sample.
    asd : AsdSpec
        Colouring (:func:`resolve_asd`); ``None`` gives white noise.
    unit_psd : bool
        One-sided PSD 1 if True, unit variance if False; white noise
        only.
    is_psd : bool
        ``asd`` is a PSD.
    low_frequency_cutoff : float or None
        Frequency (Hz) below which the ASD is zero.
    filter_duration : float
        Colouring-filter length and edge padding (s).
    """

    def __init__(self, n_detectors: int, asd=None, *, unit_psd=True, is_psd=False,
                 low_frequency_cutoff=None, filter_duration=PYCBC_FILTER_DURATION):
        self.n_detectors = int(n_detectors)
        self.asd = asd
        self.unit_psd = unit_psd
        self.is_psd = is_psd
        self.low_frequency_cutoff = low_frequency_cutoff
        self.filter_duration = filter_duration
        # Resolved ASDs by (n_full, sample_rate)
        self._asd_cache = {}

    def _resolved_asd(self, n_full, sample_rate):
        """
        The ASD on an ``n_full``-sample grid, resolved once.

        Parameters
        ----------
        n_full : int
            Padded length in samples.
        sample_rate : float
            Samples per second.

        Returns
        -------
        torch.Tensor or None
            The ASD, float64.
        """
        key = (n_full, sample_rate)
        if key not in self._asd_cache:
            freqs = torch.fft.rfftfreq(n_full, d=1.0 / sample_rate,
                                       dtype=torch.float64)
            self._asd_cache[key] = resolve_asd(
                self.asd, freqs=freqs, sample_rate=sample_rate,
                is_psd=self.is_psd,
                low_frequency_cutoff=self.low_frequency_cutoff,
                filter_duration=self.filter_duration, dtype=torch.float64,
            )
        return self._asd_cache[key]

    def generate(self, sample_length_in_num, seed=0, *, sample_rate):
        """
        Draw one noise series.

        Parameters
        ----------
        sample_length_in_num : int
            Number of samples.
        seed : int
            Stream seed.
        sample_rate : float
            Samples per second.

        Returns
        -------
        numpy.ndarray, shape ``(sample_length_in_num,)``
            The noise.
        """
        if self.asd is None:
            rng = _spawn_rng(seed)
            sigma = _td_sigma(sample_rate, self.unit_psd)
            return rng.normal(scale=sigma, size=sample_length_in_num)

        n_pad = int(round(self.filter_duration * sample_rate))
        n_full = sample_length_in_num + 2 * n_pad

        gen = _make_generator(_derive_seed(seed))
        white = white_noise_td((n_full,), sample_rate, generator=gen,
                               unit_psd=True, dtype=torch.float64)
        spectrum = colour_fd(torch.fft.rfft(white, norm="forward"),
                             self._resolved_asd(n_full, sample_rate))
        coloured = torch.fft.irfft(spectrum, n=n_full, norm="forward")
        return coloured[n_pad:n_pad + sample_length_in_num].numpy()

    def apply(self, special, det_only=""):
        """
        Noise of every detector of one sample.

        Parameters
        ----------
        special : dict
            ``"sample_seed"`` and ``"data_cfg"``, whose
            ``signal_length`` (s) and ``sample_rate`` (Hz) set the
            length.
        det_only : str
            Not used.

        Returns
        -------
        numpy.ndarray, shape ``(D, N)``
            Noise per detector.
        """
        data_cfg = special["data_cfg"]
        sample_rate = data_cfg.sample_rate
        n_samples = int(data_cfg.signal_length * sample_rate)
        seed = special["sample_seed"]
        return np.stack(
            [
                self.generate(n_samples, seed=_derive_seed(seed, d),
                              sample_rate=sample_rate)
                for d in range(self.n_detectors)
            ],
            axis=0,
        )




class WhiteGaussianNoiseSampler(torch.nn.Module):
    """
    Batch sampler of Gaussian noise with the ``MemmapNoiseSampler`` API.

    Batch ``n`` depends only on ``(seed, n)``, and :meth:`set_step`
    moves to any batch. Coloured noise is drawn white on a grid padded
    by ``filter_duration`` at each end, coloured and cropped.

    Parameters
    ----------
    asd : AsdSpec or list
        Colouring (:func:`resolve_asd`), or one spec per detector;
        ``None`` gives white noise.
    detectors : list of str
        Detectors, in channel order.
    window : Window
        Window geometry; sets the window length and sample rate.
    batch_size : int
        Windows per batch.
    seed : int
        Root seed.
    low_frequency_cutoff : float
        Frequency (Hz) below which the ASD is zero.
    filter_duration : float
        Colouring-filter length and edge padding (s).
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Dtype of the noise targets.
    domain : {"fd", "td"}
        ``(B, D, F)`` complex ``rfft(norm="forward")``, or ``(B, D, N)``
        real.
    unit_psd : bool
        One-sided PSD 1 if True, unit variance if False; white noise
        only.
    is_psd : bool
        ``asd`` is a PSD.

    Raises
    ------
    ValueError
        If ``domain`` is not ``"fd"`` or ``"td"``.

    Attributes
    ----------
    GRAPH_READY : bool
        ``False``: seeded RNG calls are not traceable by
        ``torch.compile``.
    """

    GRAPH_READY = False

    def __init__(
        self,
        asd,
        detectors: list,
        window: Window,
        batch_size: int,
        seed: int,
        low_frequency_cutoff: float,
        filter_duration: float,
        device,
        dtype: torch.dtype,
        *,
        domain: str = "fd",
        unit_psd: bool = True,
        is_psd: bool = False,
    ):
        super().__init__()

        if domain not in ("fd", "td"):
            raise ValueError(f"domain must be 'fd' or 'td', got {domain!r}")

        self.seq_len = window.padded_length_in_samples
        self.sample_rate = float(window.sample_rate_in_hz)
        self.device = device
        self.n_detectors = len(detectors)
        self.batch_size = batch_size
        self.real_dtype = torch.float32

        self.seed = int(seed)
        self.domain = domain
        self.unit_psd = unit_psd
        self.filter_duration = filter_duration
        self._step = 0

        # The ASD is resolved once, on the padded grid
        self._pad_nsamples = 0
        self._n_full = self.seq_len
        resolved = None
        if asd is not None:
            self._pad_nsamples = int(round(filter_duration * self.sample_rate))
            n_full = _fast_fft_len(self.seq_len + 2 * self._pad_nsamples)
            self._n_full = n_full
            freqs = torch.fft.rfftfreq(
                n_full, d=1.0 / self.sample_rate, device=self.device,
                dtype=torch.float64,
            )
            options = dict(
                freqs=freqs, sample_rate=self.sample_rate, is_psd=is_psd,
                low_frequency_cutoff=low_frequency_cutoff,
                filter_duration=filter_duration, device=self.device,
                dtype=self.real_dtype,
            )
            if isinstance(asd, list):
                # One spec per detector, stacked into a (D, F) ASD.
                resolved = torch.stack([_resolve_asd_cached(spec, **options) for spec in asd])
            else:
                resolved = _resolve_asd_cached(asd, n_detectors=self.n_detectors, **options)

        self._has_asd = resolved is not None
        # A buffer, so it follows the module's device and state dict
        self.register_buffer(
            "asd", resolved if resolved is not None
            else torch.empty(0, device=self.device)
        )

        self.noise_target = torch.zeros(
            (self.batch_size, 1), dtype=dtype, device=device
        )

    def set_step(self, step: int):
        """
        Move to batch ``step``; the next batch drawn is that one.

        Parameters
        ----------
        step : int
            Batch index.
        """
        self._step = int(step)

    @property
    def step(self) -> int:
        """Index of the next batch to be drawn."""
        return self._step

    def _generator(self):
        """
        Generator of the current batch.

        Returns
        -------
        torch.Generator
            Seeded by ``(seed, step)``.
        """
        return _make_generator(_derive_seed(self.seed, self._step), self.device)

    def _sample_batch(self):
        """
        Draw one batch in the configured domain.

        Returns
        -------
        torch.Tensor
            ``(B, D, F)`` complex or ``(B, D, N)`` real noise.
        """
        shape = (self.batch_size, self.n_detectors, self.seq_len)
        gen = self._generator()

        if not self._has_asd:
            if self.domain == "fd":
                return white_noise_fd(shape, self.sample_rate, generator=gen,
                                      unit_psd=self.unit_psd, device=self.device,
                                      dtype=self.real_dtype)
            return white_noise_td(shape, self.sample_rate, generator=gen,
                                  unit_psd=self.unit_psd, device=self.device,
                                  dtype=self.real_dtype)

        # White on the padded grid, coloured, then cropped
        n_pad = self._pad_nsamples
        n_full = self._n_full
        white = white_noise_td(shape[:-1] + (n_full,), self.sample_rate,
                               generator=gen, unit_psd=True, device=self.device,
                               dtype=self.real_dtype)
        spectrum = colour_fd(torch.fft.rfft(white, dim=-1, norm="forward"),
                             self.asd)
        noise_td = torch.fft.irfft(spectrum, n=n_full, dim=-1, norm="forward")
        noise_td = noise_td[..., n_pad:n_pad + self.seq_len].contiguous()

        if self.domain == "td":
            return noise_td
        return torch.fft.rfft(noise_td, dim=-1, norm="forward")

    @torch.no_grad()
    def forward(self):
        """
        Next noise batch and its zero targets; advances the step.

        Returns
        -------
        noise : torch.Tensor
            ``(B, D, F)`` complex for ``domain="fd"``, ``(B, D, N)``
            real for ``domain="td"``.
        noise_target : torch.Tensor, shape ``(B, 1)``
            Zeros.
        """
        batch = self._sample_batch()
        self._step += 1
        return batch, self.noise_target

    def shutdown(self):
        """Do nothing: there is no prefetching to stop."""

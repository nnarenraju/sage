#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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
Everything in this module follows the LAL/PyCBC one-sided PSD convention.
``S(f)`` is the one-sided power spectral density in ``1/Hz`` (strain²/Hz) and
``ASD(f) = sqrt(S(f))``.

**Time domain.**  A real series ``x[n]`` sampled at ``fs`` whose one-sided PSD
is flat at ``S`` has variance

    var(x) = S · fs / 2

so *unit-PSD* white noise has ``sigma = sqrt(fs / 2)``, **not** 1.  This is
PyCBC's ``pycbc.noise.reproduceable.normal`` convention and it is the default
here (``unit_psd=True``).  Its whole purpose is that colouring then becomes a
bare multiply by the ASD with no leftover factors anywhere in the chain.
Pass ``unit_psd=False`` for the older "zero-mean, unit-variance" behaviour.

**Frequency domain.**  Sage uses ``X = rfft(x, norm="forward")`` throughout, for
which the corresponding statement is

    E|X_k|² = S(f_k) · Δf / 2      for every k

with ``X_k`` **real** at DC and Nyquist (a real inverse FFT discards the
imaginary part of those two bins, so drawing them complex — as LAL and
``pycbc.noise.gaussian`` both do — silently halves their power).  Equivalently,
in terms of the time-domain standard deviation ``sigma_t`` of the same process:

    sigma_fd = sigma_t / sqrt(2N)   for 0 < k < N/2   (per real/imag component)
    sigma_fd = sigma_t / sqrt(N)    for k = 0 and k = N/2 (real-valued)

Both directions of this were verified numerically against PyCBC to ~1e-4.

**Reproducibility.**  ``white_series`` reproduces PyCBC's block-addressable
scheme: noise for a GPS interval is assembled from fixed-length blocks, each
seeded from its own block index, so *any* interval — requested in any order,
from any process — is bit-identical.  Blocking is done in the *white* domain
where independent blocks concatenate exactly (white noise is delta-correlated);
colouring is applied afterwards over the whole padded span.
"""

# Packages
import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# LOCAL
from sage.core.config import get_cfg, get_data_cfg


__all__ = [
    "BLOCK_SAMPLES",
    "sample_synthetic_noise",
    "available_asds",
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


# PyCBC's block size.  Kept bit-identical so that ``legacy_pycbc=True`` streams
# match ``pycbc.noise.reproduceable.normal`` exactly.  Changing it invalidates
# every dataset ever generated with it — treat as frozen.
BLOCK_SAMPLES = 1638400

# Block indices may be negative (the coloured path pads before ``start``), while
# ``SeedSequence`` spawn keys must be non-negative.  Offset covers |block| < 2^32.
_BLOCK_KEY_OFFSET = 1 << 32

# Resolved ASDs, keyed by spec + grid.  Resolution means a pycbc model lookup, an
# interpolation and possibly an inverse-spectrum truncation — all NumPy-side and
# all identical from one call to the next, so a loop drawing batches would
# otherwise repay that cost every iteration.  Entries are treated as read-only.
_ASD_CACHE = {}
_ASD_CACHE_MAX = 32


def _fast_fft_len(n: int) -> int:
    """
    Round ``n`` up to the next 5-smooth length (factors of 2, 3 and 5 only).

    FFT cost is acutely sensitive to the factorisation: a padded length that
    lands on a large prime runs several times slower than a smooth one nearby
    (measured 6.3 ms vs 1.3 ms for a 256x2 batch at n≈12k).  Padding is only ever
    cropped away, so rounding up is free correctness-wise.
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

    Falls through to an uncached resolve when the spec is unhashable (an array
    or a ``(freqs, values)`` pair) — those are already the cheap path, since
    they need no model lookup.  The returned tensor is shared, so callers must
    not mutate it; every consumer in this module only reads.
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


# ══════════════════════════════════════════════════════════════════════════════
# The easy front door
# ══════════════════════════════════════════════════════════════════════════════


# Which interferometer each detector label and each analytic model belongs to.
# Only used to warn about an obvious mismatch — nothing here changes the output.
_DETECTOR_FAMILY = {
    "H1": "LIGO", "L1": "LIGO", "I1": "LIGO", "A1": "LIGO",
    "V1": "Virgo", "K1": "KAGRA", "G1": "GEO",
    "E1": "ET", "E2": "ET", "E3": "ET", "C1": "CE",
}


# The curve each interferometer gets from ``asd="auto"``: its design sensitivity.
_DETECTOR_DEFAULT_ASD = {
    "LIGO":  "aLIGODesignSensitivityP1200087",
    "Virgo": "AdVDesignSensitivityP1200087",
    "KAGRA": "KAGRADesignSensitivityT1600593",
    "ET":    "EinsteinTelescopeP1600143",
    "CE":    "CosmicExplorerP1600143",
}

# Plain-English names for the same curves, so a caller who has never seen a
# LALSimulation model string can still ask for realistic noise.
_ASD_ALIASES = {
    "ligo": "LIGO", "aligo": "LIGO", "h1": "LIGO", "l1": "LIGO",
    "virgo": "Virgo", "advirgo": "Virgo", "adv": "Virgo", "v1": "Virgo",
    "kagra": "KAGRA", "k1": "KAGRA",
    "et": "ET", "einstein": "ET", "einstein telescope": "ET",
    "ce": "CE", "cosmic explorer": "CE", "cosmicexplorer": "CE",
}


def _model_family(name):
    """
    Which interferometer an analytic model name describes, or ``None``.

    Order matters: LALSimulation carries deprecated aliases that wrongly prefix
    other detectors' curves with ``aLIGO`` (``aLIGOAdVO4T1800545`` is an AdVirgo
    curve), so the more specific families are tested first.
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
    return None


def _expand_asd_spec(asd, detectors):
    """
    Resolve ``"auto"`` and the plain-English aliases into real model names.

    Everything else is handed back untouched, so an exact LALSimulation name, an
    array, a ``(freqs, values)`` pair or ``None`` all behave exactly as before.
    """
    def _one(spec):
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
        # Naming real detectors and saying nothing about the spectrum means you
        # want those detectors' noise.  Pass asd="white" for white noise instead.
        return [_DETECTOR_DEFAULT_ASD[f] for f in families]

    return _one(asd)


def _check_detector_asd_match(asd, detectors):
    """
    Reject a named detector paired with another interferometer's curve.

    Naming H1 and then handing it an AdVirgo model is a misunderstanding, not a
    preference, so it raises rather than warning.  Either half alone is fine —
    name the detectors and let one ASD cover them, or pass the ASDs as a list and
    let their count set the detector axis.  Passing both is fine too; they just
    have to agree.

    Only fires when it is certain: both the label and the model name have to be
    recognisable and disagree.  Integer ``detectors``, caller-supplied arrays and
    unfamiliar labels all pass without comment.
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
    Choose a low-frequency cutoff for ``spec`` when the caller gave none.

    Analytic models only.  They rise near-vertically into the seismic wall, so
    evaluating one down to 0 Hz buries the sample under low-frequency power some
    sixteen orders of magnitude above the mid-band.  An ASD the caller supplied
    is returned untouched — their array is their business.
    """
    if low_frequency_cutoff is not None:
        return low_frequency_cutoff
    if not isinstance(spec, str) or spec.lower() in ("flat", "white", "none"):
        return None
    try:
        data_cfg = get_data_cfg()
        return float(
            getattr(data_cfg, "noise_low_frequency_cutoff", None)
            or getattr(data_cfg, "signal_low_frequency_cutoff", None)
            or 10.0
        )
    except Exception:
        return 10.0


def sample_synthetic_noise(duration, asd=None, *, sample_rate=None, batch=None,
                           detectors=None, domain="td", seed=None,
                           unit_psd=True, low_frequency_cutoff=None,
                           filter_duration=None, numpy=False, device=None,
                           threads=1, dtype=torch.float32):
    """
    Draw a synthetic noise sample of a given duration.  The one to reach for.

    Sample white, colour by an ASD, hand it back.  Everything else in this
    module is the machinery underneath.

    Examples
    --------
    Four seconds of white noise::

        from sage.data.noise import sample_synthetic_noise
        x = sample_synthetic_noise(4.0)               # -> (8192,) at 2048 Hz

    Four seconds coloured by a detector ASD (see :func:`available_asds`)::

        x = sample_synthetic_noise(4.0, "aLIGOZeroDetHighPower")

    A batch for two detectors, reproducibly::

        x = sample_synthetic_noise(4.0, "aLIGOZeroDetHighPower",
                                   batch=32, detectors=2, seed=0)  # (32,2,8192)

    Straight to the frequency domain, in sage's ``rfft(norm="forward")``
    convention::

        X = sample_synthetic_noise(4.0, domain="fd")  # -> (4097,) complex

    Parameters
    ----------
    duration : float
        Length of the sample in **seconds**.
    asd : optional
        What to colour the noise with.  Accepts ``"auto"``, a detector's ordinary
        name (``"LIGO"``, ``"Virgo"``, …), an analytic model name
        (``"aLIGOZeroDetHighPower"`` — see :func:`available_asds`), an array on
        the output grid, a ``(freqs, values)`` pair on any grid, a callable
        ``f -> ASD``, or a scalar.  Full list in :func:`resolve_asd`.

        Left unset it follows the detectors: name real ones and you get their
        design curves, because nobody ever saw white noise come out of H1.  With
        an unrecognised label, or a plain count, there is nothing to infer from
        and you get white noise.  Say ``asd="white"`` for white noise regardless.

        Pass a **list** of those, one per detector, to give each detector its
        own noise curve — the one way detectors genuinely differ for noise::

            sample_synthetic_noise(4.0, ["aLIGOZeroDetHighPower", "AdVO4T1800545"],
                                   detectors=["H1", "V1"])
    sample_rate : float or None
        Samples per second.  Defaults to the registered data config's
        ``sample_rate`` if there is one, else 2048 Hz.
    batch : int or None
        Number of independent samples.  ``None`` (default) omits the axis.
    detectors : int or sequence or None
        Number of detectors, or a list like ``["H1", "L1"]``.  ``None`` (default)
        omits the axis — unless ``asd`` is a list, which sets it on its own.

        Only the **length** affects the output; the names are labels.  Nothing
        detector-specific is applied beyond the ASD you supply, because nothing
        else about noise is detector-specific (antenna patterns and time delays
        act on a signal, not on instrumental noise).  Naming a detector and then
        handing it another interferometer's curve — ``detectors=["H1"]`` with an
        AdVirgo model — raises, since that is a misunderstanding rather than a
        preference.  Give a count instead of names if the pairing is deliberate.
    domain : {"td", "fd"}
        ``"td"`` (default) returns a real time series; ``"fd"`` returns the
        complex ``rfft(norm="forward")`` spectrum with ``N // 2 + 1`` bins.
    seed : int or None
        ``None`` (default) draws fresh entropy each call.  Any integer makes the
        call reproducible.
    unit_psd : bool
        ``True`` (default) → one-sided PSD ≡ 1, ``sigma = sqrt(fs/2)``.
        ``False`` → unit variance.  Ignored when ``asd`` is given, since the
        output is then in the ASD's own units.
    low_frequency_cutoff : float or None
        Zero the ASD below this frequency (Hz).  For an **analytic model name**
        this defaults to the registered config's noise cutoff, or 10 Hz — the
        analytic curves rise near-vertically into the seismic wall, and
        evaluating one down to 0 Hz produces low-frequency power some sixteen
        orders of magnitude above the mid-band that swamps the sample.  For an
        ASD you supplied yourself, the default is ``None``: your array is used
        as given, untouched.
    filter_duration : float or None
        Inverse-spectrum-truncation length in seconds.  ``None`` (default)
        applies no truncation, which colours with the exact ASD.
    numpy : bool
        Return a NumPy array instead of a torch tensor.
    device : optional
        Where to generate.  Defaults to the registered config's ``device``, so a
        configured run generates straight onto the GPU rather than building the
        batch on the CPU and copying it over.
    threads : int
        Worker threads for the CPU random draw, which is otherwise the bottleneck
        — torch's CPU normal generator is single-threaded, and on a big batch it
        is roughly two thirds of the call.  ``1`` (default) stays serial.
        Raising it gives most of that back (measured on a 256x2x30720 draw:
        86 ms serial, 11.9 ms at 8, 3.7 ms at 32).

        **Results do not depend on this.** Chunk boundaries and per-chunk seeds
        are fixed by the batch shape, so a given seed produces identical noise at
        any thread count, on any machine.  It is purely a speed knob.

        Default off because a thread pool inside a DataLoader worker or beside a
        training loop competes with the rest of the process for cores — turn it
        up for standalone generation, leave it at 1 in-loop.  Ignored on CUDA,
        whose generator is already parallel.
    dtype : torch dtype
        The *real* dtype, ``float32`` by default; ``domain="fd"`` returns the
        matching complex type.  Keep it at float32 — ``randn`` in float64 is
        ~6x slower and an ASD needs nothing beyond single precision.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Shape is ``(...,  N)`` for ``domain="td"`` and ``(..., N // 2 + 1)``
        complex for ``domain="fd"``, where the leading axes are whichever of
        ``batch`` and ``detectors`` you asked for, in that order, and
        ``N = round(duration * sample_rate)``.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if domain not in ("td", "fd"):
        raise ValueError(f"domain must be 'td' or 'fd', got {domain!r}")

    if sample_rate is None:
        try:
            sample_rate = float(get_data_cfg().sample_rate)
        except Exception:
            sample_rate = 2048.0

    # Follow the run's device by default — generating a big batch on the CPU and
    # copying it across is far slower than generating it where it will be used.
    if device is None:
        try:
            device = get_cfg().device
        except Exception:
            device = None

    n_time = int(round(duration * sample_rate))
    if n_time < 1:
        raise ValueError(
            f"duration {duration} s at {sample_rate} Hz rounds to {n_time} samples"
        )

    # "auto" and the plain-English aliases become real model names here, so
    # everything downstream sees one kind of spec.
    asd = _expand_asd_spec(asd, detectors)

    n_det = None
    if detectors is not None:
        n_det = detectors if isinstance(detectors, int) else len(detectors)
        _check_detector_asd_match(asd, detectors)
    elif isinstance(asd, list):
        # A list of ASDs is one per detector, so it sets the detector axis on its
        # own — no need to state the count twice.
        n_det = len(asd)

    lead = []
    if batch is not None:
        lead.append(int(batch))
    if n_det is not None:
        lead.append(n_det)
    shape = tuple(lead) + (n_time,)

    # A list of specs means one ASD per detector.  Resolve each on the padded
    # grid the colouring actually runs on and stack them into a (D, F) ASD,
    # which then broadcasts across the batch axis.  Tuples are left alone —
    # those are the (freqs, values) form.
    if isinstance(asd, list):
        if n_det is None or len(asd) != n_det:
            raise ValueError(
                f"got {len(asd)} per-detector ASDs but detectors="
                f"{detectors!r}; pass one ASD per detector, or a single ASD "
                f"(as an array, not a list) to share across detectors"
            )
        pad_seconds = (filter_duration if filter_duration is not None
                       else duration / 4.0)
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
        # Already resolved, already cut off — do not re-apply either downstream.
        low_frequency_cutoff, filter_duration = None, None

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
            filter_duration=filter_duration, device=device, threads=threads,
            dtype=dtype,
        )
        if domain == "fd":
            out = torch.fft.rfft(out, dim=-1, norm="forward")

    return out.cpu().numpy() if numpy else out


def available_asds(search=None):
    """
    List the analytic ASD model names accepted by :func:`sample_synthetic_noise`.

    These are the LALSimulation / PyCBC analytic noise curves — aLIGO, AdVirgo,
    KAGRA, Einstein Telescope, Cosmic Explorer and friends, at various observing
    runs and sensitivities.

    Examples
    --------
    ::

        from sage.data.noise import available_asds
        available_asds("O4")        # every O4-era curve
        available_asds("aLIGO")     # every aLIGO curve

    Parameters
    ----------
    search : str or None
        Case-insensitive substring filter.  ``None`` returns everything.

    Returns
    -------
    list of str
        Sorted model names, each usable directly as
        ``sample_synthetic_noise(duration, name)``.

    Notes
    -----
    Requires ``pycbc``.
    """
    from pycbc.psd import get_psd_model_list

    names = sorted(get_psd_model_list())
    if search is not None:
        names = [n for n in names if search.lower() in n.lower()]
    return names


# ══════════════════════════════════════════════════════════════════════════════
# Seeding
# ══════════════════════════════════════════════════════════════════════════════


def _spawn_rng(seed, *keys) -> np.random.Generator:
    """
    Build an independent NumPy generator addressed by ``(seed, *keys)``.

    Uses ``SeedSequence`` spawn keys rather than PyCBC's ``RandomState(sv + i)``
    trick.  Both give random access; ``SeedSequence`` additionally guarantees
    that distinct keys yield statistically independent streams, whereas
    consecutive Mersenne-Twister seeds carry no such guarantee.

    Parameters
    ----------
    seed : int or None
        Root entropy.  ``None`` draws fresh OS entropy (non-reproducible).
    *keys : int
        Non-negative stream coordinates, e.g. ``(block,)`` or ``(step, det)``.

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(
        np.random.SeedSequence(entropy=seed, spawn_key=tuple(int(k) for k in keys))
    )


def _derive_seed(seed, *keys) -> int:
    """Derive a reproducible 64-bit integer seed from ``(seed, *keys)``."""
    ss = np.random.SeedSequence(entropy=seed, spawn_key=tuple(int(k) for k in keys))
    return int(ss.generate_state(2, dtype=np.uint64)[0] >> 1)  # keep it positive


def _legacy_block(seed_offset: int, block_index: int, sample_rate: float,
                  block_samples: int) -> np.ndarray:
    """
    One PyCBC-bit-identical white block (``pycbc.noise.reproduceable.block``).

    Reproduces the legacy ``RandomState((sv + i) % 2**32)`` stream exactly so
    that datasets built with PyCBC can be regenerated here without drift.
    """
    rng = np.random.RandomState((seed_offset + block_index) % 2**32)
    return rng.normal(size=block_samples, scale=(sample_rate / 2.0) ** 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════


def _td_sigma(sample_rate: float, unit_psd: bool) -> float:
    """Time-domain standard deviation of the requested white-noise convention."""
    return math.sqrt(sample_rate / 2.0) if unit_psd else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Reproducible, GPS-addressable streams  (NumPy — for dataset generation)
# ══════════════════════════════════════════════════════════════════════════════


def white_series(start, end, sample_rate=2048.0, seed=0, *, unit_psd=True,
                 legacy_pycbc=False, block_samples=BLOCK_SAMPLES):
    """
    Block-addressable white Gaussian noise over a GPS interval.

    Noise for time ``t`` is drawn from block ``floor(t / block_dur)``, each block
    seeded from its own index.  Consequences:

    * Overlapping requests agree **exactly** — ``white_series(100, 200)`` and
      ``white_series(150, 250)`` are bit-identical on their shared 50 s.
    * Any interval can be generated without materialising the ones before it.
    * The result is independent of how the request was chunked.

    Concatenating independent blocks is *exact* for white noise (it is
    delta-correlated), which is precisely why the blocking happens here and not
    after colouring — see :func:`coloured_series`.

    Parameters
    ----------
    start, end : float
        Interval bounds in seconds (GPS or otherwise).  May be negative.
    sample_rate : float
        Samples per second.  Must be held fixed for continuity across calls.
    seed : int or None
        Root seed.  Hashed before use, so neighbouring seeds do not produce
        neighbouring (overlapping) streams.
    unit_psd : bool
        ``True`` (default) → ``sigma = sqrt(fs/2)``, one-sided PSD ≡ 1.
        ``False`` → ``sigma = 1``, one-sided PSD ≡ ``2/fs``.
    legacy_pycbc : bool
        Reproduce ``pycbc.noise.reproduceable.normal`` bit-for-bit (legacy
        ``RandomState`` streams).  Implies the unit-PSD convention.  Exact for
        sample-aligned bounds only: a sub-sample offset is rounded to the
        nearest sample here (per the Returns contract below) whereas PyCBC's
        ``time_slice`` floors it, which shifts the window by one sample.  The
        underlying block streams are identical either way.
    block_samples : int
        Block length.  Only change it if you never need to match older data.

    Returns
    -------
    numpy.ndarray, shape ``(round((end - start) * sample_rate),)``
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be greater than start ({start})")

    block_dur = block_samples / sample_rate
    first = int(np.floor(start / block_dur))
    # Derive the block range from the sample indices actually requested, so the
    # slice below can never run past the assembled data (which would silently
    # return a short array).  This subsumes the "end lands exactly on a block
    # boundary" case: such an end maps to i1 == k * block_samples, needing k
    # blocks, not k + 1.
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

    # Slice the requested window out of the assembled blocks.
    out = data[i0:i1]

    if legacy_pycbc and not unit_psd:
        out = out / _td_sigma(sample_rate, True)

    return out


def coloured_series(asd, start, end, sample_rate=2048.0, seed=0, *,
                    low_frequency_cutoff=None, filter_duration=128.0,
                    is_psd=False, scale=1.0, legacy_pycbc=False,
                    block_samples=BLOCK_SAMPLES):
    """
    Block-addressable Gaussian noise coloured by an arbitrary ASD.

    Follows PyCBC's ``colored_noise``: white noise is generated over the span
    padded by ``filter_duration`` at each end, coloured in the frequency domain
    by an inverse-spectrum-truncated ASD, and the padding is then cropped off.
    The padding is what removes the circular-convolution wraparound that a bare
    FFT colouring would leave at the edges.

    Parameters
    ----------
    asd : see :func:`resolve_asd`
        Anything ``resolve_asd`` accepts — array, ``(freqs, values)`` pair,
        callable, analytic PSD name, or scalar.
    start, end : float
        Interval bounds in seconds.
    sample_rate : float
        Samples per second.
    seed : int or None
        Root seed, shared with :func:`white_series`.
    low_frequency_cutoff : float or None
        Zero the ASD below this frequency (Hz).
    filter_duration : float
        Length (s) of the colouring filter, used both for the edge padding and
        for the inverse-spectrum truncation.
    is_psd : bool
        Set when ``asd`` is really a PSD (``1/Hz``) rather than an ASD.
    scale : float
        Extra multiplicative factor applied to the ASD.
    legacy_pycbc : bool
        Use PyCBC-bit-identical white noise underneath.
    block_samples : int
        Forwarded to :func:`white_series`.

    Returns
    -------
    numpy.ndarray, shape ``(round((end - start) * sample_rate),)``
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
        low_frequency_cutoff=low_frequency_cutoff,
        filter_duration=filter_duration, dtype=torch.float64,
    )

    n_pad = int(round(filter_duration * sample_rate))
    n_out = int(round((end - start) * sample_rate))

    # ``resolve_asd`` returns None for None / "flat" / "white" — no colouring.
    if asd_t is None:
        return (x[n_pad:n_pad + n_out] * scale).numpy()

    coloured = torch.fft.irfft(X * asd_t * scale, n=n, norm="forward")
    return coloured[n_pad:n_pad + n_out].numpy()


def feather(old_tail: torch.Tensor, new_head: torch.Tensor) -> torch.Tensor:
    """
    Power-complementary crossfade between two independent noise realisations.

    This is LAL's ``XLALSimNoise`` feathering, transcribed exactly::

        x = cos(pi * j / (2 * L))
        y = sin(pi * j / (2 * L))
        out[j] = x * old[j] + y * new[j]

    Because ``x² + y² = 1`` and the two inputs are independent, the blend has
    the same PSD as either input — verified against PyCBC's LAL path to within
    0.1% over 7040 frequency bins.  Use it to stitch successive *coloured*
    segments into a long non-periodic stream (a single FFT-generated coloured
    segment is periodic; white segments need no feathering at all).

    Parameters
    ----------
    old_tail, new_head : torch.Tensor
        Equal-length overlapping regions from the outgoing and incoming
        segments.  Blending happens along the last axis.

    Returns
    -------
    torch.Tensor
        The blended overlap region, same shape as the inputs.
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


# ══════════════════════════════════════════════════════════════════════════════
# ASD resolution
# ══════════════════════════════════════════════════════════════════════════════


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
    Turn any reasonable ASD specification into a tensor on the target grid.

    Accepted forms
    --------------
    ``None`` or ``"flat"`` / ``"white"``
        No colouring (returns ``None``).
    ``float``
        Constant ASD at that level.
    1-D array of length ``F``, or 2-D array ``(D, F)``
        Used **as is** — no interpolation, so the grid must already match.
    ``(freqs, values)`` pair
        Interpolated onto the target grid (linear in ``log(ASD)`` vs ``f``,
        which is well behaved across the many decades a detector ASD spans).
        Requesting an interpolation is the explicit act of passing a grid.
    callable
        Called as ``fn(freqs)`` and expected to return an ASD on that grid.
    ``str``
        An analytic model name resolved through ``pycbc.psd.from_string``
        (e.g. ``"aLIGOZeroDetHighPower"``), evaluated exactly on the target
        grid — no interpolation involved.  See :func:`available_asds`.

    Parameters
    ----------
    spec : see above
    freqs : torch.Tensor, shape ``(F,)``
        Target frequency grid in Hz.  Must start at 0 and be uniform.
    sample_rate : float
        Samples per second, needed for analytic-PSD lookups.
    is_psd : bool
        Interpret ``spec`` as a PSD (``1/Hz``) and take its square root.
        Forced ``True`` for analytic model names, where the value passed is
        ignored.
    low_frequency_cutoff : float or None
        Zero the result below this frequency.
    filter_duration : float or None
        If given, inverse-spectrum-truncate the PSD to a filter of this many
        seconds before taking the ASD (PyCBC's ``colored_noise`` behaviour).
    n_detectors : int or None
        If given and the resolved ASD is 1-D, broadcast to ``(D, F)``.
    device, dtype
        Output placement.

    Returns
    -------
    torch.Tensor, shape ``(F,)`` or ``(D, F)``, or ``None`` for no colouring.
    """
    if spec is None:
        return None
    if isinstance(spec, str) and spec.lower() in ("flat", "white", "none"):
        return None

    # Everything downstream of here is ASD-domain, and PSD-domain intermediates
    # are held in float64 — so nothing underflows.  The one case we cannot fix is
    # a PSD that arrived already underflowed: at strain scale a PSD is ~1e-46,
    # far below float32's smallest normal (1.2e-38), so a float32 PSD array is
    # all zeros before it ever reaches us.
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

    # ── Resolve to a PSD or ASD on the target grid ────────────────────────────
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

    elif np.ndim(spec) == 0:  # covers Python/NumPy scalars and 0-d arrays alike
        values = torch.full((n_freq,), float(spec), dtype=torch.float64, device=device)

    else:
        values = _as_f64(spec, device)
        if values.shape[-1] != n_freq:
            raise ValueError(
                f"ASD array has {values.shape[-1]} bins but the target grid has "
                f"{n_freq}.  Pass a (freqs, values) pair if you want it "
                f"interpolated rather than used as is."
            )

    # ── Optional inverse-spectrum truncation, done on the PSD ─────────────────
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
    """Coerce anything array-like to a float64 tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        # Goes via torch, not NumPy — np.asarray() on a CUDA tensor raises.
        return x.detach().to(device=device, dtype=torch.float64)
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)


def _interp_log(src_f: torch.Tensor, src_v: torch.Tensor,
                dst_f: torch.Tensor) -> torch.Tensor:
    """
    Interpolate an ASD onto ``dst_f``, clamped (not extrapolated) at the edges.

    Log-linear where both bracketing samples are positive, because a detector ASD
    spans many decades and linear interpolation across them is poor.  Plain
    linear where either endpoint is zero — a zeroed band (below ``f_low``, say)
    has no logarithm, and the linear branch keeps zeros exact.  Both branches
    reproduce the source values exactly at the source nodes.
    """
    # Index the frequency axis with ``...`` so a per-detector (D, F) ASD works:
    # a bare src_v[idx] would index the detector axis instead.
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
    Inverse-spectrum-truncate a PSD, mirroring PyCBC's ``colored_noise``.

    ``inverse_spectrum_truncation`` truncates the *inverted* PSD, so to truncate
    the PSD itself we hand it ``1/PSD`` and invert the result.  Zero bins are
    replaced by the maximum beforehand (a zero would blow up the inversion) and
    restored to zero afterwards.
    """
    from sage.dsp.inverse_spectrum_truncation import inverse_spectrum_truncation_single

    fil_len = int(round(filter_duration * sample_rate))
    zero_bins = psd <= 0
    safe = torch.where(zero_bins, psd.max(), psd).to(dtype=torch.float64)

    def _one(row):
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
    else:  # per-detector PSDs — truncate each independently
        out = torch.stack([_one(row) for row in safe.reshape(-1, safe.shape[-1])])
        out = out.reshape(safe.shape)

    return torch.where(zero_bins, torch.zeros_like(out), out)


# ══════════════════════════════════════════════════════════════════════════════
# Tensor generators  (Torch — CPU or GPU, used by the batch sampler)
# ══════════════════════════════════════════════════════════════════════════════


_RNG_POOLS = {}


def _rng_chunk_count(lead: int) -> int:
    """
    How many independent RNG chunks a batch is split into.

    Deliberately a function of the batch shape **only** — never of the thread
    count or the machine — so that ``threads`` changes how fast a batch is drawn
    and nothing else.  The same seed gives the same noise whether you run it
    serially or across 32 workers, here or on another box.
    """
    if lead < 8:
        return 1
    return min(32, lead)


def _chunked_randn(shape, *, generator, device, dtype, threads):
    """
    Standard-normal fill, optionally spread across a thread pool.

    torch's CPU normal generator is single-threaded — measured at 86 ms for a
    256x2x30720 draw whether given 1 core or 96 — and it releases the GIL, so
    splitting the batch across worker threads recovers almost all of it (11.9 ms
    at 8 workers, 3.7 ms at 32).  Chunk boundaries and per-chunk seeds are fixed
    by shape, so this is purely a speed knob.

    CUDA is left alone: its generator is already massively parallel and chunking
    would only add launch overhead.
    """
    shape = tuple(shape)
    on_cpu = device is None or torch.device(device).type == "cpu"
    lead = shape[0] if len(shape) > 1 else 0
    n_chunks = _rng_chunk_count(lead) if on_cpu else 1

    if n_chunks <= 1:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    out = torch.empty(shape, device=device, dtype=dtype)
    edges = [lead * i // n_chunks for i in range(n_chunks + 1)]

    # Every chunk gets its own generator, seeded here, serially, before any
    # thread starts.  Falling back to the global RNG inside the workers would
    # be correct but pointless: it is lock-guarded, so the pool would serialise
    # on it and the whole exercise would buy nothing.
    if generator is not None:
        base = generator.initial_seed()
        seeds = [_derive_seed(base, i) for i in range(n_chunks)]
    else:  # unseeded — fresh entropy per chunk, still different every call
        seeds = list(np.random.SeedSequence().generate_state(n_chunks,
                                                             dtype=np.uint64))

    jobs = [(seeds[i], edges[i], edges[i + 1]) for i in range(n_chunks)
            if edges[i + 1] > edges[i]]

    def _fill(job):
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
    """Return a seeded ``torch.Generator`` on ``device``, or ``None`` if seedless."""
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
        Samples per second — sets the variance under the unit-PSD convention.
    seed : int or None
        Convenience seed; ignored when ``generator`` is supplied.
    generator : torch.Generator or None
        Pre-seeded generator (must live on ``device``).
    unit_psd : bool
        ``True`` → ``sigma = sqrt(fs/2)``, one-sided PSD ≡ 1.
        ``False`` → ``sigma = 1``.
    device, dtype
        Output placement.

    Returns
    -------
    torch.Tensor, shape ``shape``
    """
    gen = generator if generator is not None else _make_generator(seed, device)
    noise = _chunked_randn(shape, generator=gen, device=device, dtype=dtype,
                           threads=threads)
    sigma = _td_sigma(sample_rate, unit_psd)
    # In place — at batch scale this tensor is the largest allocation in the
    # call, and an out-of-place scale would double it for no reason.
    return noise if sigma == 1.0 else noise.mul_(sigma)


def white_noise_fd(shape, sample_rate: float, *, seed=None, generator=None,
                   unit_psd: bool = True, device=None, threads: int = 1,
                   dtype=torch.float32) -> torch.Tensor:
    """
    White Gaussian noise drawn **directly** in the frequency domain.

    Equivalent in distribution to ``rfft(white_noise_td(...), norm="forward")``
    but skips the transform, drawing ``N/2 + 1`` complex bins instead of ``N``
    real samples.  For white noise this is exact.  (It would not be for coloured
    noise: independent draws across a *shaped* spectrum give a circulant time
    series, whose autocovariance is the wrapped version of the true one.  That
    is why :func:`coloured_noise_td` pads before it colours.)

    DC and Nyquist are drawn **real**, with variance ``S·Δf/2`` matching every
    other bin.  LAL and ``pycbc.noise.gaussian`` draw them complex and then lose
    the imaginary part in the inverse real FFT, leaving those two bins at half
    power; this does not.

    Parameters
    ----------
    shape : tuple of int
        Output shape *in the time domain*; the last axis is the number of time
        samples ``N``.  The returned tensor has ``N // 2 + 1`` on that axis.
    sample_rate : float
        Samples per second.
    seed : int or None
        Convenience seed; ignored when ``generator`` is supplied.
    generator : torch.Generator or None
        Pre-seeded generator (must live on ``device``).
    unit_psd : bool
        Convention flag, as in :func:`white_noise_td`.
    device, dtype
        ``dtype`` is the *real* dtype; the output is the matching complex type.

    Returns
    -------
    torch.Tensor, complex, shape ``shape[:-1] + (N // 2 + 1,)``
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

    # DC is real; so is Nyquist, but only when N is even (otherwise there is no
    # Nyquist bin).  Both carry the same power as the interior bins.
    edge_sigma = sigma_t / math.sqrt(n_time)
    dc = torch.randn(lead, generator=gen, device=device, dtype=dtype) * edge_sigma
    noise[..., 0] = torch.complex(dc, torch.zeros_like(dc))
    if n_time % 2 == 0:
        nyq = torch.randn(lead, generator=gen, device=device, dtype=dtype) * edge_sigma
        noise[..., -1] = torch.complex(nyq, torch.zeros_like(nyq))

    return noise


def colour_fd(noise_fd: torch.Tensor, asd: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Colour unit-PSD frequency-domain noise by an ASD.

    Under the conventions at the top of this module this really is just a
    multiply — that is the entire point of generating at unit PSD.

    Parameters
    ----------
    noise_fd : torch.Tensor, complex, shape ``(..., F)``
        Unit-PSD white noise in the ``rfft(norm="forward")`` convention.
    asd : torch.Tensor or None, shape ``(F,)`` or ``(D, F)``
        Amplitude spectral density.  ``None`` is a no-op.

    Returns
    -------
    torch.Tensor, complex, same shape as ``noise_fd``.
    """
    if asd is None:
        return noise_fd
    # Skip the cast when it would be a no-op — ``.to`` is cheap but not free,
    # and the ASD is usually already resolved onto the right device and dtype.
    if asd.device != noise_fd.device or asd.dtype != noise_fd.real.dtype:
        asd = asd.to(device=noise_fd.device, dtype=noise_fd.real.dtype)
    return noise_fd * asd


def coloured_noise_td(asd, shape, sample_rate: float, *, seed=None,
                      generator=None, low_frequency_cutoff=None, is_psd=False,
                      filter_duration=None, pad_seconds=None, device=None,
                      threads: int = 1, dtype=torch.float32) -> torch.Tensor:
    """
    Coloured Gaussian noise in the time domain, generated faithfully.

    White noise is drawn over the requested span padded by ``filter_duration``
    at each end, coloured in the frequency domain, and the padding is cropped —
    PyCBC's ``colored_noise`` recipe.  The padding is what removes the
    circular-convolution transient a bare FFT colouring leaves at the edges.

    Parameters
    ----------
    asd : see :func:`resolve_asd`
    shape : tuple of int
        Output shape; the last axis is the number of time samples.
    sample_rate : float
        Samples per second.
    seed, generator, device, dtype
        As in :func:`white_noise_td`.
    low_frequency_cutoff, is_psd
        Forwarded to :func:`resolve_asd`.
    filter_duration : float or None
        Inverse-spectrum-truncation length in seconds, forwarded to
        :func:`resolve_asd`.  Also sets the edge padding unless ``pad_seconds``
        overrides it.  ``None`` skips the truncation.
    pad_seconds : float or None
        Edge padding in seconds, decoupled from ``filter_duration`` so that a
        pre-resolved ASD can be reused without re-truncating it.  Defaults to
        ``filter_duration``, or a quarter of the requested duration.

    Returns
    -------
    torch.Tensor, shape ``shape``
    """
    shape = tuple(shape)
    n_time = shape[-1]

    if asd is None:
        return white_noise_td(shape, sample_rate, seed=seed, generator=generator,
                              unit_psd=True, device=device, threads=threads,
                              dtype=dtype)

    if pad_seconds is None:
        pad_seconds = (filter_duration if filter_duration is not None
                       else (n_time / sample_rate) / 4.0)
    n_pad = int(round(pad_seconds * sample_rate))
    # Round the transform length up to a 5-smooth size; the extra lands in the
    # cropped-away tail and buys back several-fold on a bad factorisation.
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


# ══════════════════════════════════════════════════════════════════════════════
# Legacy per-sample generator
# ══════════════════════════════════════════════════════════════════════════════


class WhiteNoiseGenerator:
    """
    Per-sample Gaussian noise generator for the legacy CPU DataLoader path.

    Each detector gets its own independent stream addressed by
    ``(sample_seed, detector_index)``, so results are reproducible regardless of
    how many DataLoader workers are running and in what order they execute —
    the previous implementation called the global ``numpy.random.seed``, which
    is neither thread-safe nor addressable.

    Parameters
    ----------
    asd : see :func:`resolve_asd`
        Optional colouring.  ``None`` (default) gives white noise.
    unit_psd : bool
        ``True`` (default) → one-sided PSD ≡ 1.  ``False`` → unit variance.
        Ignored when ``asd`` is set (the output is then in strain units).
    is_psd : bool
        Set when ``asd`` is really a PSD.
    low_frequency_cutoff : float or None
        Zero the ASD below this frequency.
    filter_duration : float or None
        Colouring-filter length in seconds.
    """

    def __init__(self, asd=None, *, unit_psd=True, is_psd=False,
                 low_frequency_cutoff=None, filter_duration=None):
        self.asd = asd
        self.unit_psd = unit_psd
        self.is_psd = is_psd
        self.low_frequency_cutoff = low_frequency_cutoff
        self.filter_duration = filter_duration
        # Resolving an ASD means interpolation and possibly an inverse-spectrum
        # truncation — far too expensive to redo for every sample.
        self._asd_cache = {}

    def _resolved_asd(self, n_full, sample_rate):
        """Resolve (and cache) the ASD on an ``n_full``-sample grid."""
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

    def generate(self, sample_length_in_num, seed=0, sample_rate=2048.0):
        """
        Draw a single noise realisation.

        Parameters
        ----------
        sample_length_in_num : int
            Number of samples to generate.
        seed : int
            Stream seed for reproducibility.
        sample_rate : float
            Samples per second; sets the unit-PSD variance.

        Returns
        -------
        numpy.ndarray, shape ``(sample_length_in_num,)``
        """
        if self.asd is None:
            rng = _spawn_rng(seed)
            sigma = _td_sigma(sample_rate, self.unit_psd)
            return rng.normal(scale=sigma, size=sample_length_in_num)

        pad_seconds = (self.filter_duration if self.filter_duration is not None
                       else (sample_length_in_num / sample_rate) / 4.0)
        n_pad = int(round(pad_seconds * sample_rate))
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
        Generate noise for every detector of a single sample.

        Parameters
        ----------
        special : dict
            Must contain ``"sample_seed"`` (int) and ``"data_cfg"`` with
            ``signal_length`` (s) and ``sample_rate`` (Hz) attributes.
        det_only : str
            Unused; kept for API compatibility.

        Returns
        -------
        numpy.ndarray, shape ``(D, N)``
            Stacked per-detector noise.  ``D`` is 2 (H1, L1) unless the
            registered config says otherwise.
        """
        data_cfg = special["data_cfg"]
        sample_rate = data_cfg.sample_rate
        n_samples = int(data_cfg.signal_length * sample_rate)

        try:
            n_detectors = len(get_cfg().detectors)
        except Exception:
            n_detectors = 2

        seed = special["sample_seed"]
        return np.stack(
            [
                self.generate(n_samples, seed=_derive_seed(seed, d),
                              sample_rate=sample_rate)
                for d in range(n_detectors)
            ],
            axis=0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Batch sampler
# ══════════════════════════════════════════════════════════════════════════════


class WhiteGaussianNoiseSampler(torch.nn.Module):
    """
    Batch Gaussian noise sampler — white or coloured, frequency or time domain.

    Generates noise natively on ``cfg.device`` (no NumPy round trip, no host-to-
    device copy) and returns a batch that mirrors the
    :class:`~sage.data.noise.real_noise.MemmapNoiseSampler` API, so it can be
    dropped in wherever real noise is used.

    Reproducibility is **index-addressable**: the batch produced at step *n*
    depends only on ``(seed, n)``, not on how many batches were drawn before it.
    Resuming a run mid-stream therefore reproduces the same noise, which a plain
    sequential generator cannot do.  Call :meth:`set_step` to jump anywhere.

    White noise in the default ``domain="fd"`` mode is drawn straight in the
    frequency domain — ``N/2 + 1`` complex bins rather than ``N`` real samples
    plus a transform.  Coloured noise is sampled white on a grid padded by
    ``filter_duration`` at each end, coloured, then cropped back, so the segment
    edges carry no circular-convolution transient.

    Parameters
    ----------
    seed : int or None
        Root seed.  ``None`` means unseeded (fresh OS entropy each batch).
    asd : see :func:`resolve_asd`
        Optional colouring — a ``(freqs, values)`` pair, a callable, a scalar, an
        array on the padded grid, or an analytic model name such as
        ``"aLIGOZeroDetHighPower"``.  ``None`` (default) gives white noise.
    domain : {"fd", "td"}
        ``"fd"`` returns ``(B, D, F)`` complex in the ``rfft(norm="forward")``
        convention (the default, matching the real-noise samplers).
        ``"td"`` returns ``(B, D, N)`` real.
    unit_psd : bool
        ``True`` (default) → one-sided PSD ≡ 1, so colouring by an ASD lands
        directly in strain units.  ``False`` → unit time-domain variance.
        Ignored when ``asd`` is set.
    is_psd : bool
        Set when ``asd`` is really a PSD (``1/Hz``) rather than an ASD.
    low_frequency_cutoff : float or None
        Zero the ASD below this frequency.  Defaults to the data config's
        ``signal_low_frequency_cutoff`` when an ASD is given.
    filter_duration : float or None
        Colouring-filter length in seconds; also the edge padding.  Defaults to
        a quarter of the segment duration.

    Attributes
    ----------
    GRAPH_READY : bool
        ``False`` — seeded RNG calls are not traceable by ``torch.compile``.

    Input / Output
    --------------
    forward() : () → ``((B, D, F) complex | (B, D, N) float, (B, 1) float)``
        Noise batch and its all-zero targets.
    """

    GRAPH_READY = False

    def __init__(self, seed=None, *, asd=None, domain="fd", unit_psd=True,
                 is_psd=False, low_frequency_cutoff=None, filter_duration=None):
        super().__init__()

        if domain not in ("fd", "td"):
            raise ValueError(f"domain must be 'fd' or 'td', got {domain!r}")

        cfg = get_cfg()
        data_cfg = get_data_cfg()

        self.seq_len = data_cfg.padded_length_in_nsamples
        self.sample_rate = float(data_cfg.sample_rate)
        self.device = cfg.device
        self.n_detectors = len(cfg.detectors)
        self.batch_size = cfg.batch_size
        self.real_dtype = torch.float32

        self.seed = seed
        self.domain = domain
        self.unit_psd = unit_psd
        self.filter_duration = filter_duration
        self._step = 0

        # Accept the same friendly vocabulary as ``sample_synthetic_noise`` —
        # "auto", "LIGO", "Virgo" and so on.  Unlike that function this keeps
        # ``asd=None`` meaning white noise: the sampler's job is to stand in for
        # the real-noise samplers, and silently colouring would change what a
        # default-constructed sampler feeds a training loop.
        if asd is not None:
            asd = _expand_asd_spec(asd, cfg.detectors)

        if asd is not None and low_frequency_cutoff is None:
            low_frequency_cutoff = getattr(
                data_cfg, "signal_low_frequency_cutoff", None
            )

        # Colouring happens on a padded grid (see the class docstring), so the
        # ASD is resolved there — once, here, rather than on every batch:
        # interpolation and inverse-spectrum truncation are far too expensive to
        # repeat per step, and resolving eagerly surfaces a bad spec at
        # construction time rather than mid-run.
        self._pad_nsamples = 0
        self._n_full = self.seq_len
        resolved = None
        if asd is not None:
            pad_seconds = (filter_duration if filter_duration is not None
                           else (self.seq_len / self.sample_rate) / 4.0)
            self._pad_nsamples = int(round(pad_seconds * self.sample_rate))
            n_full = _fast_fft_len(self.seq_len + 2 * self._pad_nsamples)
            self._n_full = n_full
            freqs = torch.fft.rfftfreq(
                n_full, d=1.0 / self.sample_rate, device=self.device,
                dtype=torch.float64,
            )
            try:
                resolved = _resolve_asd_cached(
                    asd, freqs=freqs, sample_rate=self.sample_rate,
                    is_psd=is_psd, low_frequency_cutoff=low_frequency_cutoff,
                    filter_duration=filter_duration,
                    n_detectors=self.n_detectors,
                    device=self.device, dtype=self.real_dtype,
                )
            except ValueError as exc:
                raise ValueError(
                    f"colouring runs on a padded grid of {n_full} samples, but "
                    f"this ASD could not be resolved there ({exc}).  Pass it as a "
                    f"(freqs, values) pair or an analytic model name so it can be "
                    f"evaluated on any grid."
                ) from exc

        self._has_asd = resolved is not None
        # Registered so it follows the module across devices and into state dicts.
        self.register_buffer(
            "asd", resolved if resolved is not None
            else torch.empty(0, device=self.device)
        )

        if self._has_asd and unit_psd is False:
            warnings.warn(
                "unit_psd=False is ignored when an ASD is supplied — coloured "
                "noise is returned in the ASD's own units.",
                stacklevel=2,
            )

        self.noise_target = torch.zeros(
            (self.batch_size, 1), dtype=cfg.dtype, device=cfg.device
        )

    # ── Stream addressing ─────────────────────────────────────────────────────

    def set_step(self, step: int):
        """Jump the sampler to ``step``; the next batch is the one for that index."""
        self._step = int(step)

    @property
    def step(self) -> int:
        """Index of the next batch to be drawn."""
        return self._step

    def _generator(self):
        """Generator for the current step, or ``None`` when unseeded."""
        if self.seed is None:
            return None
        return _make_generator(_derive_seed(self.seed, self._step), self.device)

    # ── Generation ────────────────────────────────────────────────────────────

    def _sample_batch(self):
        """Draw one batch in the configured domain."""
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

        # Sample white on a padded grid, colour it, crop the transients off.
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
        Return the next noise batch and its zero targets, advancing the step.

        Returns
        -------
        noise : torch.Tensor
            ``(B, D, F)`` complex for ``domain="fd"``, ``(B, D, N)`` float for
            ``domain="td"``.
        noise_target : torch.Tensor, shape ``(B, 1)`` float — all zeros.
        """
        batch = self._sample_batch()
        self._step += 1
        return batch, self.noise_target

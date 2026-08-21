#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : decode.py
Description   : Decode the network's point-estimate head into physical quantities.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The head emits a blocked layout ``[mu_0..mu_K, sraw_0..sraw_K]`` for the targets in
``cfg.do_point_estimate`` (production: tc, mchirp). These are point estimates with
heteroscedastic uncertainties, not parameter estimation; masses, spins, distance and
sky location are unavailable from the network.

Those two heads are the entire inventory. Nothing in this module produces a component
mass, a spin, a distance or a sky position, and nothing it returns implies one: a chirp
mass does not separate m1 from m2, and the search reports mchirp as the head's own
estimate carrying its own sigma, never as a mass measurement.

Decoding runs inside the inference loop, which is what fixes the trigger schema: a shard
stores ``tc_gps``, ``tc_sigma``, ``mchirp`` and ``mchirp_sigma``
(:data:`sage.search.triggers.TRIGGER_COLUMNS`) and not the raw head output. Storing the
raw output would save two columns and defer the decision, but the constants that invert
the training-time encoding live in the parameter sampler built from the training prior,
so every later reader would have to rebuild that sampler from the same YAML -- and a
shard decoded a year later against a drifted prior would yield different physical
numbers while still looking valid. Decoded once, at the only point where the training
prior is unambiguously in hand, what is written is final.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from sage.search.geometry import SearchGeometry
from sage.search.triggers import TRIGGER_COLUMNS

# The production head, i.e. cfg.do_point_estimate. The ordering is part of the contract:
# it is the order of the blocked layout, and swapping the two columns is arithmetically
# silent -- tc's mean and std applied to mchirp still produce plausible numbers.
PE_TARGETS: Tuple[str, ...] = ("tc", "mchirp")

# The trigger columns each target fills, as (value, sigma). tc is the exception whose
# value column is not the decoded value: the shard stores an absolute GPS time.
PE_TRIGGER_COLUMNS: Dict[str, Tuple[str, str]] = {
    "tc": ("tc_gps", "tc_sigma"),
    "mchirp": ("mchirp", "mchirp_sigma"),
}

# The softplus bounds BCEWithPEsigmaLoss applied during training. Decoding has to use
# the same two numbers: the inverse of a clamp is the identity only inside its range.
SIGMA_MIN: float = 1e-3
SIGMA_MAX: float = 10.0


def _to_numpy(values, name: str) -> np.ndarray:
    """
    Materialise an array argument as numpy, without importing torch.

    The head output arrives as a torch tensor, usually resident on the GPU, where
    ``np.asarray`` raises rather than transferring. Duck-typing ``detach`` and ``cpu``
    handles that while keeping this module importable with numpy alone, which
    :mod:`sage.search` requires of everything the orchestrator touches.
    """
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    try:
        return np.asarray(values)
    except Exception as exc:  # noqa: BLE001 - re-raised with the argument named
        raise TypeError(f"{name} is not array-like: {type(values).__name__}") from exc


def _softplus(raw) -> np.ndarray:
    """
    ``log(1 + exp(raw))``, evaluated stably.

    ``np.logaddexp(0, raw)`` rather than ``np.log1p(np.exp(raw))``: the raw sigma output
    is an unbounded linear layer, and the direct form overflows to ``inf`` above about
    709 and loses precision long before that. torch's ``softplus`` switches to its
    linear branch above 20, where the two forms differ by 2e-9 -- far inside the region
    where :data:`SIGMA_MAX` has already clamped the result.
    """
    return np.logaddexp(0.0, np.asarray(raw, dtype=np.float64))


def _float64_gps(window_start_gps, name: str = "window_start_gps") -> np.ndarray:
    """
    Read window-start GPS times, refusing any float narrower than 64 bits.

    At 1.24e9 the float32 grid is 128 s coarse, so a float32 start does not merely lose
    precision, it relocates the trigger by up to two minutes. Upcasting silently would
    keep the error and destroy the evidence of where it came from, so a narrow float is
    refused instead. Integer GPS times are exact and are accepted.
    """
    starts = _to_numpy(window_start_gps, name)
    if starts.dtype.kind == "f" and starts.dtype.itemsize < 8:
        raise TypeError(
            f"{name} has dtype {starts.dtype}; GPS times near 1.24e9 are spaced 128 s "
            "apart in float32, so the start must be float64 (or an exact integer type)"
        )
    if starts.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be numeric, got dtype {starts.dtype}")
    return starts.astype(np.float64)


def tc_to_gps(
    geometry: SearchGeometry, window_start_gps, tc_value
) -> np.ndarray:
    """
    Absolute coalescence GPS from a raw window start and a within-window tc.

    The head predicts tc in seconds from the start of the *analysis content*, which
    begins ``padding_length_s`` after the raw window start; the training target is the
    sampled ``tc``, whose production prior is [11.0, 11.2] s inside 12 s of content. The
    conversion is written through the geometry's own quantities,

        ``tc_gps = window_start_gps + peak_offset_s + (tc - tc_mid_s)``

    so a tc at the prior midpoint lands bit-for-bit on
    :meth:`~sage.search.geometry.SearchGeometry.window_gps`, the nominal trigger time
    every other layer of the search already uses. ``peak_offset_s - tc_mid_s`` is the
    one-sided padding by the identity ``SearchGeometry.__post_init__`` asserts, so this
    is the same quantity as ``window_start_gps + padding_length_s + tc``; it is written
    this way because the offset is then accumulated at window scale and added to the GPS
    exactly once. The other ordering rounds twice near 1.24e9, where the float64 spacing
    is 2.4e-7 s.

    ``tc_value`` is a decoded tc in physical seconds -- ``decode(...).values["tc"]`` --
    not a raw head output, and the result is a coalescence time and nothing else: no
    mass, spin or sky position is involved in or implied by this conversion.

    Raises
    ------
    TypeError
        If ``window_start_gps`` is a float narrower than float64.
    ValueError
        If the two arguments are both one-dimensional with different lengths.
    """
    starts = _float64_gps(window_start_gps)
    tc = _to_numpy(tc_value, "tc_value").astype(np.float64)
    if starts.ndim == 1 and tc.ndim == 1 and starts.shape != tc.shape:
        raise ValueError(
            f"{starts.shape[0]} window starts against {tc.shape[0]} coalescence times; "
            "each trigger carries its own window start"
        )
    # One rounding at GPS magnitude: the whole offset is built at window scale first.
    offset_s = geometry.peak_offset_s + (tc - geometry.tc_mid_s)
    return starts + offset_s


@dataclass
class DecodedPE:
    """
    Physical point estimates and their standard deviations.

    One entry per target in ``cfg.do_point_estimate``, which in production is tc and
    mchirp and nothing else. There is no mass, spin, distance or sky entry here, and
    none can be recovered from these two.

    ``values["tc"]`` is in seconds from the start of the analysis content -- the frame
    the training target was drawn in -- and not an absolute GPS time; only
    :meth:`PEDecoder.tc_gps` makes it absolute. ``sigmas`` are in the same physical
    units as their values.

    ``at_prior_rail`` is a diagnostic and not a cut. It is True where the estimate sits
    at or outside the training prior, which is where the head has saturated and the
    number has stopped being a measurement.
    """

    values: Dict[str, np.ndarray]
    sigmas: Dict[str, np.ndarray]
    at_prior_rail: Dict[str, np.ndarray]

    def __post_init__(self) -> None:
        """
        Refuse a set whose three dicts do not describe the same targets and windows.

        A missing sigma or rail entry would surface only when some later stage asked for
        that column, by which point the shard it came from has been written.
        """
        keys = set(self.values)
        companions = (("sigmas", self.sigmas), ("at_prior_rail", self.at_prior_rail))
        for label, mapping in companions:
            if set(mapping) != keys:
                raise ValueError(
                    f"{label} covers {sorted(mapping)} against values' {sorted(keys)}"
                )
        lengths = {
            name: int(np.asarray(column).shape[0])
            for name, column in self.values.items()
        }
        for _, mapping in companions:
            for name, column in mapping.items():
                if int(np.asarray(column).shape[0]) != lengths[name]:
                    raise ValueError(
                        f"columns of unequal length for {name!r}: {lengths}"
                    )
        if len(set(lengths.values())) > 1:
            raise ValueError(f"targets of unequal length: {lengths}")
        for name, column in self.at_prior_rail.items():
            if np.asarray(column).dtype != bool:
                raise TypeError(
                    f"at_prior_rail[{name!r}] has dtype "
                    f"{np.asarray(column).dtype}; a numeric flag column is read as an "
                    "index by anything that masks with it"
                )

    @property
    def names(self) -> Tuple[str, ...]:
        """The decoded targets, in insertion order."""
        return tuple(self.values)

    def __len__(self) -> int:
        """Number of windows decoded."""
        for column in self.values.values():
            return int(np.asarray(column).shape[0])
        return 0

    def column(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(value, sigma)`` for one target."""
        if name not in self.values:
            raise KeyError(
                f"{name!r} was not decoded; this set holds {sorted(self.values)}. The "
                "network has two point-estimate heads, tc and mchirp, and no others"
            )
        return self.values[name], self.sigmas[name]


class PEDecoder:
    """
    Convert raw head outputs to physical units.

    The head is trained against transformed targets: standardised by default, or min-max
    normalised onto [0, 1] when ``cfg.pe_target_minmax`` is set. Both maps are affine
    and both are inverted here with the constants the sampler itself holds, so this is
    the exact inverse of the training-time encoding rather than a reconstruction of it.

    It decodes tc and mchirp. No mass, spin, distance or sky position is produced,
    inferred or implied by any method on this class.

    Parameters
    ----------
    targets : sequence of str
        ``cfg.do_point_estimate`` ordering.
    param_sampler : object
        Provides the standardisation and min-max bounds used during training.
    pe_target_minmax : bool
        Which of the two encodings the checkpoint was trained under, i.e.
        ``cfg.pe_target_minmax``. There is no way to detect it from the head output: the
        two maps differ only in scale and offset, so the wrong choice returns
        well-formed, wrong physical values.
    geometry : SearchGeometry, optional
        Needed only by :meth:`tc_gps` and :meth:`trigger_columns`. When supplied, the tc
        prior it declares is checked against the sampler's, because a mismatch shifts
        every reported merger time by half the difference and no downstream stage can
        see it.
    sigma_min, sigma_max : float
        The softplus bounds the training loss applied. The defaults are the production
        ones; a checkpoint trained under others must say so, since the inverse of a
        clamp is the identity only inside its range.
    """

    def __init__(
        self,
        targets: Sequence[str],
        param_sampler,
        pe_target_minmax: bool = False,
        geometry: Optional[SearchGeometry] = None,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
    ) -> None:
        self.targets = tuple(str(name) for name in targets)
        if not self.targets:
            raise ValueError("a decoder needs at least one point-estimate target")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(f"targets repeated in {self.targets}")
        self.param_sampler = param_sampler
        self.pe_target_minmax = bool(pe_target_minmax)
        self.geometry = geometry
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if not 0.0 < self.sigma_min <= self.sigma_max:
            raise ValueError(
                f"sigma bounds must satisfy 0 < sigma_min <= sigma_max, got "
                f"[{self.sigma_min}, {self.sigma_max}]"
            )

        if self.pe_target_minmax:
            offsets = self._buffer("_norm_mins")
            scales = self._buffer("_norm_scales")
            positions = self._positions("_norm_indices")
        else:
            offsets = self._buffer("_std_means")
            scales = self._buffer("_std_stds")
            positions = self._positions("_std_indices")
        if offsets.size != scales.size:
            raise ValueError(
                f"the sampler's offset and scale buffers hold {offsets.size} and "
                f"{scales.size} entries; they describe the same targets"
            )
        if int(positions.max(initial=-1)) >= offsets.size:
            raise ValueError(
                f"targets {self.targets} need buffer position "
                f"{int(positions.max(initial=-1))}, but the sampler encoded only "
                f"{offsets.size} parameters"
            )
        self._offsets = offsets[positions]
        self._scales = scales[positions]
        if not np.isfinite(self._offsets).all() or not np.isfinite(self._scales).all():
            raise ValueError(
                f"the sampler's encoding constants for {self.targets} are not finite: "
                f"offsets {self._offsets}, scales {self._scales}"
            )
        if (self._scales <= 0.0).any():
            raise ValueError(
                f"the sampler's scales for {self.targets} are {self._scales}; a "
                "non-positive scale collapses every window onto the offset, which "
                "decodes to a constant that still looks like an estimate"
            )

        bounds = [self._prior_bounds(name) for name in self.targets]
        self._prior_lo = np.asarray([low for low, _ in bounds], dtype=np.float64)
        self._prior_hi = np.asarray([high for _, high in bounds], dtype=np.float64)
        finite_bounds = np.isfinite(self._prior_lo).all() and np.isfinite(
            self._prior_hi
        ).all()
        if not finite_bounds:
            raise ValueError(
                f"the sampler declares non-finite prior bounds for {self.targets}"
            )
        if (self._prior_lo >= self._prior_hi).any():
            raise ValueError(
                f"prior bounds for {self.targets} are not increasing: "
                f"{list(zip(self._prior_lo, self._prior_hi))}"
            )
        if self.pe_target_minmax:
            self._check_minmax_matches_prior()
        if geometry is not None and "tc" in self.targets:
            self._check_tc_prior(geometry)

    # ------------------------------------------------------------------ construction
    def _buffer(self, attr: str) -> np.ndarray:
        """
        Read one of the sampler's registered encoding buffers as float64.

        Missing buffers are a hard error rather than a fallback. They are registered by
        ``DistributionSampler._compile_batch_standardiser`` and
        ``_compile_batch_normaliser``, which the signal sampler calls at construction;
        decoding without them would need constants this module would have to invent, and
        an invented mean and std produce physical-looking numbers that are wrong by
        whatever the prior happened to be.
        """
        values = getattr(self.param_sampler, attr, None)
        if values is None:
            raise AttributeError(
                f"the parameter sampler exposes no {attr!r}; it is registered by "
                "DistributionSampler._compile_batch_standardiser / "
                "_compile_batch_normaliser, and without it the training encoding "
                "cannot be inverted"
            )
        return _to_numpy(values, attr).astype(np.float64).ravel()

    def _positions(self, attr: str) -> np.ndarray:
        """
        Locate each target inside the sampler's buffers by name, not by position.

        The buffers are built in ``cfg.do_point_estimate`` order, which is usually the
        order the caller passes -- but "usually" is not a contract, and the failure when
        it does not hold is that tc's mean and std are applied to mchirp and vice versa,
        which is silent. Resolving through ``param_index`` makes that impossible
        wherever the sampler exposes it. A sampler that exposes neither the index buffer
        nor ``param_index`` is read positionally, with the buffer length checked, since
        there is then nothing left to resolve against.
        """
        indices = getattr(self.param_sampler, attr, None)
        param_index = getattr(self.param_sampler, "param_index", None)
        if indices is None or param_index is None:
            return np.arange(len(self.targets), dtype=np.int64)
        columns = [int(value) for value in _to_numpy(indices, attr).ravel().tolist()]
        positions = []
        for name in self.targets:
            if name not in param_index:
                raise KeyError(
                    f"the parameter sampler has no parameter named {name!r}; the prior "
                    "it was built from does not contain this target"
                )
            column = int(param_index[name])
            if column not in columns:
                raise KeyError(
                    f"{name!r} is not among the parameters the sampler encoded; the "
                    "checkpoint's do_point_estimate and the sampler's disagree"
                )
            positions.append(columns.index(column))
        return np.asarray(positions, dtype=np.int64)

    def _prior_bounds(self, name: str) -> Tuple[float, float]:
        """
        The training prior's bounds for one target, in physical units.

        Absent bounds are refused rather than defaulted. Returning an infinite interval
        would report every window as un-saturated, which is a claim about the head, not
        an absence of one.
        """
        normalisers = getattr(self.param_sampler, "normalisers", None) or {}
        if name in normalisers:
            entry = normalisers[name]
            return float(entry.min_val), float(entry.max_val)
        bounds = getattr(self.param_sampler, "bounds", None) or {}
        if name in bounds:
            low, high = bounds[name]
            return float(low), float(high)
        raise KeyError(
            f"the parameter sampler declares no prior bounds for {name!r}, so a "
            "decoded value cannot be distinguished from one that saturated the prior"
        )

    def _check_minmax_matches_prior(self) -> None:
        """
        Assert the min-max buffers are the prior bounds they are supposed to be.

        Under ``pe_target_minmax`` the affine map *is* the prior: offset is the lower
        bound and offset plus scale is the upper. If they disagree, the buffers and the
        normalisers were built from different priors, and every decoded value is wrong
        by that difference while remaining inside plausible ranges.

        Compared at 1e-6 relative because the buffers are stored in the training dtype,
        which is float32: 11.2 - 11.0 evaluates to 0.19999981 there against
        0.20000000000000018 in float64. A genuine prior mismatch is of order the prior
        width itself, five orders larger.
        """
        upper = self._offsets + self._scales
        for name, low, high, offset, top in zip(
            self.targets, self._prior_lo, self._prior_hi, self._offsets, upper
        ):
            if not (
                math.isclose(float(offset), float(low), rel_tol=1e-6, abs_tol=1e-9)
                and math.isclose(float(top), float(high), rel_tol=1e-6, abs_tol=1e-9)
            ):
                raise ValueError(
                    f"the min-max buffers for {name!r} map onto [{offset}, {top}] "
                    f"while its prior is [{low}, {high}]; the buffers and the prior "
                    "bounds were built from different configurations"
                )

    def _check_tc_prior(self, geometry: SearchGeometry) -> None:
        """
        Assert the geometry's tc prior is the one the network was trained on.

        The geometry supplies ``tc_mid_s``, which sets where in the window a trigger is
        placed; the sampler supplies the prior the head's tc was drawn from. If the two
        differ, every reported merger time is displaced by half the difference,
        uniformly and undetectably -- the triggers stay self-consistent, cluster
        normally, and land in the wrong place. Compared at 1e-6 s, a fifth of a sample
        at 2048 Hz, while a real mismatch is of order the 0.2 s prior width.
        """
        low, high = self._prior_bounds("tc")
        if not (
            math.isclose(low, geometry.tc_lower_s, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(high, geometry.tc_upper_s, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError(
                f"the geometry's tc prior [{geometry.tc_lower_s}, "
                f"{geometry.tc_upper_s}] differs from the one the network was trained "
                f"on [{low}, {high}]; every decoded merger time would be displaced by "
                "half the difference"
            )

    def _require_geometry(self) -> SearchGeometry:
        """The geometry, or a refusal naming what it is needed for."""
        if self.geometry is None:
            raise ValueError(
                "this decoder was built without a SearchGeometry, so a within-window "
                "tc cannot be placed on the GPS axis; pass geometry= at construction"
            )
        return self.geometry

    # ------------------------------------------------------------------- decoding
    def __len__(self) -> int:
        """Number of point-estimate targets."""
        return len(self.targets)

    @property
    def n_columns(self) -> int:
        """Width of the head block this decoder reads, ``2 * len(self)``."""
        return 2 * len(self.targets)

    def split(self, point_estimates) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Split the blocked layout into means and raw sigmas.

        The head concatenates ``[mu_0..mu_K, sraw_0..sraw_K]`` -- blocked, not
        interleaved -- so column ``P + j`` carries target ``j``'s raw sigma. Both halves
        are returned as float64 ``(B, P)``: the mean is added to a GPS time downstream,
        where float32 has a 128 s grid, and the promotion has to happen before that
        arithmetic rather than after it.

        A mean-only head, of width ``P``, is refused rather than padded with NaN sigmas
        the way ``sage.factory.testing`` does for diagnostics. ``tc_sigma`` and
        ``mchirp_sigma`` are first-class shard columns, and a shard full of NaN there is
        indistinguishable downstream from a heteroscedastic head that failed.
        """
        values = _to_numpy(point_estimates, "point_estimates").astype(np.float64)
        if values.ndim != 2:
            raise ValueError(
                f"point_estimates must be 2-D (windows, {self.n_columns}), got shape "
                f"{tuple(values.shape)}; a 1-D array is ambiguous between one window "
                "and one column, and reading it as the wrong one pairs means with "
                "sigmas"
            )
        width = values.shape[1]
        if width != self.n_columns:
            if width == len(self.targets):
                raise ValueError(
                    f"point_estimates is {width} columns wide, which is a mean-only "
                    f"head for targets {self.targets}. The shard schema has a sigma "
                    "column per target, and filling them with NaN would be reported as "
                    "an uncertainty rather than as a missing head"
                )
            raise ValueError(
                f"point_estimates is {width} columns wide against the "
                f"{self.n_columns} expected for targets {self.targets}"
            )
        n = len(self.targets)
        return values[:, :n], values[:, n:]

    def sigma(self, raw_sigma) -> "np.ndarray":
        """
        Map raw sigma outputs to positive standard deviations.

        ``clamp(softplus(raw) + sigma_min, sigma_min, sigma_max)``, which is exactly
        ``BCEWithPEsigmaLoss._sigma``. That is the value that entered the training
        likelihood, so decoding with anything else reports an uncertainty the network
        was never scored against. ``sage.factory.testing`` applies the floor without the
        cap; the two agree until ``softplus(raw)`` reaches ``sigma_max``, above which
        the uncapped form keeps growing through a region training cannot distinguish.

        The result is still in the target's encoded units -- standardised, or [0, 1]
        min-max. :meth:`decode` multiplies by the same scale it applies to the mean,
        since both are affine maps and a scale is all an affine map does to a width.
        """
        return np.clip(
            _softplus(raw_sigma) + self.sigma_min, self.sigma_min, self.sigma_max
        )

    def decode(self, point_estimates) -> DecodedPE:
        """
        Un-standardise to physical values and flag prior-rail saturation.

        Returns tc in seconds from the start of the analysis content, the frame the
        training target was drawn in, and not an absolute GPS time: :meth:`tc_gps` does
        that, and it needs a window start this method is not given. It returns mchirp
        and its sigma, and no mass, spin, distance or sky position -- none of those is
        predicted by the network or derivable from what is here.

        ``at_prior_rail`` marks values at or outside the training prior. The head is a
        linear layer with no bound of its own, so it emits values the prior never
        contained; those are the head saturating rather than measurements. They are
        flagged rather than clipped or dropped: clipping would build a spurious pile-up
        exactly on the bound, which then looks like structure, and dropping would
        discard a trigger the ranking statistic may still have liked.

        A non-finite head output is refused. NaN would flow into ``tc_gps`` and from
        there into clustering, where it compares false against every other trigger and
        quietly forms a cluster of its own; an infinite mean decodes to an infinite
        merger time. Either means the network produced nothing usable for that window,
        which is a fault to report rather than a value to write.
        """
        mu, raw = self.split(point_estimates)
        n_bad = int((~np.isfinite(mu)).sum() + (~np.isfinite(raw)).sum())
        if n_bad:
            raise ValueError(
                f"{n_bad} of {mu.size + raw.size} point-estimate outputs are not "
                "finite; the network produced nothing usable for those windows, and a "
                "NaN coalescence time disappears from every comparison it takes part in"
            )
        values = mu * self._scales + self._offsets
        sigmas = self.sigma(raw) * self._scales
        rail = (values <= self._prior_lo) | (values >= self._prior_hi)
        return DecodedPE(
            values={
                name: np.ascontiguousarray(values[:, j])
                for j, name in enumerate(self.targets)
            },
            sigmas={
                name: np.ascontiguousarray(sigmas[:, j])
                for j, name in enumerate(self.targets)
            },
            at_prior_rail={
                name: np.ascontiguousarray(rail[:, j])
                for j, name in enumerate(self.targets)
            },
        )

    def tc_gps(self, window_start_gps: np.ndarray, tc_value: np.ndarray) -> np.ndarray:
        """
        Absolute coalescence time from a window start and the tc estimate.

        Delegates to :func:`tc_to_gps`, which owns the convention, using the geometry
        given at construction. ``tc_value`` is a decoded tc in physical within-window
        seconds -- ``decode(...).values["tc"]`` -- not a raw head output.
        """
        return tc_to_gps(self._require_geometry(), window_start_gps, tc_value)

    def trigger_columns(
        self, point_estimates, window_start_gps
    ) -> Dict[str, np.ndarray]:
        """
        The point-estimate columns of a trigger shard, ready to write.

        Returns ``tc_gps``, ``tc_sigma``, ``mchirp`` and ``mchirp_sigma``: the four
        names in :data:`sage.search.triggers.TRIGGER_COLUMNS` that this layer owns, and
        no others. In particular it returns no masses and no spins. ``mchirp`` is the
        head's own chirp-mass estimate with its own sigma; a chirp mass does not
        determine m1 and m2, and nothing here attempts to.

        tc is the one target whose value column is not the decoded value: the shard
        stores an absolute GPS time, because a trigger is later compared against other
        detectors, other time slides and a catalogue, none of which knows which window
        it came from.

        ``window_start_gps`` must carry one raw window start per window, in float64. A
        scalar is refused for a batch: it would broadcast, and every trigger in the
        block would be dated from the first window's start.
        """
        decoded = self.decode(point_estimates)
        n_windows = len(decoded)
        starts = _float64_gps(window_start_gps)
        if starts.shape != (n_windows,):
            raise ValueError(
                f"window_start_gps has shape {tuple(starts.shape)} against "
                f"{n_windows} windows; each trigger is dated from its own window start"
            )
        columns: Dict[str, np.ndarray] = {}
        for name in self.targets:
            if name not in PE_TRIGGER_COLUMNS:
                raise KeyError(
                    f"target {name!r} has no column in the trigger schema; add it to "
                    "PE_TRIGGER_COLUMNS and to TRIGGER_COLUMNS together, or the "
                    "column is written and silently dropped by the next stage"
                )
            value_column, sigma_column = PE_TRIGGER_COLUMNS[name]
            value, sigma = decoded.column(name)
            if name == "tc":
                value = self.tc_gps(starts, value)
            columns[value_column] = value
            columns[sigma_column] = sigma
        unknown = [name for name in columns if name not in TRIGGER_COLUMNS]
        if unknown:
            raise KeyError(
                f"columns {sorted(unknown)} are not in the shard schema "
                f"{TRIGGER_COLUMNS}"
            )
        return columns

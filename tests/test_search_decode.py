#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_decode.py
Description   : Head decoding: the training inverse, tc convention, shard schema.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Three things are worth pinning here, and all three fail silently if they are wrong.

The affine inverse. The head is trained against standardised targets, so a decoded value
is a mean and a std away from the raw output. A wrong constant, or the right constants
applied to the wrong target, produces physical-looking numbers in the right range: an
mchirp of 20 solar masses and a tc inside the window are exactly what a correct decoder
also produces. So the arithmetic is checked against literal hand-computed values, and
the target-to-buffer mapping is checked with a sampler whose buffer order deliberately
differs from its parameter order.

The sigma mapping is checked against the training loss itself, ``BCEWithPEsigmaLoss``
imported and called, rather than against a reimplementation of it here, because the only
property that matters is that decoding returns the value that entered the likelihood.

The tc convention. A merger time is the search's headline number and a constant offset
in it is undetectable downstream: the triggers stay self-consistent and cluster
normally. The midpoint identity alone is not enough, since it also holds when the sign
of ``tc - tc_mid`` is flipped, so monotonicity and a millisecond check are asserted too.

The last group asserts what the schema does *not* contain. The network predicts tc and
mchirp; no mass, spin, distance or sky column may appear anywhere in this path.

Runs anywhere; needs no data, no GPU and no network.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from sage.core.math import Normalise
from sage.search.decode import (
    PE_TARGETS,
    PE_TRIGGER_COLUMNS,
    SIGMA_MAX,
    SIGMA_MIN,
    DecodedPE,
    PEDecoder,
    tc_to_gps,
)
from sage.search.geometry import SearchGeometry
from sage.search.triggers import TRIGGER_COLUMNS, TriggerTable


def _mchirp(m1: float, m2: float) -> float:
    """Chirp mass, as ``DistributionSampler.theoretical_bounds`` computes it."""
    return ((m1 * m2) ** (3.0 / 5.0)) / ((m1 + m2) ** (1.0 / 5.0))


# The production prior, runs/o3b/gwconfig.yaml: tc uniform on [11.0, 11.2] s from the
# start of the 12 s of analysis content, component masses uniform on [7, 50].
TC_PRIOR = (11.0, 11.2)
MCHIRP_PRIOR = (_mchirp(7.0, 7.0), _mchirp(50.0, 50.0))
# Moments of a uniform prior, which is what the sampler measures for tc.
TC_MEAN = 11.1
TC_STD = 0.2 / np.sqrt(12.0)

# A second set whose every constant is exact in binary floating point, so the affine and
# GPS tests can demand equality rather than closeness. Anything approximate there would
# hide a discrepancy of exactly the size a wrong-but-close constant produces.
EXACT_TC_PRIOR = (11.0, 11.25)
EXACT_MCHIRP_PRIOR = (6.0, 44.0)
EXACT_MOMENTS = {"tc": (11.125, 0.0625), "mchirp": (20.0, 6.0)}

GPS_START = 1238166018.0
# Spacing of float64 at the O3a epoch: 2**30 * 2**-52. An absolute GPS time cannot be
# resolved finer than this, which is 0.05 per cent of one sample at 2048 Hz.
GPS_ULP_S = 2.5e-7

PRODUCTION_GEOMETRY = dict(
    sample_rate=2048.0,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=205,
)


class ToyParamSampler:
    """
    A parameter sampler exposing what ``DistributionSampler`` exposes to a decoder.

    Mirrors the real attribute surface rather than a convenient subset: ``param_index``
    over every sampled parameter, the encoding buffers registered by
    ``_compile_batch_standardiser`` and ``_compile_batch_normaliser`` in
    ``cfg.do_point_estimate`` order, and the ``normalisers`` holding the prior bounds
    (the real :class:`~sage.core.math.Normalise`, not a double of it).

    Two properties are deliberately awkward, because the real sampler has them and a
    tidier stand-in would let a real bug pass. ``param_names`` is sorted, so mchirp
    precedes tc and the buffer order is *not* the parameter order -- a decoder that
    indexes the buffers positionally decodes tc with mchirp's constants and is caught.
    And the buffers are float32, the training dtype, so a decoder that assumes exact
    float64 constants fails here as it would in production.
    """

    def __init__(self, moments, priors, targets=PE_TARGETS, dtype=np.float32):
        self.param_names = sorted(priors)
        self.param_index = {name: i for i, name in enumerate(self.param_names)}
        self.normalisers = {
            name: Normalise(min_val=low, max_val=high)
            for name, (low, high) in priors.items()
        }
        self.bounds = dict(priors)

        order = [self.param_index[name] for name in targets]
        columns = np.asarray(order, dtype=np.int64)
        self._std_indices = columns
        self._norm_indices = columns.copy()
        self._std_means = np.asarray([moments[n][0] for n in targets], dtype=dtype)
        self._std_stds = np.asarray([moments[n][1] for n in targets], dtype=dtype)
        self._norm_mins = np.asarray([priors[n][0] for n in targets], dtype=dtype)
        self._norm_scales = np.asarray(
            [priors[n][1] - priors[n][0] for n in targets], dtype=dtype
        )


def _training_sigma(raw, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
    """
    The training loss's own sigma mapping, as the oracle for the decoder's.

    Constructs the real ``BCEWithPEsigmaLoss`` -- with ``get_cfg`` patched, exactly as
    tests/test_loss_functions.py does, since the loss reads only ``do_point_estimate``
    at construction -- and calls its private ``_sigma``. Reimplementing the mapping in
    this file would test the copy, not the contract; this way the assertion fails the
    moment training changes its parameterisation and the decoder does not follow.
    """
    from sage.architecture.custom_losses.loss_functions import BCEWithPEsigmaLoss

    class _Cfg:
        do_point_estimate = ["tc", "mchirp"]

    path = "sage.architecture.custom_losses.loss_functions.get_cfg"
    with patch(path, return_value=_Cfg()):
        loss = BCEWithPEsigmaLoss(sigma_min=sigma_min, sigma_max=sigma_max)
    return loss._sigma(torch.as_tensor(raw, dtype=torch.float64)).numpy()


@pytest.fixture
def sampler():
    """The production prior: uniform tc on [11.0, 11.2], masses on [7, 50]."""
    return ToyParamSampler(
        moments={"tc": (TC_MEAN, TC_STD), "mchirp": (20.0, 6.0)},
        priors={
            "distance": (100.0, 5000.0),
            "mass1": (7.0, 50.0),
            "mass2": (7.0, 50.0),
            "mchirp": MCHIRP_PRIOR,
            "tc": TC_PRIOR,
        },
    )


@pytest.fixture
def exact_sampler():
    """The same shape of sampler, with constants exact in binary floating point."""
    return ToyParamSampler(
        moments=EXACT_MOMENTS,
        priors={
            "distance": (100.0, 5000.0),
            "mass1": (7.0, 50.0),
            "mass2": (7.0, 50.0),
            "mchirp": EXACT_MCHIRP_PRIOR,
            "tc": EXACT_TC_PRIOR,
        },
    )


@pytest.fixture
def geometry():
    """Production window and stride, carrying the production tc prior."""
    return SearchGeometry(
        **PRODUCTION_GEOMETRY, tc_lower_s=TC_PRIOR[0], tc_upper_s=TC_PRIOR[1]
    )


@pytest.fixture
def exact_geometry():
    """The geometry matching ``exact_sampler``; its tc midpoint is 11.125 s exactly."""
    return SearchGeometry(
        **PRODUCTION_GEOMETRY,
        tc_lower_s=EXACT_TC_PRIOR[0],
        tc_upper_s=EXACT_TC_PRIOR[1],
    )


@pytest.fixture
def decoder(sampler, geometry):
    return PEDecoder(PE_TARGETS, sampler, geometry=geometry)


@pytest.fixture
def exact_decoder(exact_sampler, exact_geometry):
    return PEDecoder(PE_TARGETS, exact_sampler, geometry=exact_geometry)


class TestSplit:
    """The head layout is blocked, and the split has to read it that way."""

    def test_split_is_blocked(self, decoder):
        """
        Columns are ``[mu_tc, mu_mchirp, sraw_tc, sraw_mchirp]``, not interleaved.

        The two readings differ only in which columns pair with which, so an interleaved
        split still returns two arrays of the right shape and every value in range. The
        fixture uses four distinct values so the wrong pairing is arithmetically
        visible: interleaving would return means [1, 3] instead of [1, 2].
        """
        block = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        mu, raw = decoder.split(block)
        assert mu.tolist() == [[1.0, 2.0], [5.0, 6.0]]
        assert raw.tolist() == [[3.0, 4.0], [7.0, 8.0]]

    def test_split_promotes_to_float64(self, decoder):
        """
        A float32 head output is promoted before any arithmetic touches it.

        The mean becomes an offset on a GPS time near 1.24e9, where float32 is spaced
        128 s apart. Promoting after the addition would be far too late.
        """
        block = np.zeros((3, 4), dtype=np.float32)
        mu, raw = decoder.split(block)
        assert mu.dtype == np.float64 and raw.dtype == np.float64

    def test_split_refuses_1d(self, decoder):
        """A 1-D array is ambiguous between one window and one column."""
        with pytest.raises(ValueError):
            decoder.split(np.zeros(4))

    def test_split_refuses_mean_only_head(self, decoder):
        """
        A head with no sigma columns is refused, not padded with NaN.

        ``tc_sigma`` and ``mchirp_sigma`` are shard columns. A NaN there is read
        downstream as an uncertainty rather than as a missing head, and the shard is
        already written by the time anyone notices.
        """
        with pytest.raises(ValueError, match="mean-only"):
            decoder.split(np.zeros((2, 2)))

    def test_split_refuses_wrong_width(self, decoder):
        """A head of some other width belongs to a different configuration."""
        with pytest.raises(ValueError):
            decoder.split(np.zeros((2, 5)))


class TestSigma:
    """The decoded sigma must be the number that entered the training likelihood."""

    def test_sigma_matches_training_loss(self, decoder):
        """
        Agrees with ``BCEWithPEsigmaLoss._sigma`` across the whole reachable range.

        The oracle is the training code itself, called here, so this fails if either
        side changes its parameterisation. The sweep spans the floor, the linear region
        and the cap; testing near zero alone passes for an implementation with no clamp.
        """
        raw = np.linspace(-60.0, 60.0, 241)
        got = decoder.sigma(raw)
        assert np.allclose(got, _training_sigma(raw), rtol=0.0, atol=1e-12)

    def test_sigma_floor_and_cap(self, decoder):
        """
        Saturates at ``sigma_min`` below and ``sigma_max`` above, exactly.

        Literal bounds rather than a comparison against the oracle, so a coordinated
        change of both implementations still has to move these two numbers deliberately.
        """
        assert decoder.sigma(np.array([-1e3])) == pytest.approx(SIGMA_MIN, abs=0.0)
        assert decoder.sigma(np.array([1e3])) == pytest.approx(SIGMA_MAX, abs=0.0)

    def test_sigma_survives_extreme_logit(self, decoder):
        """
        A huge raw output stays finite.

        ``log1p(exp(raw))`` overflows to inf above ~709 and the clamp cannot recover
        from that -- ``clip(inf)`` is ``sigma_max``, but an ``inf`` reaching a physical
        sigma by any other route is not detectable. The stable form stays finite.
        """
        got = decoder.sigma(np.array([1e6, 709.0, 750.0]))
        assert np.isfinite(got).all()
        assert got.tolist() == [SIGMA_MAX, SIGMA_MAX, SIGMA_MAX]


class TestDecodeValues:
    """The affine inverse of the training encoding, target by target."""

    def test_decode_inverts_standardisation(self, exact_decoder):
        """
        ``value = mu * std + mean``, against literal hand-computed numbers.

        The constants here are exact in binary floating point, so equality is demanded
        rather than closeness: a decoder that used mchirp's std for tc, or dropped the
        mean, lands on a different number in the same plausible range, and an
        approximate comparison with a loose tolerance is precisely what would let that
        through.
        """
        block = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]])
        decoded = exact_decoder.decode(block)
        # tc: 11.125 + {0, 1} * 0.0625; mchirp: 20 + {0, -1} * 6.
        assert decoded.values["tc"].tolist() == [11.125, 11.1875]
        assert decoded.values["mchirp"].tolist() == [20.0, 14.0]

    def test_decode_maps_targets_by_name(self, exact_sampler, exact_geometry):
        """
        Targets are resolved through ``param_index``, not by buffer position.

        The sampler's buffers are in ``do_point_estimate`` order while its
        ``param_names`` is sorted, and here the decoder is asked for the reverse order
        again. A positional read decodes tc with mchirp's mean of 20 and mchirp with
        tc's 11.125 -- both in range, both wrong, neither raising.
        """
        decoder = PEDecoder(
            ("mchirp", "tc"), exact_sampler, geometry=exact_geometry
        )
        decoded = decoder.decode(np.zeros((1, 4)))
        assert decoded.values["tc"].tolist() == [11.125]
        assert decoded.values["mchirp"].tolist() == [20.0]

    def test_decode_scales_sigma_to_physical(self, exact_decoder):
        """
        The sigma is scaled by the same factor as the mean, and by nothing else.

        Both encodings are affine, so the physical width is the encoded width times the
        scale. Checked against the literal value of ``softplus(0) + sigma_min`` times
        each target's std, which differ by a factor of 96 here, so applying one target's
        scale to the other is unmistakable.
        """
        decoded = exact_decoder.decode(np.zeros((1, 4)))
        unit = float(np.log(2.0)) + SIGMA_MIN
        assert decoded.sigmas["tc"][0] == pytest.approx(unit * 0.0625, rel=1e-12)
        assert decoded.sigmas["mchirp"][0] == pytest.approx(unit * 6.0, rel=1e-12)

    def test_decode_minmax_inverts_bounds(self, exact_sampler, exact_geometry):
        """
        Under ``pe_target_minmax`` the map is the prior: 0 is the lower bound, 1 the
        upper.

        A checkpoint trained min-max and decoded standardised, or the reverse, produces
        values in the wrong range entirely, and nothing about the head output reveals
        which encoding it was trained with -- only the config does.
        """
        decoder = PEDecoder(
            PE_TARGETS,
            exact_sampler,
            pe_target_minmax=True,
            geometry=exact_geometry,
        )
        decoded = decoder.decode(np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0]]))
        assert decoded.values["tc"].tolist() == [11.0, 11.25]
        assert decoded.values["mchirp"].tolist() == [6.0, 25.0]

    def test_minmax_sigma_uses_prior_width(self, exact_sampler, exact_geometry):
        """
        Under min-max the sigma is scaled by the prior width, not by the std.

        ``sage.factory.testing._physical_pe`` multiplies by ``_std_stds`` whichever
        encoding is in force, which is right for the standardised path and wrong by a
        factor of four for tc here and six for mchirp. Both wrong answers are positive,
        finite and plausible, so only an explicit comparison separates them.
        """
        decoder = PEDecoder(
            PE_TARGETS,
            exact_sampler,
            pe_target_minmax=True,
            geometry=exact_geometry,
        )
        decoded = decoder.decode(np.zeros((1, 4)))
        unit = float(np.log(2.0)) + SIGMA_MIN
        assert decoded.sigmas["tc"][0] == pytest.approx(unit * 0.25, rel=1e-12)
        assert decoded.sigmas["mchirp"][0] == pytest.approx(unit * 38.0, rel=1e-12)

    def test_decode_refuses_nan(self, decoder):
        """
        A NaN head output is a fault to report, not a value to write.

        It would flow into ``tc_gps`` and from there into clustering, where it compares
        false against every other trigger and forms a cluster of its own.
        """
        block = np.zeros((2, 4))
        block[1, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            decoder.decode(block)

    def test_decode_refuses_inf(self, decoder):
        """An infinite mean decodes to an infinite merger time."""
        block = np.zeros((2, 4))
        block[0, 1] = np.inf
        with pytest.raises(ValueError, match="finite"):
            decoder.decode(block)

    def test_decode_reports_length_and_names(self, decoder):
        """The set knows how many windows it covers and which targets it holds."""
        decoded = decoder.decode(np.zeros((5, 4)))
        assert len(decoded) == 5
        assert decoded.names == PE_TARGETS
        value, sigma = decoded.column("mchirp")
        assert value.shape == (5,) and sigma.shape == (5,)

    def test_decode_rejects_unknown_column(self, decoder):
        """Asking for a parameter the network never predicted raises, not returns."""
        decoded = decoder.decode(np.zeros((2, 4)))
        with pytest.raises(KeyError):
            decoded.column("mass1")


class TestPriorRail:
    """Saturation is flagged, and flagging it does not alter the value."""

    def test_rail_false_inside_prior(self, decoder):
        """A value comfortably inside the prior is not flagged."""
        decoded = decoder.decode(np.zeros((1, 4)))
        assert decoded.at_prior_rail["tc"].tolist() == [False]
        assert decoded.at_prior_rail["mchirp"].tolist() == [False]

    def test_rail_flags_bounds(self, exact_decoder):
        """
        True at the bound and beyond it, in both directions.

        Exactly at the bound counts: the head is a linear layer with no bound of its
        own, so a value sitting on the prior edge is saturation and not coincidence.
        The standardised inputs are chosen to land on 11.0, 11.25 and past both.
        """
        # tc = 11.125 + mu * 0.0625, so mu = -2 -> 11.0 and mu = +2 -> 11.25.
        block = np.array(
            [[-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [-4.0, 0.0, 0.0, 0.0]]
        )
        decoded = exact_decoder.decode(block)
        assert decoded.at_prior_rail["tc"].tolist() == [True, True, True]

    def test_rail_leaves_values_unclipped(self, exact_decoder):
        """
        A saturated value is reported as it came out, not clipped onto the bound.

        Clipping would pile every saturated window onto one number, which then reads as
        structure in any tc distribution plotted from the shard.
        """
        decoded = exact_decoder.decode(np.array([[-4.0, 0.0, 0.0, 0.0]]))
        assert decoded.values["tc"].tolist() == [10.875]
        assert decoded.at_prior_rail["tc"].tolist() == [True]

    def test_rail_is_boolean(self, decoder):
        """
        The flag is a bool array, so masking with it selects rather than indexes.

        An integer flag column used as a mask is fancy indexing: it silently reorders
        and repeats triggers instead of selecting them.
        """
        decoded = decoder.decode(np.zeros((2, 4)))
        assert decoded.at_prior_rail["tc"].dtype == bool


class TestTcToGps:
    """Where a decoded tc lands on the GPS axis."""

    def test_tc_mid_lands_on_window_gps(self, exact_geometry):
        """
        A tc at the prior midpoint is exactly the window's nominal trigger time.

        This is the whole convention in one identity: ``window_gps`` is what every other
        layer of the search calls the window's time, so decoding must agree with it bit
        for bit at the centre of the prior rather than to some tolerance.
        """
        got = tc_to_gps(exact_geometry, GPS_START, exact_geometry.tc_mid_s)
        assert got == exact_geometry.window_gps(GPS_START)

    def test_tc_offset_is_padding_plus_tc(self, geometry):
        """
        The offset from the raw window start is the one-sided padding plus tc.

        An independent statement of the same convention, written the way
        runs/hiatus/benchmark_mlgwsc1.py writes it: tc is measured from the start of the
        analysis content, which begins ``padding_length_s`` into the raw window.

        Compared at ``GPS_ULP_S``. A float64 near 1.24e9 is spaced 2.4e-7 s, so that is
        the resolution an absolute GPS time has at all, whatever the arithmetic -- three
        orders below one sample at 2048 Hz and four below the H1-L1 baseline.
        """
        for tc in (11.0, 11.1, 11.2):
            got = tc_to_gps(geometry, GPS_START, tc)
            assert got - GPS_START == pytest.approx(
                geometry.padding_length_s + tc, abs=GPS_ULP_S
            )

    def test_tc_gps_increases_with_tc(self, geometry):
        """
        A later coalescence time gives a later GPS time.

        The midpoint identity alone does not catch a sign flip: ``window_gps -
        (tc - tc_mid)`` satisfies it too, and mirrors every trigger about the window
        centre -- a displacement of up to 0.1 s that no later stage can see.
        """
        early = tc_to_gps(geometry, GPS_START, 11.05)
        late = tc_to_gps(geometry, GPS_START, 11.15)
        assert late > early
        assert late - early == pytest.approx(0.1, abs=GPS_ULP_S)

    def test_tc_gps_keeps_millisecond_detail(self, geometry):
        """
        A 1 ms shift in tc survives into the absolute time.

        The light-travel time between H1 and L1 is 10 ms, so millisecond structure is
        the scale coincidence is decided on. The strict inequality is the part with
        teeth: in float32 both times round to the same value and the difference is
        exactly zero, since the float32 grid at 1.24e9 is 128 s wide.
        """
        base = tc_to_gps(geometry, GPS_START, 11.1)
        shifted = tc_to_gps(geometry, GPS_START, 11.101)
        assert shifted > base
        assert shifted - base == pytest.approx(0.001, abs=GPS_ULP_S)
        assert np.float32(shifted) == np.float32(base)

    def test_tc_gps_refuses_float32_start(self, geometry):
        """
        A float32 window start is refused rather than upcast.

        At 1.24e9 the float32 grid is 128 s coarse -- the fixture start is not even
        representable -- so upcasting would keep an error of up to two minutes and
        destroy the evidence of where it came from.
        """
        assert float(np.float32(GPS_START)) != GPS_START
        with pytest.raises(TypeError):
            tc_to_gps(geometry, np.float32([GPS_START]), 11.1)

    def test_tc_gps_refuses_length_mismatch(self, geometry):
        """Each trigger carries its own window start; a short list would broadcast."""
        with pytest.raises(ValueError):
            tc_to_gps(geometry, np.array([GPS_START, GPS_START]), np.array([11.1]))

    def test_tc_gps_needs_geometry(self, sampler):
        """A decoder built without a geometry cannot place tc on the GPS axis."""
        bare = PEDecoder(PE_TARGETS, sampler)
        with pytest.raises(ValueError, match="SearchGeometry"):
            bare.tc_gps(np.array([GPS_START]), np.array([11.1]))


class TestTriggerColumns:
    """What decoding writes into a shard, and what it must never write."""

    def test_columns_are_shard_schema(self, decoder):
        """
        Exactly the four point-estimate columns of the trigger schema, and no others.

        The shard schema is what fixes decoding to the inference loop: if the columns
        drifted apart, a shard would carry a column no reader knows about and the next
        stage that copies the table would drop it.
        """
        columns = decoder.trigger_columns(
            np.zeros((3, 4)), np.full(3, GPS_START, dtype=np.float64)
        )
        assert set(columns) == {"tc_gps", "tc_sigma", "mchirp", "mchirp_sigma"}
        assert set(columns) <= set(TRIGGER_COLUMNS)

    def test_columns_carry_no_masses(self, decoder):
        """
        No component mass, spin, distance or sky column is produced.

        The network has two point-estimate heads. A chirp mass does not separate m1 from
        m2, so any such column here would be a fabricated parameter presented with the
        same authority as a measured one.
        """
        columns = decoder.trigger_columns(
            np.zeros((2, 4)), np.full(2, GPS_START, dtype=np.float64)
        )
        forbidden = {
            "mass1",
            "mass2",
            "spin1z",
            "spin2z",
            "distance",
            "ra",
            "dec",
            "inclination",
        }
        assert not forbidden & set(columns)

    def test_tc_column_is_absolute_gps(self, exact_decoder, exact_geometry):
        """
        ``tc_gps`` holds an absolute time, not the within-window seconds the head emits.

        A trigger is compared against other detectors, other slides and a catalogue,
        none of which knows which window it came from. Writing the window-relative
        11.125 s would be silently accepted by every one of those comparisons.
        """
        starts = np.array([GPS_START, GPS_START + 100.0])
        columns = exact_decoder.trigger_columns(np.zeros((2, 4)), starts)
        assert columns["tc_gps"].tolist() == [
            exact_geometry.window_gps(starts[0]),
            exact_geometry.window_gps(starts[1]),
        ]

    def test_columns_refuse_scalar_start(self, decoder):
        """
        One window start per window; a scalar would broadcast.

        Every trigger in the block would then be dated from the first window's start,
        which is a monotonically growing error along the block.
        """
        with pytest.raises(ValueError):
            decoder.trigger_columns(np.zeros((3, 4)), GPS_START)

    def test_columns_accepted_by_table(self, decoder):
        """
        The columns build a ``TriggerTable`` without being rejected or renamed.

        ``TriggerTable`` refuses any column outside the schema, so this is the
        end-to-end statement that decoding and the writer agree on names and lengths.
        """
        columns = decoder.trigger_columns(
            np.zeros((4, 4)), np.full(4, GPS_START, dtype=np.float64)
        )
        columns["stat"] = np.zeros(4)
        columns["gps"] = np.full(4, GPS_START)
        table = TriggerTable(columns=columns, attrs={})
        assert len(table) == 4
        assert table["tc_gps"].dtype == np.float64


class TestConstruction:
    """Configurations that would decode to plausible wrong numbers are refused."""

    def test_missing_buffers_refused(self, sampler, geometry):
        """
        A sampler with no standardisation buffers cannot be decoded against.

        The alternative is to invent a mean and a std, which produces physical-looking
        values wrong by whatever the training prior actually was.
        """
        del sampler._std_means
        with pytest.raises(AttributeError, match="_std_means"):
            PEDecoder(PE_TARGETS, sampler, geometry=geometry)

    def test_missing_prior_bounds_refused(self, sampler, geometry):
        """
        Without prior bounds there is no rail flag, and an all-false flag is a claim.

        Reporting every window as un-saturated would state that the head never left the
        prior, which is the opposite of what an unbounded linear head does.
        """
        del sampler.normalisers["tc"]
        del sampler.bounds["tc"]
        with pytest.raises(KeyError):
            PEDecoder(PE_TARGETS, sampler, geometry=geometry)

    def test_zero_scale_refused(self, sampler, geometry):
        """
        A zero std collapses every window onto the mean.

        The result is a constant that still looks like an estimate: a whole run of
        triggers reporting the same chirp mass, with no error anywhere to trace it to.
        """
        sampler._std_stds = np.asarray([TC_STD, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="scale"):
            PEDecoder(PE_TARGETS, sampler, geometry=geometry)

    def test_geometry_tc_mismatch_refused(self, sampler):
        """
        A geometry whose tc prior is not the trained one is refused at construction.

        This is the defect the rest of the suite cannot see: the triggers stay
        self-consistent and cluster normally, and every merger time is displaced by half
        the difference between the two priors -- 3.05 s for the bounds used here.
        """
        wrong = SearchGeometry(**PRODUCTION_GEOMETRY, tc_lower_s=5.0, tc_upper_s=7.0)
        with pytest.raises(ValueError, match="tc prior"):
            PEDecoder(PE_TARGETS, sampler, geometry=wrong)

    def test_minmax_buffers_must_match_prior(self, sampler, geometry):
        """
        Under min-max the affine map is the prior, so the two must agree.

        Checked at 1e-6 relative because the buffers are float32; a genuine mismatch is
        of order the prior width, five orders larger. Shifting the lower bound by 1 s
        here is well inside that gap and must still be caught.
        """
        sampler._norm_mins = np.asarray(
            [TC_PRIOR[0] + 1.0, MCHIRP_PRIOR[0]], dtype=np.float32
        )
        with pytest.raises(ValueError, match="min-max"):
            PEDecoder(PE_TARGETS, sampler, pe_target_minmax=True, geometry=geometry)

    def test_minmax_tolerates_float32_buffers(self, sampler, geometry):
        """
        The float32 buffers the real sampler registers pass the same check.

        Without the tolerance, ``11.2 - 11.0`` evaluated in float32 (0.19999981) against
        the float64 prior width would fail every production configuration.
        """
        decoder = PEDecoder(
            PE_TARGETS, sampler, pe_target_minmax=True, geometry=geometry
        )
        decoded = decoder.decode(np.array([[0.0, 0.0, 0.0, 0.0]]))
        assert decoded.values["tc"][0] == pytest.approx(TC_PRIOR[0], abs=1e-6)

    def test_duplicate_targets_refused(self, sampler, geometry):
        """A repeated target would write one shard column twice from one head."""
        with pytest.raises(ValueError, match="repeated"):
            PEDecoder(("tc", "tc"), sampler, geometry=geometry)

    def test_unknown_target_refused(self, sampler, geometry):
        """
        A target the sampler's prior does not contain is refused by name.

        The network cannot have been trained to predict a parameter that was never
        sampled, so this is a checkpoint/prior mismatch and not a decoding choice.
        """
        with pytest.raises(KeyError):
            PEDecoder(("tc", "spin1z"), sampler, geometry=geometry)

    def test_empty_targets_refused(self, sampler, geometry):
        """A decoder with no targets would silently produce no shard columns."""
        with pytest.raises(ValueError):
            PEDecoder((), sampler, geometry=geometry)

    def test_sigma_bounds_must_be_ordered(self, sampler, geometry):
        """An inverted clamp has an empty range and returns the upper bound always."""
        with pytest.raises(ValueError, match="sigma"):
            PEDecoder(
                PE_TARGETS, sampler, geometry=geometry, sigma_min=1.0, sigma_max=0.5
            )


class TestSchemaConstants:
    """The module's constants and the shard schema must stay in step."""

    def test_pe_targets_are_tc_and_mchirp(self):
        """
        The whole inventory, in ``cfg.do_point_estimate`` order.

        Ordering is part of the contract: it is the order of the blocked head layout,
        and swapping the two is arithmetically silent.
        """
        assert PE_TARGETS == ("tc", "mchirp")

    def test_pe_columns_in_trigger_schema(self):
        """
        Every column decoding produces exists in ``TRIGGER_COLUMNS``.

        A column no reader knows about is written and then dropped by the next stage
        that copies the table, without an error anywhere.
        """
        produced = [name for pair in PE_TRIGGER_COLUMNS.values() for name in pair]
        assert set(PE_TRIGGER_COLUMNS) == set(PE_TARGETS)
        assert set(produced) <= set(TRIGGER_COLUMNS)
        assert len(set(produced)) == len(produced)

    def test_schema_has_no_mass_columns(self):
        """
        The shard schema itself contains no parameter the network does not predict.

        Stated here rather than left implicit: the moment a mass or spin column exists,
        something has to fill it, and nothing in this pipeline can.
        """
        forbidden = {"mass1", "mass2", "spin1z", "spin2z", "distance", "ra", "dec"}
        assert not forbidden & set(TRIGGER_COLUMNS)


class TestDecodedPE:
    """The container refuses a set that does not describe one batch of windows."""

    def test_mismatched_keys_refused(self):
        """A missing sigma surfaces only when a later stage asks for that column."""
        with pytest.raises(ValueError):
            DecodedPE(
                values={"tc": np.zeros(2), "mchirp": np.zeros(2)},
                sigmas={"tc": np.zeros(2)},
                at_prior_rail={"tc": np.zeros(2, bool), "mchirp": np.zeros(2, bool)},
            )

    def test_mismatched_lengths_refused(self):
        """Columns of unequal length do not describe one batch of windows."""
        with pytest.raises(ValueError):
            DecodedPE(
                values={"tc": np.zeros(3)},
                sigmas={"tc": np.zeros(2)},
                at_prior_rail={"tc": np.zeros(3, bool)},
            )

    def test_numeric_rail_flag_refused(self):
        """An integer flag used as a mask is fancy indexing, not selection."""
        with pytest.raises(TypeError):
            DecodedPE(
                values={"tc": np.zeros(2)},
                sigmas={"tc": np.zeros(2)},
                at_prior_rail={"tc": np.zeros(2, dtype=np.int64)},
            )

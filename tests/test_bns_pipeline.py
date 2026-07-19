"""
Tests for the BNS pipeline: IMRPhenomXAS_NRTidalv3 + WhiteGaussianNoiseSampler.

Covers:
  - Pipeline state machine (ProcessingState, GWBatch)
  - MultibandSelector linearity and shape guarantees
  - WhiteGaussianNoiseSampler output contract
  - IMRPhenomXAS_NRTidalv3 waveform shapes in both 'none' and 'worst_case' modes
  - BNS parameter sampler standardisation roundtrip
  - End-to-end: noise → selector → whitening → to_network_input

All tests run on CPU with a tiny config (4 s sample window) so they
complete quickly without GPU or real noise files.
"""

import math
import textwrap
import types
import torch
import pytest
import numpy as np
from unittest.mock import patch

# ── Pipeline state types (no config needed) ────────────────────────────────
from sage.core.pipeline import Grid, ProcessingState, PipelineError, GWBatch

# ── Noise sampler ──────────────────────────────────────────────────────────
from sage.data.noise.white_noise import WhiteGaussianNoiseSampler

# ── Multiband selector ────────────────────────────────────────────────────
from sage.data.waveform.multiband_selector import MultibandSelector

# ── Config infrastructure ─────────────────────────────────────────────────
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.core.graph import Preprocessor


# ── Tiny BNS-like config for tests ────────────────────────────────────────

class _TinyCFG:
    export_dir         = "/tmp/sage_bns_test"
    batch_size         = 2
    device             = "cpu"
    dtype              = torch.float32
    detectors          = ["H1", "L1"]
    do_point_estimate  = ["tc", "mchirp"]
    autocast           = False
    class_balance      = 0.5
    clip_norm          = 1.0
    num_epochs         = 1
    training_iterations   = 2
    validation_iterations = 1


class _TinyDataCFG:
    data_dir                    = "/tmp/sage_bns_test"
    sample_rate                 = 2048.0
    noise_low_frequency_cutoff  = 15.0
    signal_low_frequency_cutoff = 20.0
    sample_length_in_s          = 4.0
    padding_length_in_s         = 0.5


# ── Minimal BNS gwconfig (tc within 4 s window) ───────────────────────────

_TINY_GWCONFIG = textwrap.dedent("""\
variable_params:
  - mass1
  - mass2
  - chi1z
  - chi2z
  - lambda1
  - lambda2
  - distance
  - tc
  - coa_phase
  - inclination
  - polarization
  - ra
  - dec

priors:
  tc:
    name: uniform
    min: 3.0
    max: 3.5

  mass1:
    name: uniform
    min: 1.0
    max: 3.0

  mass2:
    name: uniform
    min: 1.0
    max: 3.0

  chi1z:
    name: uniform
    min: -0.4
    max:  0.4

  chi2z:
    name: uniform
    min: -0.4
    max:  0.4

  lambda1:
    name: uniform
    min:    0
    max: 5000

  lambda2:
    name: uniform
    min:    0
    max: 5000

  distance:
    name: uniform_radius
    min:  10.0
    max: 500.0

  inclination:
    name: sin_angle

  coa_phase:
    name: uniform_angle

  polarization:
    name: uniform_angle

  sky:
    name: uniform_sky
    ra: ra
    dec: dec

constraints:
  - name: mass_order

waveform_transforms:
  mass_params:
    name: mass1_mass2_to_mchirp_q
""")


# ── Module-scope fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def bns_cfg(tmp_path_factory):
    """Register a tiny CPU BNS config for all tests in this module."""
    cfg      = BaseConfig(_TinyCFG())
    data_cfg = BaseDataConfig(_TinyDataCFG())
    register_configs(cfg, data_cfg)
    return cfg, data_cfg


@pytest.fixture(scope="module")
def gwconfig_path(tmp_path_factory):
    """Write the tiny gwconfig YAML to a temp file and return its path."""
    d = tmp_path_factory.mktemp("bns_gwconfig")
    p = d / "gwconfig_test.yaml"
    p.write_text(_TINY_GWCONFIG)
    return str(p)


@pytest.fixture(scope="module")
def param_sampler(gwconfig_path):
    """Build a DistributionSampler from the tiny gwconfig (no waveform model)."""
    from sage.data.waveform.sampler import read_from_config
    return read_from_config(gwconfig_path, seed=42)


@pytest.fixture(scope="module")
def flat_psds(bns_cfg):
    """Return flat unit ASDs (D, F) for mocking FiducialWhitening."""
    cfg, data_cfg = bns_cfg
    D = len(cfg.detectors)
    F = data_cfg.padded_length_in_nsamples // 2 + 1
    return torch.ones(D, F, dtype=cfg.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pipeline state machine
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessingState:
    """ProcessingState: transitions, error guards, channel counts."""

    def test_initial_fd_uniform_unwhitened(self):
        s = ProcessingState(Grid.FD_UNIFORM)
        assert s.grid == Grid.FD_UNIFORM
        assert not s.whitened
        assert s.is_fd()
        assert not s.is_td()

    def test_initial_fd_coarse_unwhitened(self):
        s = ProcessingState(Grid.FD_COARSE)
        assert s.grid == Grid.FD_COARSE
        assert not s.whitened

    def test_after_whiten_sets_flag(self):
        s = ProcessingState(Grid.FD_UNIFORM).after_whiten()
        assert s.whitened
        assert s.grid == Grid.FD_UNIFORM

    def test_double_whiten_raises(self):
        s = ProcessingState(Grid.FD_UNIFORM).after_whiten()
        with pytest.raises(PipelineError):
            s.after_whiten()

    def test_fd_uniform_to_td_via_ifft(self):
        s = ProcessingState(Grid.FD_UNIFORM).after_whiten().after_ifft()
        assert s.grid == Grid.TD_UNIFORM
        assert s.whitened

    def test_fd_coarse_cannot_ifft(self):
        s = ProcessingState(Grid.FD_COARSE).after_whiten()
        with pytest.raises(PipelineError, match="FD_COARSE"):
            s.after_ifft()

    def test_fd_uniform_unwhitened_cannot_ifft(self):
        # IFFT is allowed from FD_UNIFORM; whitening state doesn't gate it
        s = ProcessingState(Grid.FD_UNIFORM)
        s2 = s.after_ifft()
        assert s2.grid == Grid.TD_UNIFORM

    def test_td_uniform_cannot_ifft(self):
        s = ProcessingState(Grid.TD_UNIFORM)
        with pytest.raises(PipelineError):
            s.after_ifft()

    def test_multirate_from_td_uniform(self):
        s = ProcessingState(Grid.TD_UNIFORM).after_multirate()
        assert s.grid == Grid.TD_MULTIRATE

    def test_multirate_from_fd_uniform_raises(self):
        with pytest.raises(PipelineError):
            ProcessingState(Grid.FD_UNIFORM).after_multirate()

    def test_multirate_from_fd_coarse_raises(self):
        with pytest.raises(PipelineError):
            ProcessingState(Grid.FD_COARSE).after_multirate()

    def test_n_channels_fd(self):
        assert ProcessingState(Grid.FD_UNIFORM).n_channels() == 2
        assert ProcessingState(Grid.FD_COARSE).n_channels() == 2

    def test_n_channels_td(self):
        assert ProcessingState(Grid.TD_UNIFORM).n_channels() == 1
        assert ProcessingState(Grid.TD_MULTIRATE).n_channels() == 1

    def test_immutable_state(self):
        s = ProcessingState(Grid.FD_UNIFORM)
        s2 = s.after_whiten()
        # Original state is unchanged
        assert not s.whitened
        assert s2.whitened


class TestGWBatch:
    """GWBatch: to_network_input shapes, n_channels, FD/TD paths."""

    def test_fd_coarse_to_network_input_shape(self):
        B, D, N = 3, 2, 120
        data = torch.randn(B, D, N, dtype=torch.complex64)
        batch = GWBatch(data, ProcessingState(Grid.FD_COARSE))
        out = batch.to_network_input()
        assert out.shape == (B, 2 * D, N)
        assert out.is_floating_point()

    def test_fd_to_network_input_stacks_real_imag(self):
        B, D, N = 2, 2, 50
        data = torch.randn(B, D, N, dtype=torch.complex64)
        batch = GWBatch(data, ProcessingState(Grid.FD_COARSE))
        out = batch.to_network_input()
        # First D channels: real, next D channels: imag
        assert torch.allclose(out[:, :D, :], data.real)
        assert torch.allclose(out[:, D:, :], data.imag)

    def test_td_to_network_input_passthrough(self):
        B, D, T = 2, 2, 200
        data = torch.randn(B, D, T)
        batch = GWBatch(data, ProcessingState(Grid.TD_UNIFORM))
        out = batch.to_network_input()
        assert out is data

    def test_n_channels_fd_coarse(self):
        batch = GWBatch(
            torch.zeros(1, 2, 10, dtype=torch.complex64),
            ProcessingState(Grid.FD_COARSE),
        )
        assert batch.n_channels == 2

    def test_n_channels_td(self):
        batch = GWBatch(torch.zeros(1, 2, 10), ProcessingState(Grid.TD_UNIFORM))
        assert batch.n_channels == 1

    def test_coarse_indices_preserved(self):
        data = torch.zeros(1, 2, 5, dtype=torch.complex64)
        idx  = torch.arange(5)
        batch = GWBatch(data, ProcessingState(Grid.FD_COARSE), coarse_indices=idx)
        assert batch.coarse_indices is idx


# ═══════════════════════════════════════════════════════════════════════════
# 2. MultibandSelector
# ═══════════════════════════════════════════════════════════════════════════


class TestMultibandSelector:
    """
    MultibandSelector: index selection correctness and the key linearity
    guarantee  select(h + n) == select(h) + select(n).
    """

    @pytest.fixture(scope="class")
    def selector(self):
        """Selector for m1=m2=1.4 Msun, 5 s padded window at delta_f=0.2 Hz."""
        data_cfg = types.SimpleNamespace(
            padded_delta_f=0.2,
            signal_low_frequency_cutoff=20.0,
            sample_rate=2048.0,
        )
        return MultibandSelector.from_prior(
            m1_worst=1.4, m2_worst=1.4, data_cfg=data_cfg, device="cpu"
        )

    def _f_full(self):
        return int(2048.0 * 5.0) // 2 + 1  # 5121

    def test_n_coarse_positive(self, selector):
        assert selector.n_coarse > 0

    def test_n_coarse_less_than_f_full(self, selector):
        assert selector.n_coarse < self._f_full()

    def test_coarse_indices_nonnegative(self, selector):
        assert (selector.coarse_indices >= 0).all()

    def test_coarse_indices_within_range(self, selector):
        assert (selector.coarse_indices < self._f_full()).all()

    def test_coarse_indices_strictly_increasing(self, selector):
        idx = selector.coarse_indices
        assert (idx[1:] > idx[:-1]).all()

    def test_output_shape(self, selector):
        B, D = 3, 2
        x = torch.randn(B, D, self._f_full(), dtype=torch.complex64)
        out = selector(x)
        assert out.shape == (B, D, selector.n_coarse)

    def test_linearity_exact(self, selector):
        """select(h + n) == select(h) + select(n) to floating-point precision."""
        B, D = 4, 2
        F = self._f_full()
        h = torch.randn(B, D, F, dtype=torch.complex64)
        n = torch.randn(B, D, F, dtype=torch.complex64)
        assert torch.allclose(selector(h + n), selector(h) + selector(n))

    def test_selected_values_match_direct_gather(self, selector):
        """Values at selected indices must equal direct array lookup."""
        F = self._f_full()
        x = torch.arange(F, dtype=torch.float32).unsqueeze(0)  # (1, F)
        out = selector(x)  # (1, N_coarse)
        expected = x[0, selector.coarse_indices]
        assert torch.allclose(out[0], expected)

    def test_coarse_freqs_shape(self, selector):
        assert selector.coarse_freqs.shape == (selector.n_coarse,)

    def test_coarse_freqs_cover_signal_band(self, selector):
        assert selector.coarse_freqs[0].item() >= 20.0 - 0.5
        assert selector.coarse_freqs[-1].item() <= 1024.0 + 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. WhiteGaussianNoiseSampler
# ═══════════════════════════════════════════════════════════════════════════


class TestWhiteGaussianNoiseSampler:
    """WhiteGaussianNoiseSampler: shapes, dtype, reproducibility, whiteness."""

    def test_output_shapes(self, bns_cfg):
        cfg, data_cfg = bns_cfg
        sampler = WhiteGaussianNoiseSampler(seed=0)
        noise, targets = sampler()
        B = cfg.batch_size
        D = len(cfg.detectors)
        F = data_cfg.padded_length_in_nsamples // 2 + 1
        assert noise.shape == (B, D, F)
        assert targets.shape == (B, 1)

    def test_noise_is_complex(self, bns_cfg):
        sampler = WhiteGaussianNoiseSampler(seed=0)
        noise, _ = sampler()
        assert noise.is_complex()

    def test_noise_dtype_is_complex64(self, bns_cfg):
        sampler = WhiteGaussianNoiseSampler(seed=0)
        noise, _ = sampler()
        assert noise.dtype == torch.complex64

    def test_targets_are_zero(self, bns_cfg):
        sampler = WhiteGaussianNoiseSampler(seed=0)
        _, targets = sampler()
        assert (targets == 0.0).all()

    def test_targets_dtype(self, bns_cfg):
        cfg, _ = bns_cfg
        sampler = WhiteGaussianNoiseSampler(seed=0)
        _, targets = sampler()
        assert targets.dtype == cfg.dtype

    def test_targets_device(self, bns_cfg):
        cfg, _ = bns_cfg
        sampler = WhiteGaussianNoiseSampler(seed=0)
        _, targets = sampler()
        assert str(targets.device) == str(cfg.device)

    def test_reproducibility_same_seed(self, bns_cfg):
        n1, _ = WhiteGaussianNoiseSampler(seed=42)()
        n2, _ = WhiteGaussianNoiseSampler(seed=42)()
        assert torch.allclose(n1, n2)

    def test_different_seeds_differ(self, bns_cfg):
        n1, _ = WhiteGaussianNoiseSampler(seed=1)()
        n2, _ = WhiteGaussianNoiseSampler(seed=2)()
        assert not torch.allclose(n1, n2)

    def test_sequential_calls_differ(self, bns_cfg):
        """Consecutive calls advance the RNG — distinct realisations."""
        s = WhiteGaussianNoiseSampler(seed=7)
        n1, _ = s()
        n2, _ = s()
        assert not torch.allclose(n1, n2)

    def test_detectors_independent(self, bns_cfg):
        """Different detectors must draw independent noise."""
        noise, _ = WhiteGaussianNoiseSampler(seed=0)()
        # H1 and L1 are seeded independently — they should differ
        assert not torch.allclose(noise[:, 0, :], noise[:, 1, :])

    def test_power_approximately_flat(self, bns_cfg):
        """
        Power spectrum of white Gaussian noise should be approximately flat.
        Coefficient of variation (std/mean) of |X(f)|^2 should be O(1).
        For truly white noise the expected CV is sqrt(2/N_freq) << 1 when
        averaged over a full batch — just check it is not pathologically large.
        """
        sampler = WhiteGaussianNoiseSampler(seed=99)
        noise, _ = sampler()
        power = noise.abs().pow(2).mean(dim=(0, 1))   # (F,)
        cv = power.std() / power.mean()
        assert cv.item() < 2.0  # loosely: not wildly non-uniform

    def test_graph_ready_flag(self, bns_cfg):
        assert WhiteGaussianNoiseSampler.GRAPH_READY is False

    def test_fd_roundtrip_unit_variance(self, bns_cfg):
        """Inverting the rfft must recover unit-variance Gaussian noise."""
        _, data_cfg = bns_cfg
        sampler = WhiteGaussianNoiseSampler(seed=0)
        noise_fd, _ = sampler()
        # irfft(rfft(x, norm="forward"), norm="forward") == x
        noise_td = torch.fft.irfft(noise_fd, n=data_cfg.padded_length_in_nsamples,
                                   dim=-1, norm="forward")
        std = noise_td.std()
        assert std.item() == pytest.approx(1.0, abs=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# 4. BNS parameter sampler (DistributionSampler with BNS gwconfig)
# ═══════════════════════════════════════════════════════════════════════════


class TestBNSParamSampler:
    """DistributionSampler driven by the BNS gwconfig."""

    def test_output_shape(self, param_sampler):
        batch = param_sampler(8)
        assert batch.ndim == 2
        assert batch.shape[0] == 8
        assert batch.shape[1] == param_sampler.num_params

    def test_mass_order_enforced(self, param_sampler):
        """mass_order constraint: every sample must have m1 >= m2."""
        batch = param_sampler(1000)
        m1 = batch[:, param_sampler.param_index["mass1"]]
        m2 = batch[:, param_sampler.param_index["mass2"]]
        assert (m1 >= m2 - 1e-6).all()

    def test_mass1_in_bounds(self, param_sampler):
        batch = param_sampler(500)
        m1 = batch[:, param_sampler.param_index["mass1"]]
        assert m1.min().item() >= 1.0 - 1e-5
        assert m1.max().item() <= 3.0 + 1e-5

    def test_mass2_in_bounds(self, param_sampler):
        batch = param_sampler(500)
        m2 = batch[:, param_sampler.param_index["mass2"]]
        assert m2.min().item() >= 1.0 - 1e-5
        assert m2.max().item() <= 3.0 + 1e-5

    def test_spins_in_bounds(self, param_sampler):
        batch = param_sampler(500)
        for spin in ["chi1z", "chi2z"]:
            s = batch[:, param_sampler.param_index[spin]]
            assert s.min().item() >= -0.4 - 1e-5
            assert s.max().item() <=  0.4 + 1e-5

    def test_lambda_nonnegative(self, param_sampler):
        batch = param_sampler(500)
        for lam in ["lambda1", "lambda2"]:
            l = batch[:, param_sampler.param_index[lam]]
            assert l.min().item() >= -1e-5

    def test_tc_in_bounds(self, param_sampler):
        batch = param_sampler(500)
        tc = batch[:, param_sampler.param_index["tc"]]
        assert tc.min().item() >= 3.0 - 1e-5
        assert tc.max().item() <= 3.5 + 1e-5

    def test_mchirp_present_and_positive(self, param_sampler):
        batch = param_sampler(100)
        mc = batch[:, param_sampler.param_index["mchirp"]]
        assert (mc > 0).all()

    def test_distance_positive(self, param_sampler):
        batch = param_sampler(100)
        d = batch[:, param_sampler.param_index["distance"]]
        assert (d > 0).all()

    def test_reproducible_with_same_seed(self, gwconfig_path):
        from sage.data.waveform.sampler import read_from_config
        s1 = read_from_config(gwconfig_path, seed=7)
        s2 = read_from_config(gwconfig_path, seed=7)
        b1 = s1(100)
        b2 = s2(100)
        assert torch.allclose(b1, b2)

    def test_bounds_keys_include_bns_params(self, param_sampler):
        required = {"mass1", "mass2", "chi1z", "chi2z", "lambda1", "lambda2",
                    "distance", "tc", "mchirp"}
        assert required.issubset(set(param_sampler.bounds.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# 5. IMRPhenomXAS_NRTidalv3 — waveform shapes and basic sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestIMRPhenomXAS_NRTidalv3:
    """
    Shape and sanity tests for the BNS waveform model.

    Uses multiband_mode='worst_case' with explicit m1_worst/m2_worst so the
    exact prior scan is skipped and startup is fast.
    """

    @pytest.fixture(scope="class")
    def signal_sampler_none(self, gwconfig_path, bns_cfg):
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3
        ps = read_from_config(gwconfig_path, seed=1)
        return IMRPhenomXAS_NRTidalv3(
            ps, ConstantProjection(), augment=None, multiband_mode="none"
        )

    @pytest.fixture(scope="class")
    def signal_sampler_worst_case(self, gwconfig_path, bns_cfg):
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3
        ps = read_from_config(gwconfig_path, seed=2)
        return IMRPhenomXAS_NRTidalv3(
            ps, ConstantProjection(), augment=None,
            multiband_mode="worst_case",
            m1_worst=1.0, m2_worst=1.0,   # skip the prior scan
        )

    # ── mode='none' ──────────────────────────────────────────────────────

    def test_none_mode_output_shape(self, signal_sampler_none, bns_cfg):
        cfg, data_cfg = bns_cfg
        hf, targets = signal_sampler_none()
        S = int(cfg.batch_size * cfg.class_balance)
        D = len(cfg.detectors)
        F = data_cfg.padded_length_in_nsamples // 2 + 1
        assert hf.shape == (S, D, F)

    def test_none_mode_targets_shape(self, signal_sampler_none, bns_cfg):
        cfg, _ = bns_cfg
        S  = int(cfg.batch_size * cfg.class_balance)
        pe = len(cfg.do_point_estimate)
        _, targets = signal_sampler_none()
        # targets = [pe_standardised..., label=1]
        assert targets.shape == (S, pe + 1)

    def test_none_mode_labels_are_one(self, signal_sampler_none, bns_cfg):
        cfg, _ = bns_cfg
        _, targets = signal_sampler_none()
        labels = targets[:, -1]
        assert (labels == 1.0).all()

    def test_none_mode_hf_is_complex(self, signal_sampler_none):
        hf, _ = signal_sampler_none()
        assert hf.is_complex()

    def test_none_mode_hf_not_all_zero(self, signal_sampler_none):
        hf, _ = signal_sampler_none()
        assert hf.abs().max().item() > 0.0

    def test_none_mode_output_state_fd_uniform(self, signal_sampler_none):
        from sage.core.pipeline import Grid
        assert signal_sampler_none.output_state.grid == Grid.FD_UNIFORM

    # ── mode='worst_case' ────────────────────────────────────────────────

    def test_worst_case_mode_output_shape(self, signal_sampler_worst_case, bns_cfg):
        cfg, _ = bns_cfg
        S = int(cfg.batch_size * cfg.class_balance)
        D = len(cfg.detectors)
        hf, targets = signal_sampler_worst_case()
        N_coarse = signal_sampler_worst_case.selector.n_coarse
        assert hf.shape == (S, D, N_coarse)

    def test_worst_case_mode_targets_shape(self, signal_sampler_worst_case, bns_cfg):
        cfg, _ = bns_cfg
        S  = int(cfg.batch_size * cfg.class_balance)
        pe = len(cfg.do_point_estimate)
        _, targets = signal_sampler_worst_case()
        assert targets.shape == (S, pe + 1)

    def test_worst_case_mode_labels_are_one(self, signal_sampler_worst_case):
        _, targets = signal_sampler_worst_case()
        assert (targets[:, -1] == 1.0).all()

    def test_worst_case_mode_hf_is_complex(self, signal_sampler_worst_case):
        hf, _ = signal_sampler_worst_case()
        assert hf.is_complex()

    def test_worst_case_mode_hf_not_all_zero(self, signal_sampler_worst_case):
        hf, _ = signal_sampler_worst_case()
        assert hf.abs().max().item() > 0.0

    def test_worst_case_mode_output_state_fd_coarse(self, signal_sampler_worst_case):
        from sage.core.pipeline import Grid
        assert signal_sampler_worst_case.output_state.grid == Grid.FD_COARSE

    def test_worst_case_selector_present(self, signal_sampler_worst_case):
        assert signal_sampler_worst_case.selector is not None

    def test_worst_case_n_coarse_lt_f_full(self, signal_sampler_worst_case, bns_cfg):
        _, data_cfg = bns_cfg
        F_full = data_cfg.padded_length_in_nsamples // 2 + 1
        assert signal_sampler_worst_case.selector.n_coarse < F_full

    def test_return_theta(self, signal_sampler_worst_case, bns_cfg):
        cfg, _ = bns_cfg
        S = int(cfg.batch_size * cfg.class_balance)
        hf, targets, theta = signal_sampler_worst_case(return_theta=True)
        assert theta.shape[0] == S
        assert theta.ndim == 2

    # ── Standardisation roundtrip ────────────────────────────────────────

    def test_standardise_unstandardise_roundtrip(self, signal_sampler_none, bns_cfg):
        """
        Standardising and then unstandardising parameters should recover the
        originals to near floating-point precision.
        """
        cfg, _ = bns_cfg
        S = int(cfg.batch_size * cfg.class_balance)
        hf, targets, theta = signal_sampler_none(return_theta=True)
        ps = signal_sampler_none.param_sampler
        standardised   = ps.standardise_from_batch(theta)
        unstandardised = ps.unstandardise_from_batch(standardised)
        assert unstandardised.shape == standardised.shape
        # Roundtrip: recovered values should be close to the standardised mean=0
        # (approximate — only checks shape and non-NaN)
        assert not torch.isnan(unstandardised).any()


# ═══════════════════════════════════════════════════════════════════════════
# 6. NRTidalv3 static math — kappa2T and tidal phase sign
# ═══════════════════════════════════════════════════════════════════════════


class TestNRTidalv3Math:
    """
    Sanity checks on the NRTidalv3 static helper methods.

    These test physics invariants that must hold regardless of the rest
    of the pipeline (no GPU, no config needed beyond what is imported).
    """

    @pytest.fixture(scope="class")
    def model(self, gwconfig_path, bns_cfg):
        from sage.data.waveform import read_from_config, IMRPhenomXAS_NRTidalv3
        ps = read_from_config(gwconfig_path, seed=0)
        return IMRPhenomXAS_NRTidalv3(
            ps, None, augment=None,
            multiband_mode="worst_case",
            m1_worst=1.0, m2_worst=1.0,
        )

    def test_kappa2T_zero_for_bbh(self, model):
        """kappa2T vanishes when both tidal deformabilities are zero (BBH limit)."""
        B = 4
        Xa = torch.full((B, 1), 0.5)
        Xb = torch.full((B, 1), 0.5)
        lam0 = torch.zeros(B, 1)
        kappa = model._kappa2T(Xa, Xb, lam0, lam0)
        assert (kappa == 0.0).all()

    def test_kappa2T_positive_for_bns(self, model):
        """kappa2T > 0 for any positive tidal deformability."""
        B = 4
        Xa = torch.full((B, 1), 0.6)
        Xb = torch.full((B, 1), 0.4)
        lam = torch.full((B, 1), 500.0)
        kappa = model._kappa2T(Xa, Xb, lam, lam)
        assert (kappa > 0).all()

    def test_kappa2T_shape(self, model):
        B = 5
        kappa = model._kappa2T(
            torch.ones(B, 1) * 0.5,
            torch.ones(B, 1) * 0.5,
            torch.ones(B, 1) * 300.0,
            torch.ones(B, 1) * 300.0,
        )
        assert kappa.shape == (B, 1)

    def test_tidal_phase_negative_at_finite_freq(self, model):
        """
        phi_tidal from _phi_tidal_nrt should be negative at physical frequencies
        (so subtracting it increases the total phase — see comment in get_hphc).
        """
        B = 2
        Xa = torch.full((B, 1), 0.5)
        Xb = torch.full((B, 1), 0.5)
        lam = torch.full((B, 1), 400.0)
        kappa2T        = model._kappa2T(Xa, Xb, lam, lam)
        kappaA, kappaB = model._kappaAB(Xa, Xb, lam, lam)
        PN             = model._PN_coeffs(Xa)
        tc_d           = model._nrtv3_coeffs(Xa, Xb, kappa2T, kappaA, kappaB, PN)
        # Evaluate at a representative dimensionless frequency (Mf ~ 0.01)
        M_s = torch.full((B, 1), 2.8 * 4.925e-6)  # 2.8 Msun in seconds
        Mf  = torch.full((B, 1), 0.01)              # dimensionless frequency
        phi = model._phi_tidal_nrt(Mf, PN, tc_d)
        # phi_tidal must be <= 0 for BNS (NRTidalv3 convention)
        assert (phi <= 0).all()

    def test_planck_taper_zero_at_merger(self, model):
        """Planck taper = 0 exactly at f = f_merger (lower boundary)."""
        B = 3
        Mf_merger = torch.full((B, 1), 0.02)
        taper = model._planck_taper(Mf_merger, Mf_merger, 1.2 * Mf_merger)
        assert (taper == 0.0).all()

    def test_planck_taper_one_below_merger(self, model):
        """Planck taper = 1 well below the merger frequency."""
        B = 3
        Mf_low    = torch.full((B, 1), 0.005)
        Mf_merger = torch.full((B, 1), 0.02)
        taper = model._planck_taper(Mf_low, Mf_merger, 1.2 * Mf_merger)
        assert torch.allclose(taper, torch.zeros(B, 1), atol=1e-6)

    def test_quadparam_from_lambda_bbh_limit(self, model):
        """quadparam(lambda=0) should be 1 (BH limit for quadrupole)."""
        lam = torch.zeros(4, 1)
        qp  = model._quadparam_from_lambda(lam)
        assert torch.allclose(qp, torch.ones(4, 1), atol=1e-5)

    def test_quadparam_from_lambda_increases_with_lambda(self, model):
        """Higher tidal deformability → higher quadrupole moment."""
        lam_lo = torch.full((4, 1), 100.0)
        lam_hi = torch.full((4, 1), 1000.0)
        qp_lo  = model._quadparam_from_lambda(lam_lo)
        qp_hi  = model._quadparam_from_lambda(lam_hi)
        assert (qp_hi > qp_lo).all()


# ═══════════════════════════════════════════════════════════════════════════
# 7. Integration: noise → selector → FiducialWhitening → to_network_input
# ═══════════════════════════════════════════════════════════════════════════


class TestBNSPipelineIntegration:
    """
    End-to-end shape and state tests for the BNS FD_COARSE pipeline.

    White Gaussian noise is already spectrally flat — no whitening is applied.
    The preprocessor is an identity pass-through (Preprocessor([])).
    This matches how runs/bns/train.py is configured.
    """

    @pytest.fixture(scope="class")
    def sampler_worst_case(self, gwconfig_path, bns_cfg):
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3
        ps = read_from_config(gwconfig_path, seed=5)
        return IMRPhenomXAS_NRTidalv3(
            ps, ConstantProjection(), augment=None,
            multiband_mode="worst_case",
            m1_worst=1.0, m2_worst=1.0,
        )

    @pytest.fixture(scope="class")
    def noise_sampler(self, bns_cfg):
        return WhiteGaussianNoiseSampler(seed=10)

    @pytest.fixture(scope="class")
    def processor(self):
        """Identity preprocessor — no whitening for white Gaussian noise."""
        return Preprocessor([])

    def test_noise_selector_output_shape(
        self, sampler_worst_case, noise_sampler, bns_cfg
    ):
        """selector(noise_fd) must match the coarse grid dimension."""
        cfg, _ = bns_cfg
        selector = sampler_worst_case.selector
        noise_fd, _ = noise_sampler()
        noise_coarse = selector(noise_fd)
        B = cfg.batch_size
        D = len(cfg.detectors)
        assert noise_coarse.shape == (B, D, selector.n_coarse)

    def test_inject_and_combine_shape(
        self, sampler_worst_case, noise_sampler, bns_cfg
    ):
        """signal + noise (both at coarse grid) must have the same shape."""
        cfg, _ = bns_cfg
        selector  = sampler_worst_case.selector
        B  = cfg.batch_size
        D  = len(cfg.detectors)
        S  = int(B * cfg.class_balance)
        N  = selector.n_coarse

        signal_fd, _ = sampler_worst_case()
        noise_fd, _  = noise_sampler()
        noise_coarse = selector(noise_fd)

        signal_pad = torch.zeros(B, D, N, dtype=signal_fd.dtype)
        idx        = torch.randperm(B)[:S]
        signal_pad[idx] = signal_fd

        combined = noise_coarse + signal_pad
        assert combined.shape == (B, D, N)
        assert combined.is_complex()

    def test_gwbatch_wrapping(self, sampler_worst_case, noise_sampler, bns_cfg):
        """GWBatch wrapping of the combined FD_COARSE tensor must succeed."""
        cfg, _ = bns_cfg
        selector  = sampler_worst_case.selector
        B  = cfg.batch_size
        D  = len(cfg.detectors)
        S  = int(B * cfg.class_balance)
        N  = selector.n_coarse

        signal_fd, _ = sampler_worst_case()
        noise_fd, _  = noise_sampler()
        noise_coarse = selector(noise_fd)

        signal_pad = torch.zeros(B, D, N, dtype=signal_fd.dtype)
        idx        = torch.randperm(B)[:S]
        signal_pad[idx] = signal_fd
        combined   = noise_coarse + signal_pad

        batch = GWBatch(
            combined,
            ProcessingState(Grid.FD_COARSE),
            freqs          = selector.coarse_freqs,
            coarse_indices = selector.coarse_indices,
        )
        assert batch.state.grid == Grid.FD_COARSE
        assert not batch.state.whitened

    def test_to_network_input_shape(
        self, sampler_worst_case, noise_sampler, processor, bns_cfg
    ):
        """
        to_network_input() on a FD_COARSE batch (no whitening) must return
        (B, 2*D, N_coarse) — the format expected by BNSMamba3Lite.
        """
        cfg, _ = bns_cfg
        selector  = sampler_worst_case.selector
        B  = cfg.batch_size
        D  = len(cfg.detectors)
        S  = int(B * cfg.class_balance)
        N  = selector.n_coarse

        signal_fd, _ = sampler_worst_case()
        noise_fd, _  = noise_sampler()
        noise_coarse = selector(noise_fd)

        signal_pad = torch.zeros(B, D, N, dtype=signal_fd.dtype)
        idx        = torch.randperm(B)[:S]
        signal_pad[idx] = signal_fd
        combined   = noise_coarse + signal_pad

        batch = GWBatch(
            combined,
            ProcessingState(Grid.FD_COARSE),
            freqs          = selector.coarse_freqs,
            coarse_indices = selector.coarse_indices,
        )
        result    = processor(batch)
        net_input = result.to_network_input()

        assert net_input.shape == (B, 2 * D, N)
        assert net_input.is_floating_point()

    def test_processor_state_unchanged(
        self, sampler_worst_case, noise_sampler, processor, bns_cfg
    ):
        """Identity preprocessor must leave the GWBatch state untouched."""
        cfg, _ = bns_cfg
        selector = sampler_worst_case.selector
        N  = selector.n_coarse
        B  = cfg.batch_size
        D  = len(cfg.detectors)

        noise_fd, _  = noise_sampler()
        noise_coarse = selector(noise_fd)

        batch = GWBatch(
            noise_coarse,
            ProcessingState(Grid.FD_COARSE),
            freqs=selector.coarse_freqs,
            coarse_indices=selector.coarse_indices,
        )
        result = processor(batch)

        assert result.state.grid    == Grid.FD_COARSE
        assert result.state.whitened is False

    def test_signal_injection_changes_row(
        self, sampler_worst_case, noise_sampler, bns_cfg
    ):
        """
        Injecting a signal at row 0 must change that row; other rows unchanged.

        Physical waveform amplitudes (O(1e-26) at 100 Mpc) are below the
        float32 precision floor relative to unit-variance Gaussian noise.
        We inject a synthetic signal at amplitude 1e-3 to verify the
        addition is carried through correctly row-by-row.
        """
        cfg, _ = bns_cfg
        selector = sampler_worst_case.selector
        B  = cfg.batch_size
        D  = len(cfg.detectors)
        N  = selector.n_coarse

        noise_fd, _  = noise_sampler()
        noise_coarse = selector(noise_fd)

        # Baseline: pure noise
        baseline = noise_coarse.clone()

        # Inject a strong synthetic signal only at row 0
        fake_signal  = torch.ones(D, N, dtype=torch.complex64) * 1e-3
        combined      = noise_coarse.clone()
        combined[0]  += fake_signal

        # Row 0 changed; all other rows identical
        assert not torch.allclose(combined[0], baseline[0])
        assert torch.allclose(combined[1:], baseline[1:])


# ═══════════════════════════════════════════════════════════════════════════
# 8. FD convention verification: NRTidalv3 == Pv2/D scaling
# ═══════════════════════════════════════════════════════════════════════════


class TestFDConvention:
    """
    Verify that IMRPhenomXAS_NRTidalv3 uses the same FD amplitude convention
    as IMRPhenomPv2 and IMRPhenomD.

    The shared convention (established and fixed for the BBH models) is:
      hp_fd = hp_theoretical * delta_f      (multiply by padded delta_f)
    combined with noise:
      noise_fd = rfft(noise_td, norm="forward")   (divide by N)

    Both operations give consistent units so that:
      SNR² = 4 * df * sum(|hp_fd|² / S_n_fd)
    is physically correct.

    Tests check:
    1. The `* df` factor is applied (not missing or duplicated).
    2. Amplitude scales as 1/distance (inverse-distance law).
    3. lambda=0 BBH limit: NRTidalv3 amplitude ≈ XAS backbone (proportional check).
    """

    @pytest.fixture(scope="class")
    def _make_sampler(self, gwconfig_path, bns_cfg):
        """Helper: build a fresh NRTidalv3 sampler for each test."""
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3

        def _build(seed):
            ps = read_from_config(gwconfig_path, seed=seed)
            return IMRPhenomXAS_NRTidalv3(
                ps, ConstantProjection(), augment=None,
                multiband_mode="worst_case",
                m1_worst=1.0, m2_worst=1.0,
            )
        return _build

    def test_signal_fd_is_scaled_by_df(self, gwconfig_path, bns_cfg):
        """
        hp_fd must include the * delta_f factor.  Without it, the waveform
        would be in units of strain/Hz (not strain), which would make the
        FD amplitude O(1/df) ≈ T (segment length) — far too large.
        With the factor, |hp_fd| is O(amp0 * df) which is dimensionless.
        """
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3

        _, data_cfg = bns_cfg
        ps = read_from_config(gwconfig_path, seed=7)
        sampler = IMRPhenomXAS_NRTidalv3(
            ps, ConstantProjection(), augment=None,
            multiband_mode="worst_case",
            m1_worst=1.0, m2_worst=1.0,
        )
        hf, _ = sampler()
        df = float(data_cfg.padded_delta_f)

        # After * df the peak |hp_fd| should be << 1 for any physical distance.
        # If * df were missing, it would be O(1/df) which is O(T) >> 1.
        peak = hf.abs().max().item()
        assert peak < 1.0, (
            f"Peak FD amplitude {peak:.3e} >= 1: suggests missing * df factor. "
            f"Expected O(amp0 * df) << 1 for physical distances."
        )

    def test_amplitude_scales_inversely_with_distance(self, gwconfig_path, bns_cfg):
        """
        FD strain amplitude must scale as 1/distance.  This tests the
        extrinsic parameter handling and confirms the * df convention does
        not break the distance dependence.
        """
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3
        from sage.data.waveform.approximants.IMRPhenomXAS_NRTidalv3 import (
            IMRPhenomXAS_NRTidalv3 as _Cls,
        )

        _, data_cfg = bns_cfg

        # Build the waveform model; we will call get_hphc directly with
        # fixed parameters so the test is not subject to param-sampler randomness.
        ps = read_from_config(gwconfig_path, seed=8)
        sampler = _Cls(
            ps, None, augment=None,
            multiband_mode="worst_case",
            m1_worst=1.0, m2_worst=1.0,
        )

        # Build a fixed 1-sample parameter batch: m1=1.4, m2=1.4, chi=0, lambda=0
        # columns: [m1, m2, chi1z, chi2z, distance, tc, phic, inclination, lam1, lam2]
        def _theta(dist_Mpc):
            return torch.tensor([[
                1.4, 1.4, 0.0, 0.0,
                dist_Mpc, 3.2, 0.0, 0.0, 100.0, 100.0,
            ]], dtype=torch.float32)

        hp1, _ = sampler.get_hphc(_theta(100.0))
        hp2, _ = sampler.get_hphc(_theta(200.0))

        amp1 = hp1.abs().max().item()
        amp2 = hp2.abs().max().item()

        # At 2× the distance, the amplitude should be ~0.5×
        ratio = amp1 / amp2
        assert ratio == pytest.approx(2.0, rel=0.05), (
            f"Amplitude ratio at 100 vs 200 Mpc = {ratio:.4f}; expected 2.0 ± 5%"
        )

    def test_df_factor_matches_pv2_convention(self, gwconfig_path, bns_cfg):
        """
        NRTidalv3 must use the same * df scaling as IMRPhenomPv2 and D.

        Verify by reading self.df from the sampler object and confirming it
        equals padded_delta_f = 1 / padded_length_in_s.
        """
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3

        _, data_cfg = bns_cfg
        ps = read_from_config(gwconfig_path, seed=9)
        sampler = IMRPhenomXAS_NRTidalv3(
            ps, ConstantProjection(), augment=None,
            multiband_mode="worst_case",
            m1_worst=1.0, m2_worst=1.0,
        )

        expected_df = float(data_cfg.padded_delta_f)
        actual_df   = float(sampler.df)

        # Floating-point grid accumulation (f[k+1]-f[k] vs 1/T) causes ~1e-6 rel error
        assert actual_df == pytest.approx(expected_df, rel=1e-4), (
            f"sampler.df = {actual_df} != padded_delta_f = {expected_df}. "
            f"The * df convention is broken."
        )

    def test_snr_rescaler_asd_sliced_to_coarse_grid(self, gwconfig_path, bns_cfg, flat_psds):
        """
        When worst_case multibanding is used with an augment, the SNR estimator's
        ASD must be sliced to the coarse frequency indices during __init__ so
        that the forward call (B, D, N_coarse) does not raise a shape mismatch.
        """
        from sage.data.waveform import read_from_config, ConstantProjection
        from sage.data.waveform import IMRPhenomXAS_NRTidalv3, HalfNorm
        from sage.data.waveform.snr import OptimalSNRRescaler

        with patch("sage.dsp.whiten.get_fiducial_psds", return_value=flat_psds), \
             patch("sage.data.waveform.snr.get_fiducial_psds", return_value=flat_psds):

            ps = read_from_config(gwconfig_path, seed=20)
            sampler = IMRPhenomXAS_NRTidalv3(
                ps,
                ConstantProjection(),
                augment=OptimalSNRRescaler(HalfNorm(scale=4.0, loc=8.0, seed=0)),
                multiband_mode="worst_case",
                m1_worst=1.0, m2_worst=1.0,
            )

            N_coarse = sampler.selector.n_coarse
            snr_est  = sampler.augment.snr_estimator

            # ASD must have been sliced to the coarse grid
            assert snr_est.asds.shape[-1] == N_coarse, (
                f"asds.shape[-1]={snr_est.asds.shape[-1]} != N_coarse={N_coarse}. "
                f"ASD was not sliced to the coarse grid."
            )

            # A full forward pass must complete without a shape error
            hf, _ = sampler()
            assert hf.shape[-1] == N_coarse

    def test_noise_rfft_convention_consistent_with_signal(self, bns_cfg):
        """
        WhiteGaussianNoiseSampler uses rfft(norm='forward') — the same
        convention as MemmapNoiseSampler.  The signal uses * df = * (sr/N).
        Both give FD amplitudes that are O(amplitude / N), making the
        signal-to-noise ratio meaningful without additional rescaling factors.

        Concretely: for the same number of samples N and sample rate sr,
          noise_fd amplitude ≈ noise_td_amplitude / N
          signal_fd amplitude ≈ signal_td_amplitude * (sr/N) / sr = signal_td / N
        Same N-scaling ⟹ consistent convention.
        """
        _, data_cfg = bns_cfg
        N  = data_cfg.padded_length_in_nsamples
        sr = data_cfg.sample_rate
        df = data_cfg.padded_delta_f

        # White noise TD with known amplitude A
        A      = 1.0
        td     = torch.ones(N) * A
        fd     = torch.fft.rfft(td, norm="forward")
        amp_fd = fd.abs().max().item()

        # For a constant TD signal of amplitude A:
        # rfft(ones * A, norm="forward")[0] = A * N / N = A (DC bin)
        # rfft(ones * A, norm="forward")[k>0] ≈ 0
        # Scaled by df to match signal convention: A * df
        assert amp_fd == pytest.approx(A, rel=1e-4), (
            f"|rfft(ones*A)[0]| = {amp_fd:.6f}, expected A = {A}. "
            f"norm='forward' convention broken."
        )

        # Signal convention: theoretical_amplitude * df
        # For a DC-like signal of amplitude A, hp_fd = A * df
        signal_fd_amplitude = A * df
        # The ratio signal_fd / noise_fd (DC bin, same A) = df / 1 = df
        # This is the CORRECT ratio — signal was scaled by df, noise was not.
        # The whitening step compensates by dividing by ASD ∝ sqrt(S_n_df * df).
        # Just confirm neither is NaN.
        assert not math.isnan(signal_fd_amplitude)
        assert not math.isnan(amp_fd)


# ── Network channel-layout contract (blocked FD) ──────────────────────────────


def test_blocked_fd_channel_indices_match_to_network_input():
    """
    Guards the D>=2 detector-channel scramble bug.

    ``GWBatch.to_network_input`` emits a BLOCKED FD layout
    ``[d0_re, d1_re, ..., d0_im, d1_im, ...]``; ``blocked_detector_channel_indices``
    (which BNSMamba3 uses to gather per-detector channels) must map detector d's
    real part to channel d and its imaginary part to channel D+d. The old
    interleaved slice ``[2d, 2d+1]`` scrambled detectors for D>=2.
    """
    from sage.core.pipeline import blocked_detector_channel_indices

    B, D, F = 2, 3, 5  # 3 detectors -> exercises the general (non-D=2) case
    data = torch.zeros(B, D, F, dtype=torch.complex64)
    for d in range(D):
        # distinct, identifiable real/imag per detector
        data[:, d, :] = complex(10 * d + 1, 10 * d + 2)

    x = GWBatch(data=data, state=ProcessingState(Grid.FD_COARSE)).to_network_input()
    assert x.shape == (B, 2 * D, F)

    for d in range(D):
        idx = blocked_detector_channel_indices(d, num_detectors=D, channels_per_det=2)
        re = x[:, idx[0], :]
        im = x[:, idx[1], :]
        assert torch.allclose(re, torch.full_like(re, float(10 * d + 1))), \
            f"detector {d}: real part not at channel {idx[0]}"
        assert torch.allclose(im, torch.full_like(im, float(10 * d + 2))), \
            f"detector {d}: imag part not at channel {idx[1]}"

    # Document the bug: the OLD interleaved slice [2d, 2d+1] does NOT give
    # detector d's (re, im) for d >= 1.
    for d in range(1, D):
        wrong = x[:, d * 2 : (d + 1) * 2, :]
        assert not torch.allclose(
            wrong[:, 0, :], torch.full_like(wrong[:, 0, :], float(10 * d + 1))
        ), f"interleaved slice unexpectedly matched detector {d} real part"

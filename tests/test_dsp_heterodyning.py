"""Unit tests for sage.dsp.heterodyning."""

import math
import pytest
import numpy as np
import torch

from sage.dsp.heterodyning import apply_heterodyne, residual_chirp_time


# ---------------------------------------------------------------------------
# apply_heterodyne — numpy complex arrays
# ---------------------------------------------------------------------------

class TestApplyHeterodyneNumpy:
    def test_zero_phase_is_identity(self):
        h = np.array([1 + 2j, 3 - 4j, -5 + 0j], dtype=np.complex128)
        phase = np.zeros(3)
        result = apply_heterodyne(h, phase)
        np.testing.assert_allclose(result, h, atol=1e-14)

    def test_pi_phase_negates(self):
        # exp(-i*π) = -1 → h_het = -h
        h = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128)
        phase = np.full(3, math.pi)
        result = apply_heterodyne(h, phase)
        np.testing.assert_allclose(result, -h, atol=1e-14)

    def test_two_pi_phase_is_identity(self):
        h = np.array([2 + 3j, -1 + 0.5j], dtype=np.complex128)
        phase = np.full(2, 2 * math.pi)
        result = apply_heterodyne(h, phase)
        np.testing.assert_allclose(result, h, atol=1e-14)

    def test_real_array_raises(self):
        h = np.array([1.0, 2.0, 3.0])  # real, not complex
        phase = np.zeros(3)
        with pytest.raises(TypeError, match="complex"):
            apply_heterodyne(h, phase)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            apply_heterodyne([1 + 2j, 3 + 4j], np.zeros(2))

    def test_output_same_shape_as_input(self):
        N = 64
        h = np.random.randn(N) + 1j * np.random.randn(N)
        phase = np.random.randn(N)
        result = apply_heterodyne(h, phase)
        assert result.shape == h.shape


# ---------------------------------------------------------------------------
# apply_heterodyne — torch complex tensors
# ---------------------------------------------------------------------------

class TestApplyHeterodyneTorchComplex:
    def test_zero_phase_is_identity(self):
        h = torch.tensor([1 + 2j, 3 - 4j, -5 + 0j], dtype=torch.complex64)
        phase = torch.zeros(3)
        result = apply_heterodyne(h, phase)
        assert torch.allclose(result, h, atol=1e-6)

    def test_pi_phase_negates(self):
        h = torch.tensor([1 + 0j, 0 + 1j], dtype=torch.complex64)
        phase = torch.full((2,), math.pi)
        result = apply_heterodyne(h, phase)
        assert torch.allclose(result, -h, atol=1e-6)

    def test_roundtrip_double_application(self):
        # Applying twice with the same phase should recover the original
        torch.manual_seed(0)
        N = 128
        h = torch.randn(N, dtype=torch.float32) + 1j * torch.randn(N, dtype=torch.float32)
        phase = torch.rand(N) * 2 * math.pi
        h_het = apply_heterodyne(h, phase)
        h_back = apply_heterodyne(h_het, -phase)
        assert torch.allclose(h_back, h, atol=1e-5)

    def test_batch_shape_preserved(self):
        B, D, N = 4, 2, 32
        h = torch.randn(B, D, N, dtype=torch.complex64)
        phase = torch.zeros(N)
        result = apply_heterodyne(h, phase)
        assert result.shape == (B, D, N)

    def test_output_is_complex(self):
        h = torch.randn(16, dtype=torch.complex64)
        phase = torch.zeros(16)
        result = apply_heterodyne(h, phase)
        assert torch.is_complex(result)


# ---------------------------------------------------------------------------
# apply_heterodyne — real/imag encoded tensors
# ---------------------------------------------------------------------------

class TestApplyHeterodyneRealImag:
    def test_shape_preserved_real_imag(self):
        # Shape: (B, 2, N) — axis -2 is [re, im]
        B, N = 3, 64
        h = torch.randn(B, 2, N)
        phase = torch.zeros(N)
        result = apply_heterodyne(h, phase)
        assert result.shape == (B, 2, N)

    def test_zero_phase_is_identity_real_imag(self):
        B, N = 2, 32
        h = torch.randn(B, 2, N)
        phase = torch.zeros(N)
        result = apply_heterodyne(h, phase)
        assert torch.allclose(result, h, atol=1e-6)

    def test_extra_channels_passed_through(self):
        # Shape (B, 4, N): channels 0,1 are re/im; channels 2,3 are extra
        B, N = 2, 32
        h = torch.randn(B, 4, N)
        phase = torch.zeros(N)
        result = apply_heterodyne(h, phase)
        assert result.shape == (B, 4, N)
        # Extra channels (index 2 onward) must be unchanged
        assert torch.equal(result[:, 2:, :], h[:, 2:, :])

    def test_pi_rotation_swaps_real_imag_sign(self):
        # exp(-i*π) = -1, so re'= re*(-1)+im*0 = -re, im'= im*(-1)-re*0 = -im
        B, N = 1, 8
        h = torch.randn(B, 2, N)
        phase = torch.full((N,), math.pi)
        result = apply_heterodyne(h, phase)
        assert torch.allclose(result[:, 0, :], -h[:, 0, :], atol=1e-6)
        assert torch.allclose(result[:, 1, :], -h[:, 1, :], atol=1e-6)


# ---------------------------------------------------------------------------
# residual_chirp_time
# ---------------------------------------------------------------------------

class TestResidualChirpTime:
    def test_output_shape(self):
        N = 100
        h = np.random.randn(N) + 1j * np.random.randn(N)
        tau = residual_chirp_time(h, duration=16.0)
        assert tau.shape == (N - 1,)

    def test_zero_input_gives_zero_output(self):
        N = 50
        h = np.zeros(N, dtype=np.complex128)
        tau = residual_chirp_time(h, duration=8.0)
        np.testing.assert_array_equal(tau, np.zeros(N - 1))

    def test_non_negative_output(self):
        N = 128
        np.random.seed(42)
        h = np.random.randn(N) + 1j * np.random.randn(N)
        tau = residual_chirp_time(h, duration=16.0)
        assert (tau >= 0).all()

    def test_constant_phase_gives_zero_chirp_time(self):
        # h(f) = A * exp(i*constant) → no phase gradient → τ = 0
        N = 64
        A = np.ones(N) * 2.0
        constant_phase = 0.3
        h = A * np.exp(1j * constant_phase)
        tau = residual_chirp_time(h, duration=4.0, valid_threshold=0.0)
        np.testing.assert_allclose(tau, np.zeros(N - 1), atol=1e-12)

    def test_valid_threshold_masks_small_amplitudes(self):
        N = 10
        h = np.zeros(N, dtype=np.complex128)
        h[3] = 1.0 + 0.5j  # only one nonzero bin
        tau = residual_chirp_time(h, duration=2.0, valid_threshold=0.0)
        # Only the transition at index 3 can be nonzero
        assert tau[2] >= 0  # adjacent bins (h[2]=0 and h[3]≠0) → masked → 0
        assert tau[3] >= 0  # adjacent bins (h[3]≠0 and h[4]=0) → masked → 0


# ---------------------------------------------------------------------------
# compute_reference_phase — skipped without pycbc
# ---------------------------------------------------------------------------

def test_compute_reference_phase_requires_pycbc():
    pycbc = pytest.importorskip("pycbc", reason="pycbc not installed")
    from sage.dsp.heterodyning import compute_reference_phase
    phase = compute_reference_phase(
        m1=30.0, m2=10.0, s1z=0.0, s2z=0.0,
        sample_rate=4096.0, duration=4.0,
        f_min=20.0, f_max=1024.0,
    )
    assert phase.shape == (4096 * 4 // 2 + 1,)

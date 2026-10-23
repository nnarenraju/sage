"""Unit tests for sage.core.torch — differentiable utilities.

These wrap PyTorch autograd to give JAX-style grad / value_and_grad APIs
used throughout the waveform and training code.
"""

import pytest
import torch
from sage.core.torch import (
    nudge_backward_,
    nudge_forward_,
    torch_grad,
    torch_value_and_grad,
)


# ---------------------------------------------------------------------------
# nudge_backward_ / nudge_forward_
# ---------------------------------------------------------------------------

class TestNudgeOps:
    def test_nudge_backward_pushes_at_max(self):
        x = torch.tensor([2.0, 3.0])
        nudge_backward_(x, max_limit=2.0)
        assert (x < 2.0).all()


    def test_nudge_forward_pushes_at_min(self):
        x = torch.tensor([0.0, 0.5])
        nudge_forward_(x, min_limit=1.0)
        assert (x > 1.0).all()


    def test_nudge_backward_custom_factor(self):
        x = torch.tensor([10.0])
        nudge_backward_(x, max_limit=10.0, nudge_factor=0.5)
        assert x.item() == pytest.approx(9.5)

    def test_nudge_forward_custom_factor(self):
        x = torch.tensor([0.0])
        nudge_forward_(x, min_limit=0.0, nudge_factor=0.1)
        assert x.item() == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# torch_grad
# ---------------------------------------------------------------------------

class TestTorchGrad:
    def test_gradient_of_quadratic(self):
        # d/dx (x^2) at x=3 → 6
        x = torch.tensor([3.0])
        g = torch_grad(lambda t: (t ** 2).sum(), (x,))
        assert g.item() == pytest.approx(6.0)

    def test_gradient_of_linear(self):
        # d/dx (5x) = 5 everywhere
        x = torch.rand(8)
        g = torch_grad(lambda t: (5 * t).sum(), (x,))
        assert torch.allclose(g, torch.full_like(x, 5.0))

    def test_gradient_of_scalar_output(self):
        # d/dx x^3 at x=2 → 3*4 = 12
        x = torch.tensor(2.0)
        g = torch_grad(lambda t: t ** 3, (x,))
        assert g.item() == pytest.approx(12.0)

    def test_gradient_wrt_second_arg(self):
        # d/db (a + b^2) at b=3 → 6
        a = torch.tensor(1.0)
        b = torch.tensor(3.0)
        g = torch_grad(lambda a, b: a + b ** 2, (a, b), argnums=1)
        assert g.item() == pytest.approx(6.0)

    def test_gradient_wrt_multiple_args(self):
        # d/da (a*b) = b,  d/db (a*b) = a
        a = torch.tensor(3.0)
        b = torch.tensor(4.0)
        ga, gb = torch_grad(lambda a, b: a * b, (a, b), argnums=(0, 1))
        assert ga.item() == pytest.approx(4.0)
        assert gb.item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# torch_value_and_grad
# ---------------------------------------------------------------------------

class TestTorchValueAndGrad:
    def test_value_is_correct(self):
        x = torch.tensor([2.0, 3.0])
        val, _ = torch_value_and_grad(lambda t: (t ** 2).sum(), (x,))
        assert val.item() == pytest.approx(13.0)

    def test_grad_is_correct(self):
        x = torch.tensor([2.0, 3.0])
        _, g = torch_value_and_grad(lambda t: (t ** 2).sum(), (x,))
        assert torch.allclose(g, torch.tensor([4.0, 6.0]))

    def test_tensor_input_not_tuple(self):
        x = torch.tensor(5.0)
        val, g = torch_value_and_grad(lambda t: t ** 2, x)
        assert val.item() == pytest.approx(25.0)
        assert g.item() == pytest.approx(10.0)

    def test_multi_argnums_returns_tuple_of_grads(self):
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        val, (ga, gb) = torch_value_and_grad(
            lambda a, b: a * b, (a, b), argnums=(0, 1)
        )
        assert val.item() == pytest.approx(6.0)
        assert ga.item() == pytest.approx(3.0)  # d/da (a*b) = b
        assert gb.item() == pytest.approx(2.0)  # d/db (a*b) = a

    def test_create_graph_enables_second_order(self):
        # Second derivative of x^3 at x=2: d^2/dx^2 (x^3) = 6x = 12
        x = torch.tensor(2.0, requires_grad=True)
        _, g = torch_value_and_grad(lambda t: t ** 3, (x,), create_graph=True)
        g2 = torch.autograd.grad(g, x)[0]
        assert g2.item() == pytest.approx(12.0)



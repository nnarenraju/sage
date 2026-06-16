"""
Tests for the architecture dropout + Monte-Carlo dropout utilities.

CPU-only and config-free: exercises the dropout-equipped frontend block, a
backend residual block, and the MC-dropout helpers directly.
"""

import torch
import torch.nn as nn

from sage.architecture.frontend.mscnn1d_cbam import ConvBlock
from sage.architecture.backend.resnet2d_cbam import Bottleneck
from sage.architecture.network import enable_mc_dropout, mc_predict


def test_convblock_dropout_noop_and_stochastic():
    x = torch.randn(2, 1, 4096)
    # p=0 is a no-op even in train mode
    b0 = ConvBlock(32, 64, dropout=0.0).train()
    assert torch.allclose(b0(x), b0(x))
    # p>0 is stochastic in train, deterministic in eval
    b = ConvBlock(32, 64, dropout=0.5).train()
    assert not torch.allclose(b(x), b(x))
    b.eval()
    assert torch.allclose(b(x), b(x))


def test_bottleneck_dropout_stochastic():
    blk = Bottleneck(64, 16, dropout=0.5).train()
    x = torch.randn(2, 64, 8, 8)
    assert not torch.allclose(blk(x), blk(x))
    blk.eval()
    assert torch.allclose(blk(x), blk(x))


def test_dropout_adds_no_parameters():
    # dropout layers carry no params -> state_dict keys are unchanged, so a
    # checkpoint trained at one rate loads into a model built at another.
    a = ConvBlock(32, 64, dropout=0.0)
    b = ConvBlock(32, 64, dropout=0.3)
    assert set(a.state_dict()) == set(b.state_dict())


def _toy():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3))


def test_enable_mc_dropout_modes():
    m = _toy()
    # add a BN to confirm it is NOT reactivated
    m = nn.Sequential(m[0], nn.BatchNorm1d(16), m[2], m[3])
    enable_mc_dropout(m)
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            assert mod.training is True
        if isinstance(mod, nn.BatchNorm1d):
            assert mod.training is False


def test_mc_predict_spread_and_shapes():
    m = _toy()
    x = torch.randn(5, 8)
    mean, std = mc_predict(m, x, n_samples=20)
    assert mean.shape == (5, 3) and std.shape == (5, 3)
    assert float(std.mean()) > 1e-6   # dropout -> non-zero epistemic spread


def test_mc_predict_handles_tuple_output():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = nn.Dropout(0.5)
            self.l = nn.Linear(8, 4)

        def forward(self, x):
            h = self.l(self.d(x))
            return h, h.sum(-1)

    mean, std = mc_predict(Two(), torch.randn(6, 8), n_samples=10)
    assert isinstance(mean, tuple) and len(mean) == 2
    assert mean[0].shape == (6, 4) and mean[1].shape == (6,)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL DROPOUT TESTS PASSED <<<")

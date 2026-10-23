"""
Tests for the architecture dropout + Monte-Carlo dropout utilities.

CPU-only and config-free: exercises the dropout-equipped frontend block, a
backend residual block, and the MC-dropout helpers directly.
"""

import torch
import torch.nn as nn

from sage.architecture.frontend.mscnn1d_cbam import ConvBlock
from sage.architecture.backend.resnet2d_cbam import Bottleneck
from sage.architecture.network import enable_mc_dropout


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL DROPOUT TESTS PASSED <<<")

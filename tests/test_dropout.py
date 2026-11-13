"""Tests for block dropout and the Monte-Carlo dropout helper."""

import torch
import torch.nn as nn

from sage.architecture.frontend.mscnn1d_cbam import ConvBlock
from sage.architecture.backend.resnet2d_cbam import Bottleneck
from sage.architecture.network import enable_mc_dropout


def test_bottleneck_dropout_stochastic():
    """Bottleneck dropout varies outputs in train mode, not in eval."""
    blk = Bottleneck(64, 16, dropout=0.5).train()
    x = torch.randn(2, 64, 8, 8)
    assert not torch.allclose(blk(x), blk(x))
    blk.eval()
    assert torch.allclose(blk(x), blk(x))


def test_dropout_adds_no_parameters():
    """Dropout adds no parameters, so checkpoints load at any rate."""
    a = ConvBlock(32, 64, dropout=0.0)
    b = ConvBlock(32, 64, dropout=0.3)
    assert set(a.state_dict()) == set(b.state_dict())


def _toy():
    """A small MLP with one dropout layer."""
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3))


def test_enable_mc_dropout_modes():
    """enable_mc_dropout turns on dropout but not batch norm."""
    m = _toy()
    # Batch norm must stay in eval mode.
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

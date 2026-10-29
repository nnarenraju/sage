"""Tests for sage.factory.training: the training loop."""

import torch
from torch import nn

from sage.architecture.custom_losses import BCE_PE_Sigma
from sage.core.graph import Preprocessor
from sage.core.pipeline import Grid, ProcessingState
from sage.factory.training import SageVanillaTraining


class Recorder(nn.Module):
    """Two-head network that records whether CPU autocast was on in forward."""

    def __init__(self):
        super().__init__()
        self.rank, self.estimates = nn.Linear(16, 1), nn.Linear(16, 4)
        self.autocast = []

    def forward(self, x):
        self.autocast.append(torch.is_autocast_enabled("cpu"))
        h = x.flatten(1)
        return self.rank(h), self.estimates(h)


def test_autocast_follows_device():
    g = torch.Generator().manual_seed(0)
    signals = lambda: (torch.randn(2, 2, 4, dtype=torch.complex64, generator=g),
                       torch.cat([torch.randn(2, 2, generator=g), torch.ones(2, 1)], dim=1))
    noise = lambda: (torch.randn(4, 2, 4, dtype=torch.complex64, generator=g), torch.zeros(4, 1))
    model = Recorder()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SageVanillaTraining(
        signals, noise, Preprocessor([], ProcessingState(Grid.FD_UNIFORM)), model,
        BCE_PE_Sigma(["tc", "mchirp"]), optimiser, torch.optim.lr_scheduler.StepLR(optimiser, 1),
        None, num_iterations=2, num_epochs=1, batch_size=4, signal_batch_size=2, num_estimates=2,
        placement=torch.Generator().manual_seed(1), clip_norm=100.0, device="cpu", dtype=torch.float32, autocast_dtype=torch.bfloat16,
        scheduler_mode="batch", callbacks=[],
    )
    trainer(nepoch=0)
    assert model.autocast == [True, True]
    assert torch.isfinite(trainer.loss_components).all()

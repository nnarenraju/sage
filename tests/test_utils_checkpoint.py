"""Tests for sage.utils.checkpoint: the self-contained checkpoint format."""

import h5py
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from sage.core.logger import HDF5LossLogger
from sage.utils.checkpoint import CheckpointManager, next_epoch


def run_dir(tmp_path):
    """A run directory with a minimal ``resolved/`` record."""
    resolved = tmp_path / "resolved"
    (resolved / "fiducial").mkdir(parents=True)
    for name in ("run", "prior", "derived", "versions"):
        (resolved / f"{name}.yaml").write_text(f"{name}: 1\n")
    (resolved / "fiducial" / "fiducial_H1_psd.bin").write_bytes(bytes(8))
    return tmp_path


def parts(seed):
    torch.manual_seed(seed)
    model = nn.Linear(3, 2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.1)
    schedule = torch.optim.lr_scheduler.StepLR(optimiser, 1, 0.5)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    return model, optimiser, schedule, ema


def step(model, optimiser, schedule, ema):
    model(torch.randn(4, 3)).square().mean().backward()
    optimiser.step()
    schedule.step()
    ema.update_parameters(model)


def test_checkpoint_files(tmp_path):
    directory = run_dir(tmp_path)
    model, optimiser, schedule, ema = parts(0)
    manager = CheckpointManager(directory, model, optimiser, schedule, None, ema, keep=1)
    for epoch, loss in enumerate([1.0, 0.5, 0.7]):
        step(model, optimiser, schedule, ema)
        manager.save(epoch, loss)
    checkpoints = directory / "checkpoints"
    assert sorted(p.name for p in checkpoints.glob("*.pt")) == ["best.pt", "ema.pt", "epoch_0002.pt", "latest.pt"]
    best = torch.load(checkpoints / "best.pt", weights_only=False)
    assert best["epoch"] == 1 and best["record"]["run"] == "run: 1\n"
    assert best["record"]["fiducial"] == {"fiducial_H1_psd.bin": bytes(8)}
    ema_file = torch.load(checkpoints / "ema.pt", weights_only=False)
    assert ema_file["optimiser"] is None and set(ema_file["model"]) == set(model.state_dict())
    assert next_epoch(directory) == 3


def test_resume_exact(tmp_path):
    directory = run_dir(tmp_path)
    model, optimiser, schedule, ema = parts(0)
    manager = CheckpointManager(directory, model, optimiser, schedule, None, ema, keep="all")
    step(model, optimiser, schedule, ema)
    manager.save(0, 1.0)
    saved = {"model": model.state_dict(), "ema": ema.state_dict(), "lr": schedule.get_last_lr()}
    generator = torch.get_rng_state()

    fresh = parts(1)
    resumed = CheckpointManager(directory, *fresh[:3], None, fresh[3], keep="all")
    assert resumed.load_latest("cpu") == 1
    assert all(torch.equal(fresh[0].state_dict()[k], v) for k, v in saved["model"].items())
    assert all(torch.equal(fresh[3].state_dict()[k], v) for k, v in saved["ema"].items())
    assert fresh[2].get_last_lr() == saved["lr"] and resumed.best_validation_loss == 1.0
    assert torch.equal(torch.get_rng_state(), generator)


def test_loss_components_named(tmp_path):
    path = tmp_path / "results" / "losses.h5"
    HDF5LossLogger(path, 4, ["total", "classification"]).log(torch.ones(4, 2), 0, "training")
    HDF5LossLogger(path, 4, ["total", "classification"])  # resume keeps the rows
    with h5py.File(path, "r") as file:
        loss = file["training"]["loss"]
        assert loss[0].tolist() == [1.0, 1.0]
        assert [str(name) for name in loss.attrs["components"]] == ["total", "classification"]

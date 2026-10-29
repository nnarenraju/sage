"""``sage plot`` on a small results directory written by the run's own writers."""

import numpy as np
import torch

from sage.cli import main
from sage.core.logger import HDF5LossLogger
from sage.factory.validation import save_validation

NAMES = ["distance", "inclination", "mass1", "mass2", "mchirp", "q", "tc"]
TARGETS = ["tc", "mchirp"]


def write_pass(path, group, generator, iterations=50, batch_size=8, signals=4):
    """One validation pass whose ranking statistic separates signals from noise."""
    params = torch.rand(iterations * signals, len(NAMES), generator=generator, dtype=torch.float64) + 1
    columns = [NAMES.index(target) for target in TARGETS]
    output = torch.randn(iterations * batch_size, 1 + 2 * len(TARGETS), generator=generator) - 2
    target = torch.zeros(iterations * batch_size, len(TARGETS) + 1)
    positions = []
    for k in range(iterations):
        place = torch.randperm(batch_size, generator=generator)[:signals]
        rows = k * batch_size + place.sort().values
        batch = params[k * signals:(k + 1) * signals][place.argsort()]
        output[rows, 0] += 4
        output[rows, 1:1 + len(TARGETS)] = batch[:, columns].float() + 0.01 * torch.randn(signals, 2, generator=generator)
        output[:, 1 + len(TARGETS):] = output[:, 1 + len(TARGETS):].abs()
        target[rows, :len(TARGETS)] = (batch[:, columns].float() - 1.5) / 0.3
        target[rows, -1] = 1
        positions.append(place)
    save_validation(group, output, target, params.float(), NAMES, torch.stack(positions), path)


def test_plot(tmp_path, capsys):
    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    (run / "run.yaml").write_text("extends: hl\n")
    generator = torch.Generator().manual_seed(0)
    losses = HDF5LossLogger(run / "results" / "losses.h5", 4, ["total", "classification", "tc", "mchirp"])
    for epoch, total in enumerate([0.5, 0.4, 0.35, 0.37]):
        for split in ("training", "validation"):
            losses.log(torch.tensor([[total, 0.3, 0.1, 0.05]] * (epoch + 1)), epoch, split)
        write_pass(run / "results" / "validation.h5", f"epoch_{epoch:04d}", generator)
    write_pass(run / "results" / "validate.h5", "validate_best", generator)

    assert main(["plot", str(run)]) == 0
    assert "for epoch_0002 in" in capsys.readouterr().out  # the lowest validation loss
    plots = run / "results" / "plots"
    assert (plots / "loss_curves.png").exists() and (plots / "ROC" / "roc_curve_epoch_0002.png").exists()
    assert sorted(p.name for p in (plots / "DIAGONAL" / "epoch_0002").iterdir()) == [
        "diagonal_mchirp_epoch_0002.png", "diagonal_tc_epoch_0002.png",
    ]
    assert main(["plot", str(run), "--group", "validate_best"]) == 0
    assert (plots / "ROC" / "roc_curve_validate_best.png").exists()
    assert main(["plot", str(run), "--group", "final_best"]) == 1
    assert "fix: sage plot" in capsys.readouterr().err

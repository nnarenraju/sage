"""Contract of every named network: outputs, mining tap and the search's split."""

import pytest
import torch

from sage.factory.build import MODELS, build_model
from sage.search.model.split import check_split

SMALL = {"backend_resnet_size": "18", "frontend_filters": "16"}


def network(name: str, **settings) -> torch.nn.Module:
    """A small two-detector, two-estimate network in evaluation mode."""
    torch.manual_seed(0)
    return build_model({"name": name, **SMALL, **settings}, 2, 2, "cpu", torch.float32).eval()


def strain() -> torch.Tensor:
    return torch.randn(2, 2, 1024, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_contract(name):
    model = network(name)
    x = strain()
    with torch.no_grad():
        ranking, estimates = model(x)
        model.return_frontend_embedding = True
        model(x)
    assert ranking.shape == (2, 1)
    assert estimates.shape == (2, 4 if model.uncertainty else 2)
    per_detector = model.frontend_embedding.reshape(2, 2, -1)
    assert torch.allclose(per_detector.norm(dim=-1), torch.ones(2, 2))
    if model.separable:
        check_split(model, x)
    else:
        with pytest.raises(ValueError, match="mixes detectors before its backend"):
            check_split(model, x)


def test_groupnorm_not_separable():
    with pytest.raises(ValueError, match="detector 1 depends on detector 0"):
        check_split(network("MSCNN1Datt_2DResNetCBAM", norm_type="groupnorm"), strain())

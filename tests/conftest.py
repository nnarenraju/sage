import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def float_dtype():
    return torch.float32

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : manager.py
Description   : Short description of the file

Created on 2026-01-19 16:47:21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    pipeline_config = [
        NoiseRead(),
        Choice(
            modules=[NoiseAug1(), NoiseAug2()],
            probabilities=[0.2, 0.8]
        ),
        SignalGenerate(),
        MakeNoisySignal(),
        Choice(
            modules=[GeneralAug1(), GeneralAug2(), GeneralAug3()],
            probabilities=[0.1, 0.5, 0.4]
        ),
        Whiten(),
        MultiRateSample(),
    ]

    pipeline = Sequential(pipeline_config)

    for batch in loader:
    output = pipeline(batch)

    For the config:

    config = [
        NoiseRead,
        ([NoiseAug1, NoiseAug2], [0.2, 0.8]),
        SignalGenerate,
        MakeNoisySignal,
        ([GeneralAug1, GeneralAug2, GeneralAug3], [0.1, 0.5, 0.4]),
        Whiten,
        MultiRateSample
    ]

    Incorporating full flow manager:

    flow = CodeFlowManager(cfg=cfg, data_cfg=data_cfg)
    pipeline = build_pipeline(config, flow)
    output = pipeline(batch)

"""

# Packages
import inspect
import numpy as np

from functools import wraps
from typing import List, Type, Optional, Union

# Torch
import torch
import torch.nn as nn


class ProbabilityManager:
    """Internal manager to handle probabilities of classes/options."""

    def __init__(self):
        self.probs = {}  # name -> probability

    def register(self, cls: Type, prob: float):
        name = cls.__name__
        self.probs[name] = prob

    def get_normalized_probs(self, classes: List[Type]) -> np.ndarray:
        """Return normalized probability array for a list of classes"""
        names = [cls.__name__ for cls in classes]
        probs = np.zeros(len(classes), dtype=float)
        assigned_total = 0.0
        unassigned_idx = []

        for i, name in enumerate(names):
            if name in self.probs:
                p = self.probs[name]
                probs[i] = p
                assigned_total += p
            else:
                unassigned_idx.append(i)

        remaining = max(0, 1.0 - assigned_total)
        if unassigned_idx:
            split = remaining / len(unassigned_idx)
            for i in unassigned_idx:
                probs[i] = split
        else:
            # All assigned, sum <1 -> normalize
            if assigned_total > 0 and assigned_total < 1:
                probs /= probs.sum()
        return probs

    def sample(self, classes: List[Type]) -> Type:
        probs = self.get_normalized_probs(classes)
        idx = np.random.choice(len(classes), p=probs)
        return classes[idx]


# === Flow types ===
class Sequential:
    """Calls all classes in order. Can handle single class as well."""

    def __init__(self, classes: Union[Type, List[Type]]):
        if not isinstance(classes, list):
            classes = [classes]  # wrap single class in a list
        self.classes = classes

    def execute(self, *args, **kwargs):
        results = []
        for cls in self.classes:
            obj = cls(*args, **kwargs)
            results.append(obj)
        return results


class TorchSequential(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class Choice:
    """Selects one class probabilistically. Can handle single class as well."""

    def __init__(
        self,
        classes: Union[Type, List[Type]],
        probabilities: Optional[List[float]] = None,
    ):
        if not isinstance(classes, list):
            classes = [classes]  # wrap single class in a list
        self.classes = classes
        self.prob_manager = ProbabilityManager()
        if probabilities is not None:
            if len(probabilities) != len(classes):
                raise ValueError("Length of probabilities must match classes")
            for cls, p in zip(classes, probabilities):
                self.prob_manager.register(cls, p)

    def execute(self, *args, **kwargs):
        cls = self.prob_manager.sample(self.classes)
        return cls(*args, **kwargs)


class TorchBatchChoice(nn.Module):
    def __init__(self, modules: List[nn.Module], probabilities: List[float]):
        super().__init__()

        assert len(modules) == len(probabilities)

        self.modules_list = nn.ModuleList(modules)

        probs = torch.tensor(probabilities, dtype=torch.float32)
        probs = probs / probs.sum()
        self.register_buffer("probs", probs)

    def forward(self, x, generator=None):
        """
        x: Tensor of shape [B, ...]
        """
        B = x.shape[0]
        device = x.device

        probs = self.probs.to(device)
        dist = torch.distributions.Categorical(probs)

        # one choice per sample
        choices = dist.sample((B,), generator=generator)

        output = torch.empty_like(x)

        for idx, module in enumerate(self.modules_list):
            idxs = torch.nonzero(choices == idx, as_tuple=False).squeeze(1)

            if idxs.numel() == 0:
                continue

            selected = x.index_select(0, idxs)
            processed = module(selected)

            output.index_copy_(0, idxs, processed)

        return output


class TorchChoice(nn.Module):
    def __init__(self, modules: List[nn.Module], probabilities: List[float]):
        super().__init__()

        assert len(modules) == len(probabilities)

        self.modules_list = nn.ModuleList(modules)

        probs = torch.tensor(probabilities, dtype=torch.float32)
        probs = probs / probs.sum()

        self.register_buffer("probs", probs)

    def forward(self, x):
        dist = torch.distributions.Categorical(self.probs)
        idx = dist.sample()
        return self.modules_list[idx](x)


# === Main Codeflow Manager ===


def build_pipeline(config, flow_manager):
    modules = []

    for item in config:
        if isinstance(item, tuple):
            classes, probs = item

            # inject cfg, data_cfg here
            branch_modules = [flow_manager.call(cls) for cls in classes]

            modules.append(Choice(modules=branch_modules, probabilities=probs))
        else:
            # inject here too
            modules.append(flow_manager.call(item))

    return Sequential(modules)


class SharedConfig(nn.Module):
    """
    Make a class as follows for CodeFlowManager

    class NoiseAug1(SharedConfig):
    def forward(self, x):
        return x + torch.randn_like(x) * self.cfg.noise_level

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)


class CodeFlowManager:
    def __init__(self, **global_kwargs):
        """
        Store all global flow objects.
        Example:
            CodeFlowManager(cfg=cfg, data_cfg=data_cfg)
        """
        self.global_kwargs = global_kwargs

    def call(self, target, *args, **kwargs):
        """
        Calls a function or instantiates a class.
        Automatically merges global kwargs.
        """
        merged_kwargs = {**self.global_kwargs, **kwargs}

        if isinstance(target, type):  # class
            return target(*args, **merged_kwargs)
        elif callable(target):  # function
            return target(*args, **merged_kwargs)
        else:
            raise TypeError("Target must be callable or class")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : testing.py
Description     : Short description of the file

Created on 2026-03-28 01:53:48

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import os
import torch

from tqdm import tqdm
from contextlib import nullcontext

# LOCAL
from sage.core.config import get_cfg


class SageUncompiledTesting(torch.nn.Module):
    """
    Inference runner for offline testing on a pre-recorded data segment.

    Slides a trained Sage model over a continuous data segment using a
    DataLoader-backed slicer, applies the same preprocessing used during
    training, and collects triggers above ``trigger_threshold``.  Results
    (triggers, full network outputs, and slice times) are saved to HDF5 at
    ``cfg.export_dir/testing_data.h5``.

    This is the uncompiled version — suitable for export/analysis runs where
    ``torch.compile`` is not required.  The compiled variant lives in
    :class:`~sage.factory.manager.CompiledValidationBlock`.

    Parameters
    ----------
    slicer : torch.utils.data.Dataset
        Dataset that yields ``(x, slice_times)`` pairs for each data window.
    processor : callable
        Preprocessing callable applied to each batch before the model.
    model : torch.nn.Module
        Trained Sage model.
    trigger_threshold : float
        Ranking-statistic threshold; only windows above this are saved as
        triggers.
    batch_size : int
        DataLoader batch size.
    num_workers : int
        Number of DataLoader worker processes.
    """

    def __init__(
        self,
        slicer,
        processor,
        model,
        trigger_threshold,
        batch_size,
        num_workers,
    ):
        super().__init__()

        # Shared config
        self.cfg = get_cfg()

        # Components
        self.slicer = slicer
        self.processor = processor
        self.model = model

        # Trigger config
        self.trigger_threshold = trigger_threshold

        # DataLoader config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self):

        device = self.cfg.device

        # Eval mode
        self.model.eval()

        # Storage (mirrors validation style)
        save = {}
        save["triggers"] = []
        save["network_output"] = []
        save["slice_times"] = []

        # DataLoader
        data_loader = torch.utils.data.DataLoader(
            self.slicer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
        )

        iterable = tqdm(data_loader, desc="Testing")

        max_trigger = torch.tensor(-1e9, device=device)

        with torch.inference_mode():

            for x, slice_times in iterable:

                x = x.to(device=device, dtype=self.cfg.dtype)

                # Preprocess (same as validation)
                x = self.processor(x)

                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.cfg.autocast
                    else nullcontext()
                ):
                    out = self.model(x)

                # Match validation output structure
                network_output = torch.cat([*out], dim=1)

                # Extract ranking score
                ranking = network_output[:, 0]

                # Track max trigger
                max_trigger = torch.max(max_trigger, torch.max(ranking))

                iterable.set_description(
                    f"Max Trigger = {max_trigger.detach().cpu().item():.4f}"
                )

                # Trigger mask
                trigger_mask = ranking > self.trigger_threshold

                # Vectorized trigger extraction (fast)
                if torch.any(trigger_mask):
                    times = slice_times[trigger_mask].cpu()
                    values = ranking[trigger_mask].cpu()

                    triggers_batch = torch.stack([times, values], dim=1)
                    save["triggers"].append(triggers_batch)

                # Optional: save everything (like validation)
                save["network_output"].append(network_output.cpu())
                save["slice_times"].append(slice_times.cpu())

        # ---- Final aggregation ----

        if len(save["triggers"]) == 0:
            raise ValueError("No triggers found when searching for events!")

        triggers = torch.vstack(save["triggers"])
        network_output = torch.vstack(save["network_output"])
        slice_times = torch.hstack(save["slice_times"])

        print(
            f"A total of {len(triggers)} slices exceeded threshold {self.trigger_threshold}"
        )

        print(
            "raw values of output: max = {}, min = {}".format(
                triggers[:, 1].max().item(),
                triggers[:, 1].min().item(),
            )
        )

        # Save (same philosophy as validation)
        savepath = os.path.join(self.cfg.export_dir, "testing_data.h5")

        save_testing(
            triggers=triggers,
            network_output=network_output,
            slice_times=slice_times,
            savepath=savepath,
        )

        return triggers

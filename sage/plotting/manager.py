#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : manager.py
Description     : Short description of the file

Created on 2026-03-21 17:16:10

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
import h5py
import numpy as np

# LOCAL
from sage.plotting import plot_roc_curve


class ValidationPlotManager:

    def __init__(self, validation_h5, losses_h5, export_dir=None):
        self.validation_h5 = validation_h5
        self.losses_h5 = losses_h5
        self.export_dir = export_dir

        self.validation_data = {}
        self.training_loss = None
        self.validation_loss = None

        self._load_losses()
        self._load_validation()

    # -------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------

    def _load_losses(self):
        with h5py.File(self.losses_h5, "r") as fp:
            self.training_loss = fp["training"]["loss"][:]
            self.validation_loss = fp["validation"]["loss"][:]

    def _load_validation(self):
        with h5py.File(self.validation_h5, "r") as fp:

            for epoch_key in fp.keys():

                network_output = np.vstack(fp[epoch_key]["network_output"][:])
                network_target = np.vstack(fp[epoch_key]["network_target"][:])

                ranking_stat = network_output[:, 0]
                labels = network_target[:, -1]

                self.validation_data[epoch_key] = {
                    "ranking_stat": ranking_stat,
                    "labels": labels,
                    "network_output": network_output,
                    "network_target": network_target,
                }

    # -------------------------------------------------------
    # MASTER DRIVER
    # -------------------------------------------------------

    def make_all_plots(self, save=True):

        for epoch_key, data in self.validation_data.items():

            plot_roc_curve(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
            )

            plot_prediction_raw(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
            )

            plot_prediction_probability(
                epoch=epoch_key,
                pred_prob=data["pred_prob"],
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
            )

            plot_efficiency_curves(
                epoch=epoch_key,
                source_params=data["source_params"],
                pred_stat=data["ranking_stat"],  # or pred_prob depending what you want
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
                save_name="ranking_stat",
            )

            plot_learning_parameter_prior(
                epoch=epoch_key,
                source_params=data["source_params"],
                pred_stat=data["ranking_stat"],  # or pred_prob
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
                save_name="ranking_stat",
            )

            plot_outputbin_param_distribution(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                sample_params=data["source_params"],
                export_dir=self.export_dir,
                save=save,
            )

            plot_paramfrac_detected_above_thresh(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                sample_params=data["source_params"],
                export_dir=self.export_dir,
                save=save,
            )

            plot_loss_curves(
                training_loss=self.training_loss,
                validation_loss=self.validation_loss,
                export_dir=self.export_dir,
                save=save,
                best_epoch=None,  # or compute
            )

            plot_calibration_curve(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
                nbins=20,
            )

            plot_output_vs_param_heatmap(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                source_params=data["source_params"],
                param_name="distance",  # or "mchirp", "tc", etc.
                export_dir=self.export_dir,
                save=save,
            )

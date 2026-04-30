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

from scipy.special import expit

# LOCAL
from sage.plotting import (
    plot_2d_efficiency,
    plot_2d_param_density,
    plot_calibration_curve,
    plot_confidence_vs_snr,
    plot_correlation_matrix,
    plot_cumulative_volume,
    plot_diagonal_compare,
    plot_efficiency_curves,
    plot_joint_cdfs,
    plot_learning_parameter_prior,
    plot_loss_curves,
    plot_output_gradient,
    plot_output_trajectory_over_epochs,
    plot_output_vs_param_heatmap,
    plot_output_vs_uncertainty,
    plot_outputbin_param_distribution,
    plot_param_recovery_heatmap,
    plot_paramfrac_detected_above_thresh,
    plot_perturbation_sensitivity,
    plot_prediction_probability,
    plot_prediction_raw,
    plot_roc_curve,
    plot_separation_over_epochs,
    plot_uncertainty_vs_gradient,
)

# Ordered list matching signal_params column layout (from gwconfig param_names)
_PARAM_NAMES = [
    "chirp_distance", "coa_phase", "dec", "distance", "inclination",
    "injection_time", "mass1", "mass2", "mchirp", "polarization", "q", "ra",
    "spin1_a", "spin1_azimuthal", "spin1_polar", "spin1x", "spin1y", "spin1z",
    "spin2_a", "spin2_azimuthal", "spin2_polar", "spin2x", "spin2y", "spin2z",
    "tc",
]


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

                network_output = fp[epoch_key]["network_output"][:]   # (N, 5)
                network_target = fp[epoch_key]["network_target"][:]   # (N, 3)
                signal_params_raw = fp[epoch_key]["signal_params"][:] # (S_total, 25)

                ranking_stat = network_output[:, 0]
                labels = network_target[:, -1]
                signal_mask = labels == 1.0

                # --------------------------------------------------
                # Align signal_params with signal rows in network_output
                # using saved signal_idx (batch placement indices).
                #
                # Within each iteration, theta[i] was placed at batch
                # position idx[i].  When we later filter network_output
                # for label==1 we get signals in ascending batch-position
                # order.  argsort(idx) maps generation order → sorted
                # batch-position order, giving the correct alignment.
                # --------------------------------------------------
                source_params = {}
                if "signal_idx" in fp[epoch_key]:
                    signal_idx = fp[epoch_key]["signal_idx"][:]  # (num_iter, S)
                    num_iter, S = signal_idx.shape

                    aligned = []
                    for k in range(num_iter):
                        batch_params = signal_params_raw[k * S : (k + 1) * S]
                        batch_idx = signal_idx[k]
                        aligned.append(batch_params[np.argsort(batch_idx)])
                    aligned_params = np.concatenate(aligned, axis=0)  # (S_total, 25)

                    # Embed into full-length array (NaN for noise rows)
                    full_params = np.full((len(network_output), len(_PARAM_NAMES)), np.nan)
                    full_params[signal_mask] = aligned_params

                    source_params = {
                        name: full_params[:, i]
                        for i, name in enumerate(_PARAM_NAMES)
                    }

                self.validation_data[epoch_key] = {
                    "ranking_stat": ranking_stat,
                    "pred_prob": expit(ranking_stat),
                    "labels": labels,
                    "network_output": network_output,
                    "network_target": network_target,
                    "source_params": source_params,
                }

    # -------------------------------------------------------
    # MASTER DRIVER
    # -------------------------------------------------------

    def make_all_plots(self, save=True):

        epochs = sorted(self.validation_data.keys())
        best_epoch = np.argmin(self.validation_loss[:, 0])

        # -------------------------------------------------------
        # Per-epoch plots
        # -------------------------------------------------------
        for epoch_key in epochs:
            data = self.validation_data[epoch_key]
            sp = data["source_params"]  # {} if signal_idx not saved

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

            plot_loss_curves(
                training_loss=self.training_loss,
                validation_loss=self.validation_loss,
                export_dir=self.export_dir,
                save=save,
                best_epoch=best_epoch,
            )

            plot_calibration_curve(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
                nbins=20,
            )

            plot_joint_cdfs(
                epoch=epoch_key,
                ranking_stat=data["ranking_stat"],
                labels=data["labels"],
                export_dir=self.export_dir,
                save=save,
            )

            # Source-params-dependent plots (require signal_idx saved during training)
            if sp:
                plot_efficiency_curves(
                    epoch=epoch_key,
                    source_params=sp,
                    pred_stat=data["ranking_stat"],
                    labels=data["labels"],
                    export_dir=self.export_dir,
                    save=save,
                    save_name="ranking_stat",
                )

                plot_learning_parameter_prior(
                    epoch=epoch_key,
                    source_params=sp,
                    pred_stat=data["ranking_stat"],
                    labels=data["labels"],
                    export_dir=self.export_dir,
                    save=save,
                    save_name="ranking_stat",
                )

                plot_outputbin_param_distribution(
                    epoch=epoch_key,
                    ranking_stat=data["ranking_stat"],
                    labels=data["labels"],
                    sample_params=sp,
                    export_dir=self.export_dir,
                    save=save,
                )

                plot_paramfrac_detected_above_thresh(
                    epoch=epoch_key,
                    ranking_stat=data["ranking_stat"],
                    labels=data["labels"],
                    sample_params=sp,
                    export_dir=self.export_dir,
                    save=save,
                )

                for param_name in ("distance", "mchirp", "tc"):
                    if param_name in sp:
                        plot_output_vs_param_heatmap(
                            epoch=epoch_key,
                            ranking_stat=data["ranking_stat"],
                            labels=data["labels"],
                            source_params=sp,
                            param_name=param_name,
                            export_dir=self.export_dir,
                            save=save,
                        )

                plot_correlation_matrix(
                    ranking_stat=data["ranking_stat"],
                    source_params={
                        k: sp[k] for k in ("distance", "mchirp", "tc", "inclination", "q")
                        if k in sp
                    },
                    labels=data["labels"],
                    export_dir=self.export_dir,
                    save=save,
                    epoch=epoch_key,
                )

                plot_cumulative_volume(
                    epoch=epoch_key,
                    ranking_stat=data["ranking_stat"],
                    labels=data["labels"],
                    source_params=sp,
                    distance_param="distance",
                    export_dir=self.export_dir,
                    save=save,
                )

        # -------------------------------------------------------
        # Cross-epoch plots (run once using all epochs)
        # -------------------------------------------------------
        all_stats = {ek: self.validation_data[ek]["ranking_stat"] for ek in epochs}
        all_labels = {ek: self.validation_data[ek]["labels"] for ek in epochs}

        plot_separation_over_epochs(
            all_network_outputs=all_stats,
            all_labels=all_labels,
            epochs=epochs,
            export_dir=self.export_dir,
            save=save,
        )

        # Trajectory: track a fixed set of samples over all epochs
        # Use the last epoch's labels as the shared mask
        last_labels = self.validation_data[epochs[-1]]["labels"]
        plot_output_trajectory_over_epochs(
            all_ranking_stats=[self.validation_data[ek]["ranking_stat"] for ek in epochs],
            labels=last_labels,
            epoch_list=list(range(len(epochs))),
            export_dir=self.export_dir,
            save=save,
        )

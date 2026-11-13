#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loader of a run's validation outputs and losses that draws every
validation plot.

Filename        : manager.py
Description     : Short description of the file

Created on 2026-03-21 17:16:10

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import os
import re
import h5py
import numpy as np

from scipy.special import expit

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
    plot_prediction_raw_comparison,
    plot_roc_curve,
    plot_separation_over_epochs,
    plot_uncertainty_vs_gradient,
)
from sage.plotting.loss_curves import best_validated_epoch
from sage.plotting.pp_calibration import plot_pp_calibration
from sage.plotting._params import select_params as _select_params
from sage.plotting._epochs import epoch_number
from sage.core.logger import get_logger

logger = get_logger(__name__)

# A regular pass; final checks and sage validate use other names
EPOCH_GROUP = re.compile(r"epoch_\d+")


class ValidationPlotManager:
    """
    Validation outputs and losses of a run, and the plots of them.

    Reads every group of the validation files and the losses file on
    construction; :meth:`make_all_plots` draws the plots.

    Parameters
    ----------
    validation_h5 : str or list of str
        Validation output files; their groups are read together.
    losses_h5 : str
        Epoch loss file written by
        :class:`~sage.core.logger.HDF5LossLogger`.
    export_dir : str or None
        Directory of the plots; subdirectories are made per plot type.
    params : iterable of str or None
        Source parameters of the per-parameter plots; see
        :func:`~sage.plotting._params.select_params`.
    param_pairs : iterable of (str, str) or None
        Parameter pairs of the 2D plots; None uses
        :attr:`DEFAULT_PARAM_PAIRS`.
    efficiency_threshold : float
        Ranking-statistic threshold of the 2D efficiency plots.
    min_abs_correlation : float
        Smallest ``|r|`` with the ranking statistic for a parameter to
        enter the correlation matrix; 0 keeps every parameter.
    param_names : list of str or None
        Column names of ``signal_params`` for files that do not record
        them; recorded names take precedence.

    Raises
    ------
    ValueError
        If a group is in more than one validation file, or a file
        records no column names and ``param_names`` is None.
    """

    #: Parameter pairs used for the two-dimensional views.
    DEFAULT_PARAM_PAIRS = (
        ("mchirp", "distance"),
        ("mass1", "mass2"),
        ("mchirp", "q"),
        ("spin1z", "spin2z"),
        ("mchirp", "inclination"),
        ("distance", "inclination"),
    )

    def __init__(
        self,
        validation_h5,
        losses_h5,
        export_dir=None,
        params=None,
        param_pairs=None,
        efficiency_threshold=0.5,
        min_abs_correlation=0.05,
        param_names=None,
    ):
        self.validation_h5 = [validation_h5] if isinstance(validation_h5, (str, os.PathLike)) else list(validation_h5)
        self.param_names = param_names
        self.losses_h5 = losses_h5
        self.export_dir = export_dir
        #: Source parameters to sweep over; None keeps all available.
        self.params = params
        self.param_pairs = (
            self.DEFAULT_PARAM_PAIRS if param_pairs is None else tuple(param_pairs)
        )
        self.efficiency_threshold = efficiency_threshold
        self.min_abs_correlation = min_abs_correlation
        self._warned_no_snr = False

        self.validation_data = {}
        self.training_loss = None
        self.validation_loss = None

        self._load_losses()
        self._load_validation()

    def _load_losses(self):
        """Read the training and validation losses."""
        # Unlocked, so a running training job can still write to it
        with h5py.File(self.losses_h5, "r", locking=False) as fp:
            self.training_loss = fp["training"]["loss"][:]
            self.validation_loss = fp["validation"]["loss"][:]
            # recorded since 2026-10
            components = fp["training"]["loss"].attrs.get("components")
            self.loss_components = None if components is None else [str(name) for name in components]

    def _load_validation(self):
        """Read every validation file into ``validation_data``."""
        for path in self.validation_h5:
            self._load_validation_file(path)

    def _load_validation_file(self, path):
        """
        Read each group of one validation file into ``validation_data``.

        Parameters
        ----------
        path : str
            Validation output file.

        Raises
        ------
        ValueError
            If a group is already loaded, or ``signal_params`` records
            no column names and ``param_names`` is None.
        """
        with h5py.File(path, "r", locking=False) as fp:

            for epoch_key in fp.keys():
                if epoch_key in self.validation_data:
                    raise ValueError(f"{path}: group {epoch_key} is also in another validation file")
                names = fp[epoch_key]["signal_params"].attrs.get("names", self.param_names)
                if names is None:
                    raise ValueError(f"{path}: {epoch_key}/signal_params has no recorded column names; "
                                     "pass param_names")
                names = [str(name) for name in names]

                network_output = fp[epoch_key]["network_output"][:]  # (N, 1 + 2 * P)
                network_target = fp[epoch_key]["network_target"][:]  # (N, P + 1)
                signal_params_raw = fp[epoch_key]["signal_params"][:] # (S_total, names)

                ranking_stat = network_output[:, 0]
                labels = network_target[:, -1]
                signal_mask = labels == 1.0

                # signal_params is in generation order and signal_idx
                # holds each signal's batch position; argsort per
                # iteration gives the order of the label == 1 rows.
                source_params = {}
                if "signal_idx" in fp[epoch_key]:
                    signal_idx = fp[epoch_key]["signal_idx"][:]  # (num_iter, S)
                    num_iter, S = signal_idx.shape

                    aligned = []
                    for k in range(num_iter):
                        batch_params = signal_params_raw[k * S : (k + 1) * S]
                        batch_idx = signal_idx[k]
                        aligned.append(batch_params[np.argsort(batch_idx)])
                    aligned_params = np.concatenate(aligned, axis=0)  # (S_total, names)

                    # NaN on noise rows
                    full_params = np.full((len(network_output), len(names)), np.nan)
                    full_params[signal_mask] = aligned_params

                    source_params = {
                        name: full_params[:, i]
                        for i, name in enumerate(names)
                    }

                entry = {
                    "ranking_stat": ranking_stat,
                    "pred_prob": expit(ranking_stat),
                    "labels": labels,
                    "network_output": network_output,
                    "network_target": network_target,
                    "source_params": source_params,
                }
                entry.update(
                    self._resolve_point_estimates(
                        network_output, network_target, labels, source_params
                    )
                )
                self.validation_data[epoch_key] = entry

    def _correlated_params(self, ranking_stat, source_params, labels):
        """
        The source parameters correlated with the ranking statistic.

        A parameter is kept when its ``|r|`` with the ranking statistic
        over the signals is >= ``min_abs_correlation``; parameters with
        fewer than 10 finite values or zero spread are dropped.

        Parameters
        ----------
        ranking_stat : array-like, shape ``(N,)``
            Ranking statistic of every event.
        source_params : dict[str, array-like]
            Parameter name to the value of every event.
        labels : array-like, shape ``(N,)``
            Class labels; signals are ``1``.

        Returns
        -------
        dict[str, array-like]
            The kept entries; ``source_params`` itself when
            ``min_abs_correlation`` is 0.
        """
        if not self.min_abs_correlation:
            return source_params

        sig = labels == 1.0
        stat = np.asarray(ranking_stat)[sig]
        kept = {}
        for name, values in source_params.items():
            v = np.asarray(values)[sig]
            ok = np.isfinite(v) & np.isfinite(stat)
            if ok.sum() < 10 or np.std(v[ok]) == 0:
                continue
            if abs(np.corrcoef(v[ok], stat[ok])[0, 1]) >= self.min_abs_correlation:
                kept[name] = values
        return kept

    @staticmethod
    def _resolve_point_estimates(network_output, network_target, labels, source_params):
        """
        Point estimates, sigmas and true values in physical units.

        ``network_output`` is ``[ranking, mu..., sigma...]`` in physical
        units; ``network_target`` holds standardised targets. Each
        target column is matched to the source parameter of largest
        ``|r|``, when that is >= 0.999.

        Parameters
        ----------
        network_output : numpy.ndarray, shape ``(N, 1 + 2 * P)``
            Ranking statistic, then ``P`` means and ``P`` sigmas.
        network_target : numpy.ndarray, shape ``(N, P + 1)``
            Standardised targets, then the class label.
        labels : numpy.ndarray, shape ``(N,)``
            Class labels; signals are ``1``.
        source_params : dict[str, array-like]
            Parameter name to the value of every event.

        Returns
        -------
        dict
            ``pe_names`` (list) and ``pe_pred``, ``pe_sigma``,
            ``pe_true`` (dicts by name), over signal rows only; empty
            when there are no point estimates, signals or parameters.
        """
        empty = {"pe_names": [], "pe_pred": {}, "pe_sigma": {}, "pe_true": {}}

        n_pe = (network_output.shape[1] - 1) // 2
        if n_pe < 1 or not source_params:
            return empty

        sig = labels == 1.0
        if not sig.any():
            return empty

        names, pred, sigma, true = [], {}, {}, {}
        for i in range(n_pe):
            tgt = network_target[sig, i]
            if not np.isfinite(tgt).all() or np.std(tgt) == 0:
                continue

            best_name, best_r = None, 0.0
            for pname, pvals in source_params.items():
                phys = np.asarray(pvals)[sig]
                if not np.isfinite(phys).all() or np.std(phys) == 0:
                    continue
                r = abs(np.corrcoef(tgt, phys)[0, 1])
                if r > best_r:
                    best_name, best_r = pname, r

            # The target is an affine rescaling of its parameter
            if best_name is None or best_r < 0.999:
                continue

            names.append(best_name)
            pred[best_name] = network_output[sig, 1 + i]
            sigma[best_name] = network_output[sig, 1 + n_pe + i]
            true[best_name] = np.asarray(source_params[best_name])[sig]

        return {"pe_names": names, "pe_pred": pred, "pe_sigma": sigma, "pe_true": true}

    def resolve_epochs(self, epochs=None):
        """
        Validation group names of an epoch selection.

        Parameters
        ----------
        epochs : int, str, list of int or str, or None
            Epoch numbers (``127``), group names (``"epoch_0127"``,
            ``"final_best"``), ``"best"``, or ``"all"`` (every regular
            pass); None selects the best validated epoch.

        Returns
        -------
        list of str
            Group names, regular passes first, without duplicates; empty
            when there is no regular pass.

        Raises
        ------
        KeyError
            If an epoch, or the best validated epoch, has no group.
        """
        available = sorted(key for key in self.validation_data if EPOCH_GROUP.fullmatch(key))
        if not available:
            return []
        if epochs == "all":
            return available

        by_number = {epoch_number(k): k for k in available}
        if isinstance(epochs, (str, int)):
            epochs = [epochs]

        out = []
        for e in [None] if epochs is None else epochs:
            if e is None or e == "best":
                best_idx = best_validated_epoch(self.validation_loss)
                if best_idx not in by_number:
                    raise KeyError(f"the best validated epoch, {best_idx}, has no validation group; "
                                   f"groups: {sorted(self.validation_data)}")
                out.append(by_number[best_idx])
            elif e in self.validation_data:
                out.append(e)
            elif epoch_number(e) in by_number:
                out.append(by_number[epoch_number(e)])
            else:
                raise KeyError(f"no validation group {e!r}; groups: {sorted(self.validation_data)}")
        # Regular passes first, in order; duplicates dropped
        return [k for k in [*available, *sorted(set(self.validation_data) - set(available))] if k in set(out)]

    def make_all_plots(self, save=True, epochs=None):
        """
        Draw every validation plot into ``export_dir``.

        Loss curves once, per-epoch plots for the selected epochs, and
        cross-epoch plots over every regular pass.

        Parameters
        ----------
        save : bool
            Save the plots; otherwise show them.
        epochs : int, str, list of int or str, or None
            Epochs of the per-epoch plots; see :meth:`resolve_epochs`.

        Raises
        ------
        KeyError
            If a selected epoch has no validation group.
        """
        selected = self.resolve_epochs(epochs)
        all_epochs = self.resolve_epochs("all")
        if save:
            os.makedirs(self.export_dir, exist_ok=True)
        best_epoch = best_validated_epoch(self.validation_loss)

        plot_loss_curves(
            training_loss=self.training_loss,
            validation_loss=self.validation_loss,
            export_dir=self.export_dir,
            save=save,
            best_epoch=best_epoch,
            component_names=self.loss_components,
        )

        # best.pt against ema_calibrated.pt on the final check's windows
        finals = {"final_best": "best.pt", "final_ema": "ema_calibrated.pt"}
        if all(group in self.validation_data for group in finals):
            plot_prediction_raw_comparison(
                {name: (self.validation_data[group]["ranking_stat"], self.validation_data[group]["labels"])
                 for group, name in finals.items()},
                name="final_best_vs_final_ema",
                export_dir=self.export_dir,
                save=save,
            )

        # Per-epoch plots
        for epoch_key in selected:
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

            # Plots of source parameters, which need signal_idx
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

                for param_name in _select_params(sp, self.params):
                    plot_output_vs_param_heatmap(
                        epoch=epoch_key,
                        ranking_stat=data["ranking_stat"],
                        labels=data["labels"],
                        source_params=sp,
                        param_name=param_name,
                        export_dir=self.export_dir,
                        save=save,
                    )
                    plot_output_gradient(
                        epoch=epoch_key,
                        ranking_stat=data["ranking_stat"],
                        labels=data["labels"],
                        source_params=sp,
                        param_name=param_name,
                        export_dir=self.export_dir,
                        save=save,
                    )

                # Parameters with |r| >= min_abs_correlation only
                plot_correlation_matrix(
                    ranking_stat=data["ranking_stat"],
                    source_params=self._correlated_params(
                        data["ranking_stat"], _select_params(sp, self.params),
                        data["labels"],
                    ),
                    labels=data["labels"],
                    export_dir=self.export_dir,
                    save=save,
                    epoch=epoch_key,
                )

                # Two-parameter views over the configured pairs.
                for px, py in self.param_pairs:
                    if px in sp and py in sp:
                        plot_2d_efficiency(
                            epoch=epoch_key,
                            ranking_stat=data["ranking_stat"],
                            labels=data["labels"],
                            source_params=sp,
                            param_x=px, param_y=py,
                            threshold=self.efficiency_threshold,
                            export_dir=self.export_dir,
                            save=save,
                        )
                        plot_2d_param_density(
                            epoch=epoch_key,
                            ranking_stat=data["ranking_stat"],
                            labels=data["labels"],
                            source_params=sp,
                            param_x=px, param_y=py,
                            export_dir=self.export_dir,
                            save=save,
                        )

                # SNR plots only when signal_params records an SNR
                snr_key = next((k for k in ("snr", "network_snr") if k in sp), None)
                if snr_key is None and not self._warned_no_snr:
                    logger.info(
                        "No network SNR in signal_params, so SNR-based plots are "
                        "skipped. The optimal SNR is computed during generation "
                        "but not recorded; store it in the validation output to "
                        "enable them."
                    )
                    self._warned_no_snr = True
                if snr_key is not None:
                    plot_confidence_vs_snr(
                        epoch=epoch_key,
                        ranking_stat=data["ranking_stat"],
                        labels=data["labels"],
                        network_snrs=sp[snr_key],
                        export_dir=self.export_dir,
                        save=save,
                    )

                # Point-estimate recovery, in physical units
                if data["pe_names"]:
                    plot_diagonal_compare(
                        epoch=epoch_key,
                        pred_params=data["pe_pred"],
                        true_params=data["pe_true"],
                        network_snrs=(
                            np.asarray(sp[snr_key])[data["labels"] == 1.0]
                            if snr_key is not None else None
                        ),
                        labels=data["labels"][data["labels"] == 1.0],
                        export_dir=self.export_dir,
                        save=save,
                    )
                    plot_pp_calibration(
                        mu=np.column_stack([data["pe_pred"][p] for p in data["pe_names"]]),
                        sigma=np.column_stack([data["pe_sigma"][p] for p in data["pe_names"]]),
                        y=np.column_stack([data["pe_true"][p] for p in data["pe_names"]]),
                        param_names=list(data["pe_names"]),
                        epoch=epoch_key,
                        export_dir=self.export_dir,
                        save=save,
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

        # Cross-epoch plots, over every regular pass
        all_stats = {ek: self.validation_data[ek]["ranking_stat"] for ek in all_epochs}
        all_labels = {ek: self.validation_data[ek]["labels"] for ek in all_epochs}

        plot_separation_over_epochs(
            all_network_outputs=all_stats,
            all_labels=all_labels,
            epochs=all_epochs,
            export_dir=self.export_dir,
            save=save,
        )

        last_labels = self.validation_data[all_epochs[-1]]["labels"]
        plot_output_trajectory_over_epochs(
            all_ranking_stats=[self.validation_data[ek]["ranking_stat"] for ek in all_epochs],
            labels=[self.validation_data[ek]["labels"] for ek in all_epochs],
            epoch_list=[epoch_number(ek) for ek in all_epochs],
            export_dir=self.export_dir,
            save=save,
        )

        # Point-estimate recovery across epochs, in physical units
        pe_names = self.validation_data[all_epochs[-1]]["pe_names"]
        if pe_names:
            pe_pred = {ek: self.validation_data[ek]["pe_pred"] for ek in all_epochs}
            pe_true = {ek: self.validation_data[ek]["pe_true"] for ek in all_epochs}
            for pname in pe_names:
                plot_param_recovery_heatmap(
                    all_network_outputs=pe_pred,
                    all_labels=pe_true,
                    param_name=pname,
                    epochs=all_epochs,
                    export_dir=self.export_dir,
                    save=save,
                )

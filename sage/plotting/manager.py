#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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
from sage.plotting.loss_curves import best_validated_epoch
from sage.plotting.pp_calibration import plot_pp_calibration
from sage.plotting._params import select_params as _select_params
from sage.plotting._epochs import epoch_number
from sage.core.logger import get_logger

logger = get_logger(__name__)

# Ordered list matching signal_params column layout (from gwconfig param_names)
_PARAM_NAMES = [
    "chirp_distance", "coa_phase", "dec", "distance", "inclination",
    "injection_time", "mass1", "mass2", "mchirp", "polarization", "q", "ra",
    "spin1_a", "spin1_azimuthal", "spin1_polar", "spin1x", "spin1y", "spin1z",
    "spin2_a", "spin2_azimuthal", "spin2_polar", "spin2x", "spin2y", "spin2z",
    "tc",
]


class ValidationPlotManager:
    """
    Loads saved validation results from HDF5 and dispatches all diagnostic plots.

    Reads the per-epoch validation HDF5 (network outputs, targets, signal
    parameters, signal injection indices) and the losses HDF5, then exposes
    a single :meth:`plot_all` method that generates the full suite of
    training-diagnostics plots (ROC, loss curves, efficiency, parameter
    recovery, etc.) into ``export_dir``.

    Parameters
    ----------
    validation_h5 : str
        Path to the per-epoch validation output HDF5 file.
    losses_h5 : str
        Path to the epoch loss HDF5 file produced by
        :class:`~sage.utils.checkpoint.HDF5LossLogger`.
    export_dir : str or None
        Directory to save plots.  Subdirectories are created per plot type.
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
    ):
        self.validation_h5 = validation_h5
        self.losses_h5 = losses_h5
        self.export_dir = export_dir
        #: Which source parameters to sweep over. None = all available.
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
        """Keep only parameters the ranking statistic actually correlates with.

        Showing all 25 columns makes the matrix unreadable and mostly reports
        structural redundancy: the spin components are different
        parameterisations of the same two spin vectors, and the isotropic
        angles are uncorrelated with anything by construction.

        A parameter is kept when ``|r|`` against the ranking statistic reaches
        ``self.min_abs_correlation``. Set that to ``0`` to keep everything.
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

    # -------------------------------------------------------
    # POINT ESTIMATES
    # -------------------------------------------------------

    @staticmethod
    def _resolve_point_estimates(network_output, network_target, labels, source_params):
        """Put predictions and truth for the PE parameters on the same scale.

        ``network_output`` is ``[ranking, mu..., sigma...]`` where ``mu`` and
        ``sigma`` have already been **unstandardised into physical units** by
        the validation loop. ``network_target``, however, holds the
        *standardised* regression targets the network was trained against.

        Differencing the two directly is a unit mismatch, and it silently
        produces nonsense: chirp-mass "residuals" of ~20 against a "true
        mchirp" axis running -2 to 2. Every plot that compares predictions to
        truth (diagonal, P-P calibration, recovery heatmap) hit this.

        The physical truth is already available in ``source_params``, so rather
        than hardcode which columns are which, each target column is matched to
        the source parameter it correlates with. The standardisation is affine,
        so the correct match has ``|r| = 1`` to floating-point precision and
        there is no ambiguity.

        Returns
        -------
        dict
            ``pe_names``, ``pe_pred``, ``pe_sigma``, ``pe_true`` -- all
            physical, all restricted to signal rows. Empty lists/dicts when
            there are no PE outputs or no source parameters to match against.
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

            # An affine rescaling of the same quantity; anything less is not it.
            if best_name is None or best_r < 0.999:
                continue

            names.append(best_name)
            pred[best_name] = network_output[sig, 1 + i]
            sigma[best_name] = network_output[sig, 1 + n_pe + i]
            true[best_name] = np.asarray(source_params[best_name])[sig]

        return {"pe_names": names, "pe_pred": pred, "pe_sigma": sigma, "pe_true": true}

    # -------------------------------------------------------
    # MASTER DRIVER
    # -------------------------------------------------------

    def resolve_epochs(self, epochs=None):
        """Normalise an epoch selection to keys present in the data.

        Accepts epoch numbers (``127``), full keys (``"epoch_0127"``), the
        string ``"best"``, or ``"all"``. ``None`` selects the best validated
        epoch, which is almost always the one you want to look at.
        """
        available = sorted(self.validation_data)
        if not available:
            return []
        if epochs == "all":
            return available

        by_number = {epoch_number(k): k for k in available}
        best_idx = best_validated_epoch(self.validation_loss)
        best_key = by_number.get(best_idx, available[-1])

        if epochs is None or epochs == "best":
            return [best_key]

        if isinstance(epochs, (str, int)):
            epochs = [epochs]

        out = []
        for e in epochs:
            if e == "best":
                out.append(best_key)
            elif e in self.validation_data:
                out.append(e)
            elif epoch_number(e) in by_number:
                out.append(by_number[epoch_number(e)])
            else:
                raise KeyError(
                    f"epoch {e!r} not in this run. Available: "
                    f"{[epoch_number(k) for k in available]}"
                )
        # preserve chronological order, drop duplicates
        return [k for k in available if k in set(out)]

    def make_all_plots(self, save=True, epochs=None):
        """
        Dispatch the full suite of validation diagnostic plots.

        Iterates over all saved epochs, generates per-epoch plots (ROC, loss
        curves, efficiency, parameter recovery, etc.), and produces
        cross-epoch summaries (separation trajectory, parameter evolution).
        All figures are written to ``self.export_dir``.

        Parameters
        ----------
        save : bool
            If ``True`` (default), save all plots to disk; otherwise display.
        """

        # Per-epoch plots run only for the requested epochs (default: the best
        # one). Cross-epoch plots further down deliberately ignore this and use
        # every available epoch -- a "residual over epochs" heatmap built from
        # a single epoch is a blank strip, which is exactly what happened.
        selected = self.resolve_epochs(epochs)
        all_epochs = sorted(self.validation_data.keys())
        # Ignore epochs where validation never ran: losses.h5 is pre-allocated
        # and those rows are still zero, so a plain argmin returns the first
        # unwritten row instead of the best epoch.
        best_epoch = best_validated_epoch(self.validation_loss)

        # Loss curves depend only on the loss arrays, not on the per-epoch
        # validation data -- drawing them once, rather than redundantly inside
        # the per-epoch loop below.
        plot_loss_curves(
            training_loss=self.training_loss,
            validation_loss=self.validation_loss,
            export_dir=self.export_dir,
            save=save,
            best_epoch=best_epoch,
        )

        # -------------------------------------------------------
        # Per-epoch plots
        # -------------------------------------------------------
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

                # Heatmap and gradient for every parameter, not a hardcoded
                # three: which parameter the ranking statistic leans on is
                # exactly what these are for, so pre-selecting defeats them.
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

                # Correlation matrix over all available parameters.
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

                # Confidence against the loudness proxy actually available.
                # Only a real network SNR. Chirp distance was standing in for
                # it, which put a distance on an axis labelled SNR.
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

                # Point-estimate recovery, on physical scales for both sides.
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

        # -------------------------------------------------------
        # Cross-epoch plots (run once using all epochs)
        # -------------------------------------------------------
        all_stats = {ek: self.validation_data[ek]["ranking_stat"] for ek in all_epochs}
        all_labels = {ek: self.validation_data[ek]["labels"] for ek in all_epochs}

        plot_separation_over_epochs(
            all_network_outputs=all_stats,
            all_labels=all_labels,
            epochs=all_epochs,
            export_dir=self.export_dir,
            save=save,
        )

        # Trajectory: track a fixed set of samples over all epochs
        # Use the last epoch's labels as the shared mask
        last_labels = self.validation_data[all_epochs[-1]]["labels"]
        plot_output_trajectory_over_epochs(
            all_ranking_stats=[self.validation_data[ek]["ranking_stat"] for ek in all_epochs],
            labels=[self.validation_data[ek]["labels"] for ek in all_epochs],
            epoch_list=[epoch_number(ek) for ek in all_epochs],
            export_dir=self.export_dir,
            save=save,
        )

        # Recovery of each point-estimate parameter across epochs, physical
        # scale on both sides (see _resolve_point_estimates).
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

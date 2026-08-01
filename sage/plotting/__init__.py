#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : __init__.py
Description     : Short description of the file

Created on 2026-03-21 18:13:05

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

from .calibration_curve import plot_calibration_curve
from .correllation_matrix import plot_correlation_matrix
from .cumulative_volume import plot_cumulative_volume
from .density2d import plot_2d_param_density
from .diagonal_compare import plot_diagonal_compare
from .efficiency_curves import plot_efficiency_curves
from .efficiency2d import plot_2d_efficiency
from .epoch_separation import plot_separation_over_epochs
from .gradient_uncertainty import plot_uncertainty_vs_gradient
from .joint_cdfs import plot_joint_cdfs
from .learning_parameter_prior import plot_learning_parameter_prior
from .loss_curves import plot_loss_curves
from .output_gradient import plot_output_gradient
from .output_param_heatmap import plot_output_vs_param_heatmap
from .output_trajectory import plot_output_trajectory_over_epochs
from .output_uncertainty import plot_output_vs_uncertainty
from .param_distribution import plot_outputbin_param_distribution
from .parameter_recovery import plot_param_recovery_heatmap
from .pp_calibration import plot_pp_calibration
from .paramfrac_above_thresh import plot_paramfrac_detected_above_thresh
from .perturbation_sensitivity import plot_perturbation_sensitivity
from .prediction_probability import plot_prediction_probability
from .prediction_raw import plot_prediction_raw
from .roc_curve import plot_roc_curve
from .snr_confidence import plot_confidence_vs_snr

from .manager import ValidationPlotManager

__all__ = [
    "ValidationPlotManager",
    "plot_calibration_curve",
    "plot_correlation_matrix",
    "plot_cumulative_volume",
    "plot_2d_param_density",
    "plot_diagonal_compare",
    "plot_efficiency_curves",
    "plot_2d_efficiency",
    "plot_separation_over_epochs",
    "plot_uncertainty_vs_gradient",
    "plot_joint_cdfs",
    "plot_learning_parameter_prior",
    "plot_loss_curves",
    "plot_output_gradient",
    "plot_output_vs_param_heatmap",
    "plot_output_trajectory_over_epochs",
    "plot_output_vs_uncertainty",
    "plot_outputbin_param_distribution",
    "plot_param_recovery_heatmap",
    "plot_paramfrac_detected_above_thresh",
    "plot_perturbation_sensitivity",
    "plot_prediction_probability",
    "plot_prediction_raw",
    "plot_roc_curve",
    "plot_confidence_vs_snr",
]

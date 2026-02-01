#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomD_utils.py
Description     : Short description of the file

Created on 2026-01-22 10:38:55

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
import torch

# LOCAL
from sage.core.torch import nudge_backward_


def EradRational0815_s(eta, s):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta

    return (
        (
            0.055974469826360077 * eta
            + 0.5809510763115132 * eta2
            - 0.9606726679372312 * eta3
            + 3.352411249771192 * eta4
        )
        * (
            1.0
            + (
                -0.0030302335878845507
                - 2.0066110851351073 * eta
                + 7.7050567802399215 * eta2
            )
            * s
        )
    ) / (
        1.0
        + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta2)
        * s
    )


def EradRational0815(eta, chi1, chi2):
    # This should prevents NaNs
    nudge_backward_(eta, 0.25, 1e-6)
    Seta = torch.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1 * m1
    m2s = m2 * m2
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)

    return EradRational0815_s(eta, s)


def FinalSpin0815_s(eta, S):
    eta2 = eta * eta
    eta3 = eta2 * eta
    S2 = S * S
    S3 = S2 * S
    return eta * (
        3.4641016151377544
        - 4.399247300629289 * eta
        + 9.397292189321194 * eta2
        - 13.180949901606242 * eta3
        + S
        * (
            (1.0 / eta - 0.0850917821418767 - 5.837029316602263 * eta)
            + (0.1014665242971878 - 2.0967746996832157 * eta) * S
            + (-1.3546806617824356 + 4.108962025369336 * eta) * S2
            + (-0.8676969352555539 + 2.064046835273906 * eta) * S3
        )
    )

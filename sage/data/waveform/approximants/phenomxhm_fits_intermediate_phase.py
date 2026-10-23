#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : phenomxhm_fits_intermediate_phase.py
Description   : Batched PyTorch port of the IMRPhenomXHM intermediate-region
                phase parameter-space fits.

                Mirrors the "Phase Parameter Space Fits" section of
                lalsimulation-v6.1.0 LALSimIMRPhenomXHM_intermediate.c
                (vendored at sage/data/waveform/lalsim_src/
                LALSimIMRPhenomXHM_intermediate.c, lines 2642-3252):
                IMRPhenomXHM_Inter_Phase_{21,33,32,44}_p{1..6}, i.e. all 24
                fits of the phase derivative dphi/df at the intermediate
                collocation frequencies for the (2,1), (3,3), (3,2) and
                (4,4) modes.

                Only the InterPhaseFlag == 122019 branch is ported — the
                v6.1.0 default (release 122022 changed only amplitudes;
                LALSimIMRPhenomXHM.c:2389).  Every numeric literal is
                transcribed verbatim from the C source and the C body
                structure (local variable names, evaluation order) is
                preserved.

                Each function takes a dict pWF of IMRPhenomXWaveformStruct
                members as (B,1) float64 torch tensors and returns a (B,1)
                tensor.  Keys consumed: "eta", "STotR", "chi1L", "chi2L",
                "dchi" (= chi1L - chi2L).

Created on 2026-08-04

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

References
----------
IMRPhenomXHM  : García-Quirós et al. (2020), arXiv:2001.10914
IMRPhenomXAS  : García-Quirós et al. (2020), arXiv:2001.10914
"""

import torch

# Fits of phase derivatives at collocation points for each mode

# ---------------------------------------------------------------------------
# Collocation point 1
# ---------------------------------------------------------------------------


def IMRPhenomXHM_Inter_Phase_21_p1(pWF):
    """IMRPhenomXHM_Inter_Phase_21_p1, LALSimIMRPhenomXHM_intermediate.c:2642-2666, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 4045.84 + 7.63226/eta - 1956.93*eta - 23428.1*eta2 + 369153.*eta3 - 2.28832e6*eta4 + 6.82533e6*eta5 - 7.86254e6*eta6
    eqSpin = - 347.273*S + 83.5428*S2 - 355.67*S3 + (4.44457*S + 16.5548*S2 + 13.6971*S3)/eta + eta*( - 79.761*S - 355.299*S2 + 1114.51*S3 - 1077.75*S4) + 92.6654*S4 + eta2*(- 619.837*S - 722.787*S2 + 2392.73*S3 + 2689.18*S4)
    uneqSpin = ( 918.976*pWF["chi1L"]*torch.sqrt(1.-4.*eta) - 918.976*pWF["chi2L"]*torch.sqrt(1.-4.*eta))*eta + ( 91.7679*pWF["chi1L"]*torch.sqrt(1.-4.*eta) - 91.7679*pWF["chi2L"]*torch.sqrt(1.-4.*eta))*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_33_p1(pWF):
    """IMRPhenomXHM_Inter_Phase_33_p1, LALSimIMRPhenomXHM_intermediate.c:2668-2690, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    noSpin = 4360.19 + 4.27128/eta - 8727.4*eta + 18485.9*eta2 + 371303.00000000006*eta3 - 3.22792e6*eta4 + 1.01799e7*eta5 - 1.15659e7*eta6
    eqSpin = ((11.6635 - 251.579*eta - 3255.6400000000003*eta2 + 19614.6*eta3 - 34860.2*eta4)*S + (14.8017 + 204.025*eta - 5421.92*eta2 + 36587.3*eta3 - 74299.5*eta4)*S2)/(eta)
    uneqSpin = eta*(223.65100000000004*pWF["chi1L"]*torch.sqrt(1.-4.*eta)*(3.9201300240106223 + 1.*eta) - 223.65100000000004*pWF["chi2L"]*torch.sqrt(1.-4.*eta)*(3.9201300240106223 + 1.*eta))
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_32_p1(pWF):
    """IMRPhenomXHM_Inter_Phase_32_p1, LALSimIMRPhenomXHM_intermediate.c:2692-2716, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 4414.11 + 4.21564/eta - 10687.8*eta + 58234.6*eta2 - 64068.40000000001*eta3 - 704442.*eta4 + 2.86393e6*eta5 - 3.26362e6*eta6
    eqSpin = ((6.39833 - 610.267*eta + 2095.72*eta2 - 3970.89*eta3)*S + (22.956700000000005 - 99.1551*eta + 331.593*eta2 - 794.79*eta3)*S2 + (10.4333 + 43.8812*eta - 541.261*eta2 + 294.289*eta3)*S3 + eta*(106.047 - 1569.0299999999997*eta + 4810.61*eta2)*S4)/(eta)
    uneqSpin = 132.244*torch.sqrt(1.-4.*eta)*eta*(pWF["chi1L"]*(6.227738120444028 - 1.*eta) + pWF["chi2L"]*(-6.227738120444028 + 1.*eta))
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_44_p1(pWF):
    """IMRPhenomXHM_Inter_Phase_44_p1, LALSimIMRPhenomXHM_intermediate.c:2718-2741, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 4349.66 + 4.34125/eta - 8202.33*eta + 5534.1*eta2 + 536500.*eta3 - 4.33197e6*eta4 + 1.37792e7*eta5 - 1.60802e7*eta6
    eqSpin = ((12.0704 - 528.098*eta + 1822.9100000000003*eta2 - 9349.73*eta3 + 17900.9*eta4)*S + (10.4092 + 253.334*eta - 5452.04*eta2 + 35416.6*eta3 - 71523.*eta4)*S2 + eta*(492.60300000000007 - 9508.5*eta + 57303.4*eta2 - 109418.*eta3)*S3)/(eta)
    uneqSpin = -262.143*torch.sqrt(1.-4.*eta)*eta*(pWF["chi1L"]*(-3.0782778864970646 - 1.*eta) + pWF["chi2L"]*(3.0782778864970646 + 1.*eta))
    return noSpin + eqSpin + uneqSpin


# ---------------------------------------------------------------------------
# Collocation point 2
# ---------------------------------------------------------------------------


def IMRPhenomXHM_Inter_Phase_21_p2(pWF):
    """IMRPhenomXHM_Inter_Phase_21_p2, LALSimIMRPhenomXHM_intermediate.c:2744-2768, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta2*eta
    eta4 = eta3*eta
    eta5 = eta3*eta2
    eta6 = eta4*eta2
    S2 = S**2
    S3 = S**3
    S4 = S3*S
    noSpin = 3509.09 + 0.91868/eta + 194.72*eta - 27556.2*eta2 + 369153.*eta3 - 2.28832e6*eta4 + 6.82533e6*eta5 - 7.86254e6*eta6
    eqSpin = ((0.7083999999999999 - 60.1611*eta + 131.815*eta2 - 619.837*eta3)*S + (6.104720000000001 - 59.2068*eta + 278.588*eta2 - 722.787*eta3)*S2 + (5.7791 + 117.913*eta - 1180.4*eta2 + 2392.73*eta3)*S3 + eta*(92.6654 - 1077.75*eta + 2689.18*eta2)*S4)/(eta)
    uneqSpin = -91.7679*torch.sqrt(1.-4.*eta)*eta*(pWF["chi1L"]*(-1.6012352903357276 - 1.*eta) + pWF["chi2L"]*(1.6012352903357276 + 1.*eta))
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_33_p2(pWF):
    """IMRPhenomXHM_Inter_Phase_33_p2, LALSimIMRPhenomXHM_intermediate.c:2770-2792, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    noSpin = 3797.06 + 0.786684/eta - 2397.09*eta - 25514.*eta2 + 518314.99999999994*eta3 - 3.41708e6*eta4 + 1.01799e7*eta5 - 1.15659e7*eta6
    eqSpin = ((6.7812399999999995 + 39.4668*eta - 3520.37*eta2 + 19614.6*eta3 - 34860.2*eta4)*S + (4.80384 + 293.215*eta - 5914.61*eta2 + 36587.3*eta3 - 74299.5*eta4)*S2)/(eta)
    uneqSpin = -223.65100000000004*torch.sqrt(1.-4.*eta)*eta*(pWF["chi1L"]*(-1.3095134830606614 - 1.*eta) + pWF["chi2L"]*(1.3095134830606614 + 1.*eta))
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_32_p2(pWF):
    """IMRPhenomXHM_Inter_Phase_32_p2, LALSimIMRPhenomXHM_intermediate.c:2794-2818, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3980.7 + 0.956703/eta - 6202.38*eta + 29218.1*eta2 + 24484.2*eta3 - 807629.*eta4 + 2.86393e6*eta5 - 3.26362e6*eta6
    eqSpin = ((1.92692 - 226.825*eta + 75.246*eta2 + 1291.56*eta3)*S + (15.328700000000001 - 99.1551*eta + 608.328*eta2 - 2402.94*eta3)*S2 + (10.4333 + 43.8812*eta - 541.261*eta2 + 294.289*eta3)*S3 + eta*(106.047 - 1569.0299999999997*eta + 4810.61*eta2)*S4)/(eta)
    uneqSpin = 132.244*torch.sqrt(1.-4.*eta)*eta*(pWF["chi1L"]*(2.5769789177580837 - 1.*eta) + pWF["chi2L"]*(-2.5769789177580837 + 1.*eta))
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_44_p2(pWF):
    """IMRPhenomXHM_Inter_Phase_44_p2, LALSimIMRPhenomXHM_intermediate.c:2820-2843, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 3804.19 + 0.66144/eta - 2421.77*eta - 33475.8*eta2 + 665951.*eta3 - 4.50145e6*eta4 + 1.37792e7*eta5 - 1.60802e7*eta6
    eqSpin = ((5.83038 - 172.047*eta + 926.576*eta2 - 7676.87*eta3 + 17900.9*eta4)*S + (6.17601 + 253.334*eta - 5672.02*eta2 + 35722.1*eta3 - 71523.*eta4)*S2 + eta*(492.60300000000007 - 9508.5*eta + 57303.4*eta2 - 109418.*eta3)*S3)/(eta)
    uneqSpin = -262.143*torch.sqrt(1.-4.*eta)*eta*(pWF["chi1L"]*(-1.0543062374352932 - 1.*eta) + pWF["chi2L"]*(1.0543062374352932 + 1.*eta))
    return noSpin + eqSpin + uneqSpin


# ---------------------------------------------------------------------------
# Collocation point 3
# ---------------------------------------------------------------------------


def IMRPhenomXHM_Inter_Phase_21_p3(pWF):
    """IMRPhenomXHM_Inter_Phase_21_p3, LALSimIMRPhenomXHM_intermediate.c:2846-2871, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    delta = torch.sqrt(1.-4.*eta)
    eta2 = eta**2
    eta3 = eta2*eta
    eta4 = eta2*eta2
    eta5 = eta2*eta3
    eta6 = eta3*eta3
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3241.68 + 890.016*eta - 28651.9*eta2 + 369153.*eta3 - 2.28832e6*eta4 + 6.82533e6*eta5 - 7.86254e6*eta6
    eqSpin = (-2.2484 + 187.641*eta - 619.837*eta2)*S + (3.22603 + 166.323*eta - 722.787*eta2)*S2 + (117.913 - 1094.59*eta + 2392.73*eta2)*S3 + (92.6654 - 1077.75*eta + 2689.18*eta2)*S4
    uneqSpin = 91.7679*(pWF["dchi"])*delta*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_33_p3(pWF):
    """IMRPhenomXHM_Inter_Phase_33_p3, LALSimIMRPhenomXHM_intermediate.c:2873-2895, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    noSpin = 3321.83 + 1796.03*eta - 52406.1*eta2 + 605028.*eta3 - 3.52532e6*eta4 + 1.01799e7*eta5 - 1.15659e7*eta6
    eqSpin = (223.601 - 3714.77*eta + 19614.6*eta2 - 34860.2*eta3)*S + (314.317 - 5906.46*eta + 36587.3*eta2 - 74299.5*eta3)*S2
    uneqSpin = 223.651*(pWF["dchi"])*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_32_p3(pWF):
    """IMRPhenomXHM_Inter_Phase_32_p3, LALSimIMRPhenomXHM_intermediate.c:2897-2921, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3416.57 + 2308.63*eta - 84042.9*eta2 + 1.01936e6*eta3 - 6.0644e6*eta4 + 1.76399e7*eta5 - 2.0065e7*eta6
    eqSpin = (24.6295 - 282.354*eta - 2582.55*eta2 + 12750.*eta3)*S + (433.675 - 8775.86*eta + 56407.8*eta2 - 114798.*eta3)*S2 + (559.705 - 10627.4*eta + 61581.*eta2 - 114029.*eta3)*S3 + (106.047 - 1569.03*eta + 4810.61*eta2)*S4
    uneqSpin = 63.9466*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_44_p3(pWF):
    """IMRPhenomXHM_Inter_Phase_44_p3, LALSimIMRPhenomXHM_intermediate.c:2923-2946, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 3308.97 + 2353.58*eta - 66340.1*eta2 + 777272.*eta3 - 4.64438e6*eta4 + 1.37792e7*eta5 - 1.60802e7*eta6
    eqSpin = (-21.5697 + 926.576*eta - 7989.26*eta2 + 17900.9*eta3)*S + (353.539 - 6403.24*eta + 37599.5*eta2 - 71523.*eta3)*S2 + (492.603 - 9508.5*eta + 57303.4*eta2 - 109418.*eta3)*S3
    uneqSpin = 262.143*(pWF["dchi"])*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


# ---------------------------------------------------------------------------
# Collocation point 4
# ---------------------------------------------------------------------------


def IMRPhenomXHM_Inter_Phase_21_p4(pWF):
    """IMRPhenomXHM_Inter_Phase_21_p4, LALSimIMRPhenomXHM_intermediate.c:2949-2973, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3160.88 + 974.355*eta - 28932.5*eta2 + 369780.*eta3 - 2.28832e6*eta4 + 6.82533e6*eta5 - 7.86254e6*eta6
    eqSpin = (26.3355 - 196.851*eta + 438.401*eta2)*S + (45.9957 - 256.248*eta + 117.563*eta2)*S2 + (-20.0261 + 467.057*eta - 1613.*eta2)*S3 + (-61.7446 + 577.057*eta - 1096.81*eta2)*S4
    uneqSpin = 65.3326*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_33_p4(pWF):
    """IMRPhenomXHM_Inter_Phase_33_p4, LALSimIMRPhenomXHM_intermediate.c:2975-2998, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 3239.44 - 661.15*eta + 5139.79*eta2 + 3456.2*eta3 - 248477.*eta4 + 1.17255e6*eta5 - 1.70363e6*eta6
    eqSpin = (225.859 - 4150.09*eta + 24364.*eta2 - 46537.3*eta3)*S + (35.2439 - 994.971*eta + 8953.98*eta2 - 23603.5*eta3)*S2 + (-310.489 + 5946.15*eta - 35337.1*eta2 + 67102.4*eta3)*S3
    uneqSpin = 30.484*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_32_p4(pWF):
    """IMRPhenomXHM_Inter_Phase_32_p4, LALSimIMRPhenomXHM_intermediate.c:3000-3024, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3307.49 - 476.909*eta - 5980.37*eta2 + 127610.*eta3 - 919108.*eta4 + 2.86393e6*eta5 - 3.26362e6*eta6
    eqSpin = (-5.02553 - 282.354*eta + 1291.56*eta2)*S + (-43.8823 + 740.123*eta - 2402.94*eta2)*S2 + (43.8812 - 370.362*eta + 294.289*eta2)*S3 + (106.047 - 1569.03*eta + 4810.61*eta2)*S4
    uneqSpin = -132.244*(pWF["dchi"])*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_44_p4(pWF):
    """IMRPhenomXHM_Inter_Phase_44_p4, LALSimIMRPhenomXHM_intermediate.c:3026-3049, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 3245.63 - 928.56*eta + 8463.89*eta2 - 17422.6*eta3 - 165169.*eta4 + 908279.*eta5 - 1.31138e6*eta6
    eqSpin = (32.506 - 590.293*eta + 3536.61*eta2 - 6758.52*eta3)*S + (-25.7716 + 738.141*eta - 4867.87*eta2 + 9129.45*eta3)*S2 + (-15.7439 + 620.695*eta - 4679.24*eta2 + 9582.58*eta3)*S3
    uneqSpin = 87.0832*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


# ---------------------------------------------------------------------------
# Collocation point 5
# ---------------------------------------------------------------------------


def IMRPhenomXHM_Inter_Phase_21_p5(pWF):
    """IMRPhenomXHM_Inter_Phase_21_p5, LALSimIMRPhenomXHM_intermediate.c:3052-3073, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3102.36 + 315.911*eta - 1688.26*eta2 + 3635.76*eta3
    eqSpin = (-23.0959 + 320.93*eta - 1029.76*eta2)*S + (-49.5435 + 826.816*eta - 3079.39*eta2)*S2 + (40.7054 - 365.842*eta + 1094.11*eta2)*S3 + (81.8379 - 1243.26*eta + 4689.22*eta2)*S4
    uneqSpin = 119.014*(pWF["dchi"])*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_33_p5(pWF):
    """IMRPhenomXHM_Inter_Phase_33_p5, LALSimIMRPhenomXHM_intermediate.c:3075-3098, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 3114.3 + 2143.06*eta - 49428.3*eta2 + 563997.*eta3 - 3.35991e6*eta4 + 9.99745e6*eta5 - 1.17123e7*eta6
    eqSpin = (190.051 - 3705.08*eta + 23046.2*eta2 - 46537.3*eta3)*S + (63.6615 - 1414.2*eta + 10166.1*eta2 - 23603.5*eta3)*S2 + (-257.524 + 5179.97*eta - 33001.4*eta2 + 67102.4*eta3)*S3
    uneqSpin = 54.9833*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_32_p5(pWF):
    """IMRPhenomXHM_Inter_Phase_32_p5, LALSimIMRPhenomXHM_intermediate.c:3100-3125, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    eta7 = eta**7
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3259.03 - 3967.58*eta + 111203.*eta2 - 1.81883e6*eta3 + 1.73811e7*eta4 - 9.56988e7*eta5 + 2.75056e8*eta6 - 3.15866e8*eta7
    eqSpin = (19.7509 - 1104.53*eta + 3810.18*eta2)*S + (-230.07 + 2314.51*eta - 5944.49*eta2)*S2 + (-201.633 + 2183.43*eta - 6233.99*eta2)*S3 + (106.047 - 1569.03*eta + 4810.61*eta2)*S4
    uneqSpin = 112.714*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_44_p5(pWF):
    """IMRPhenomXHM_Inter_Phase_44_p5, LALSimIMRPhenomXHM_intermediate.c:3127-3151, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    eta7 = eta**7
    S2 = S**2
    S3 = S**3
    noSpin = 3108.38 + 3722.46*eta - 119588.*eta2 + 1.92148e6*eta3 - 1.69796e7*eta4 + 8.39194e7*eta5 - 2.17143e8*eta6 + 2.2829700000000003e8*eta7
    eqSpin = (118.319 - 529.854*eta)*eta*S + (21.0314 - 240.648*eta + 516.333*eta2)*S2 + (20.3384 - 356.241*eta + 999.417*eta2)*S3
    uneqSpin = 97.1364*(pWF["dchi"])*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


# ---------------------------------------------------------------------------
# Collocation point 6
# ---------------------------------------------------------------------------


def IMRPhenomXHM_Inter_Phase_21_p6(pWF):
    """IMRPhenomXHM_Inter_Phase_21_p6, LALSimIMRPhenomXHM_intermediate.c:3154-3175, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3089.18 + 4.89194*eta + 190.008*eta2 - 255.245*eta3
    eqSpin = (2.96997 + 57.1612*eta - 432.223*eta2)*S + (-18.8929 + 630.516*eta - 2804.66*eta2)*S2 + (-24.6193 + 549.085*eta2)*S3 + (-12.8798 - 722.674*eta + 3967.43*eta2)*S4
    uneqSpin = 74.0984*(pWF["dchi"])*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_33_p6(pWF):
    """IMRPhenomXHM_Inter_Phase_33_p6, LALSimIMRPhenomXHM_intermediate.c:3177-3200, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 3111.46 + 384.121*eta - 13003.6*eta2 + 179537.*eta3 - 1.19313e6*eta4 + 3.79886e6*eta5 - 4.64858e6*eta6
    eqSpin = (182.864 - 3834.22*eta + 24532.9*eta2 - 50165.9*eta3)*S + (21.0158 - 746.957*eta + 6701.33*eta2 - 17842.3*eta3)*S2 + (-292.855 + 5886.62*eta - 37382.4*eta2 + 75501.8*eta3)*S3
    uneqSpin = 75.5162*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_32_p6(pWF):
    """IMRPhenomXHM_Inter_Phase_32_p6, LALSimIMRPhenomXHM_intermediate.c:3202-3227, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    eta7 = eta**7
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3259.03 - 3967.58*eta + 111203.*eta2 - 1.81883e6*eta3 + 1.73811e7*eta4 - 9.56988e7*eta5 + 2.75056e8*eta6 - 3.15866e8*eta7
    eqSpin = (19.7509 - 1104.53*eta + 3810.18*eta2)*S + (-230.07 + 2314.51*eta - 5944.49*eta2)*S2 + (-201.633 + 2183.43*eta - 6233.99*eta2)*S3 + (106.047 - 1569.03*eta + 4810.61*eta2)*S4
    uneqSpin = 112.714*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def IMRPhenomXHM_Inter_Phase_44_p6(pWF):
    """IMRPhenomXHM_Inter_Phase_44_p6, LALSimIMRPhenomXHM_intermediate.c:3229-3252, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    noSpin = 3096.03 + 986.752*eta - 20371.1*eta2 + 220332.*eta3 - 1.31523e6*eta4 + 4.29193e6*eta5 - 6.01179e6*eta6
    eqSpin = (-9.96292 - 118.526*eta + 2255.76*eta2 - 6758.52*eta3)*S + (-14.4869 + 370.039*eta - 3605.8*eta2 + 9129.45*eta3)*S2 + (17.0209 + 70.1931*eta - 3070.08*eta2 + 9582.58*eta3)*S3
    uneqSpin = 23.0759*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin

# End of Phase Parameter Space Fits

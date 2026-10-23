#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : phenomxhm_fits_ringdown_phase.py
Description   : Batched PyTorch port of the IMRPhenomXHM ringdown-phase
                parameter-space fits.

                Mirrors sage/data/waveform/lalsim_src/
                LALSimIMRPhenomXHM_ringdown.c (lalsimulation-v6.1.0),
                lines 1068-1351, porting the case 122019 branch of every
                function -- the v6.1.0 default ringdown phase version
                (LALSimIMRPhenomXHM_internals.c:146).

                  IMRPhenomXHM_RD_Phase_22_alpha2 / _alphaL
                      Ringdown phase coefficients for the mode-mixing-off
                      modes (2,1), (3,3), (4,4).  The per-mode rescaling
                      (wlm/fRING^2 on alpha2, 1/eta on alphaL) is applied
                      at the call site (LALSimIMRPhenomXHM_internals.c:
                      2143 and 2147), not inside the fits.
                  IMRPhenomXHM_RD_Phase_32_p1 .. _p4
                      Spheroidal (3,2) ringdown-phase collocation values.
                      The 122019 path uses nCollocPtsRDPhase = 4
                      (internals.c:282, 355) and hence RingdownPhaseFits
                      [0..3] only (internals.c:571-574, 2493-2506);
                      IMRPhenomXHM_RD_Phase_32_p5 (c:1354-1377) has no
                      122019 branch (122022-only) and is not ported.
                  IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift /
                  IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift
                      Time and phase shift of the spheroidal (3,2)
                      ringdown ansatz relative to the 22 phase.

                Each function takes a single dict pWF of (B,1) float64
                torch tensors keyed by the IMRPhenomXWaveformStruct
                member names used by the C body (eta, STotR, dchi,
                chi1L, chi2L) and returns a (B,1) float64 tensor.

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
Mode mixing   : Berti & Klein (2014), arXiv:1408.1860
"""

import torch

# Fits over parameter space for the ringdown phase quantities.

# The spin parameter S = (m1^2*chi1 + m2^2*chi2)/(m1^2 + m2^2)

# ---------------------------------------------------------------------------
# Coefficients of the mode-mixing-off ringdown ansatz -- used for (21),(33),(44)
# via the call-site rescalings in LALSimIMRPhenomXHM_internals.c:2143,2147.
# ---------------------------------------------------------------------------


def IMRPhenomXHM_RD_Phase_22_alpha2(pWF):
    """IMRPhenomXHM_RD_Phase_22_alpha2, LALSimIMRPhenomXHM_ringdown.c:1068-1087, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    delta = torch.sqrt(1 - 4 * eta)
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    noSpin = 0.2088669311744758 - 0.37138987533788487*eta + 6.510807976353186*eta2 - 31.330215053905395*eta3 + 55.45508989446867*eta4
    eqSpin = ((0.2393965714370633 + 1.6966740823756759*eta - 16.874355161681766*eta2 + 38.61300158832203*eta3)*S)/(1. - 0.633218538432246*S)
    uneqSpin = pWF["dchi"]*(0.9088578269496244*eta**2.5 + 15.619592332008951*pWF["dchi"]*eta**3.5)*delta
    total = noSpin + eqSpin + uneqSpin
    return total


def IMRPhenomXHM_RD_Phase_22_alphaL(pWF):
    """IMRPhenomXHM_RD_Phase_22_alphaL, LALSimIMRPhenomXHM_ringdown.c:1089-1109, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    delta = torch.sqrt(1. - 4.*eta)
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    S2 = S**2
    noSpin = eta*(-1.1926122248825484 + 2.5400257699690143*eta - 16.504334734464244*eta2 + 27.623649807617376*eta3)
    eqSpin = eta3*S*(35.803988443700824 + 9.700178927988006*S - 77.2346297158916*S2) + eta*S*(0.1034526554654983 - 0.21477847929548569*S - 0.06417449517826644*S2) + eta2*S*(-4.7282481007397825 + 0.8743576195364632*S + 8.170616575493503*S2) + eta4*S*(-72.50310678862684 - 39.83460092417137*S + 180.8345521274853*S2)
    uneqSpin = (-0.7428134042821221*pWF["chi1L"]*eta**3.5 + 0.7428134042821221*pWF["chi2L"]*eta**3.5 + 17.588573345324154*pWF["chi1L"]**2*eta**4.5 - 35.17714669064831*pWF["chi1L"]*pWF["chi2L"]*eta**4.5 + 17.588573345324154*pWF["chi2L"]**2*eta**4.5)*delta
    total = noSpin + eqSpin + uneqSpin
    return total


# ---------------------------------------------------------------------------
# 32 specific fits
# ---------------------------------------------------------------------------


def IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift(pWF):
    """IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift, LALSimIMRPhenomXHM_ringdown.c:1112-1149, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 11.851438981981772 + 167.95086712701223*eta - 4565.033758777737*eta2 + 61559.132976189896*eta3 - 364129.24735853914*eta4 + 739270.8814129328*eta5
    eqSpin = (9.506768471271634 + 434.31707030999445*eta - 8046.364492927503*eta2 + 26929.677144312944*eta3)*S + (-5.949655484033632 - 307.67253970367034*eta + 1334.1062451631644*eta2 + 3575.347142399199*eta3)*S2 + (3.4881615575084797 - 2244.4613237912527*eta + 24145.932943269272*eta2 - 60929.87465551446*eta3)*S3 + (15.585154698977842 - 2292.778112523392*eta + 24793.809334683185*eta2 - 65993.84497923202*eta3)*S4
    uneqSpin = 465.7904934097202*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    total = noSpin + eqSpin + uneqSpin
    return total


def IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift(pWF):
    """IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift, LALSimIMRPhenomXHM_ringdown.c:1151-1194, case 122019."""
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
    noSpin = -1.3328895897490733 - 22.209549522908667*eta + 1056.2426481245027*eta2 - 21256.376324666326*eta3 + 246313.12887984765*eta4 - 1.6312968467540336e6*eta5 + 5.614617173188322e6*eta6 - 7.612233821752137e6*eta7
    eqSpin = (S*(-1.622727240110213 + 0.9960210841611344*S - 1.1239505323267036*S2 - 1.9586085340429995*S3 + eta2*(196.7055281997748 + 135.25216875394943*S + 1086.7504825459278*S2 + 546.6246807461155*S3 - 312.1010566468068*S4) + 0.7638287749489343*S4 + eta*(-47.475568056234245 - 35.074072557604445*S - 97.16014978329918*S2 - 34.498125910065156*S3 + 24.02858084544326*S4) + eta3*(62.632493533037625 - 22.59781899512552*S - 2683.947280170815*S2 - 1493.177074873678*S3 + 805.0266029288334*S4)))/(-2.950271397057221 + 1.*S)
    uneqSpin = (torch.sqrt(1.-4.*eta)*(pWF["chi2L"]*eta**2.5*(88.56162028006072 - 30.01812659282717*S) + pWF["chi2L"]*eta2*(43.126266433486435 - 14.617728550838805*S) + pWF["chi1L"]*eta2*(-43.126266433486435 + 14.617728550838805*S) + pWF["chi1L"]*eta**2.5*(-88.56162028006072 + 30.01812659282717*S)))/(-2.950271397057221 + 1.*S)
    total = noSpin + eqSpin + uneqSpin
    return total


# collocation points
def IMRPhenomXHM_RD_Phase_32_p1(pWF):
    """IMRPhenomXHM_RD_Phase_32_p1, LALSimIMRPhenomXHM_ringdown.c:1197-1236, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    S2 = S**2
    S3 = S**3
    S4 = S**4
    S5 = S**5
    noSpin = 3169.372056189274 + 426.8372805022653*eta - 12569.748101922158*eta2 + 149846.7281073725*eta3 - 817182.2896823225*eta4 + 1.5674053633767858e6*eta5
    eqSpin = (19.23408352151287 - 1762.6573670619173*eta + 7855.316419853637*eta2 - 3785.49764771212*eta3)*S + (-42.88446003698396 + 336.8340966473415*eta - 5615.908682338113*eta2 + 20497.5021807654*eta3)*S2 + (13.918237996338371 + 10145.53174542332*eta - 91664.12621864353*eta2 + 201204.5096556517*eta3)*S3 + (-24.72321125342808 - 4901.068176970293*eta + 53893.9479532688*eta2 - 139322.02687945773*eta3)*S4 + (-61.01931672442576 - 16556.65370439302*eta + 162941.8009556697*eta2 - 384336.57477596396*eta3)*S5
    uneqSpin = pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2*(641.2473192044652 - 1600.240100295189*pWF["chi1L"]*eta + 1600.240100295189*pWF["chi2L"]*eta + 13275.623692212472*eta*S)
    total = noSpin + eqSpin + uneqSpin
    return total


# collocation points
def IMRPhenomXHM_RD_Phase_32_p2(pWF):
    """IMRPhenomXHM_RD_Phase_32_p2, LALSimIMRPhenomXHM_ringdown.c:1238-1273, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3131.0260952676376 + 206.09687819102305*eta - 2636.4344627081873*eta2 + 7475.062269742079*eta3
    eqSpin = (49.90874152040307 - 691.9815135740145*eta - 434.60154548208334*eta2 + 10514.68111669422*eta3)*S + (97.3078084654917 - 3458.2579971189534*eta + 26748.805404989867*eta2 - 56142.13736008524*eta3)*S2 + (-132.49105074500454 + 429.0787542102207*eta + 7269.262546204149*eta2 - 27654.067482558712*eta3)*S3 + (-227.8023564332453 + 5119.138772157134*eta - 34444.2579678986*eta2 + 69666.01833764123*eta3)*S4
    uneqSpin = 477.51566939885424*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    total = noSpin + eqSpin + uneqSpin
    return total


# collocation points
def IMRPhenomXHM_RD_Phase_32_p3(pWF):
    """IMRPhenomXHM_RD_Phase_32_p3, LALSimIMRPhenomXHM_ringdown.c:1275-1312, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3082.803556599222 + 76.94679795837645*eta - 586.2469821978381*eta2 + 977.6115755788503*eta3
    eqSpin = (45.08944710349874 - 807.7353772747749*eta + 1775.4343704616288*eta2 + 2472.6476419567534*eta3)*S + (95.57355060136699 - 2224.9613131172046*eta + 13821.251641893134*eta2 - 25583.314298758105*eta3)*S2 + (-144.96370424517866 + 2268.4693587493093*eta - 10971.864789147161*eta2 + 16259.911572457446*eta3)*S3 + (-227.8023564332453 + 5119.138772157134*eta - 34444.2579678986*eta2 + 69666.01833764123*eta3)*S4
    uneqSpin = 378.2359918274837*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    total = noSpin + eqSpin + uneqSpin
    return total


# collocation points
def IMRPhenomXHM_RD_Phase_32_p4(pWF):
    """IMRPhenomXHM_RD_Phase_32_p4, LALSimIMRPhenomXHM_ringdown.c:1314-1351, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = 3077.0657367004565 + 64.99844502520415*eta - 357.38692756785395*eta2
    eqSpin = (34.793450080444714 - 986.7751755509875*eta - 9490.641676924794*eta**3 + 5700.682624203565*eta2)*S + (57.38106384558743 - 1644.6690499868596*eta - 19906.416384606226*eta**3 + 11008.881935880598*eta2)*S2 + (-126.02362949830213 + 3169.3397351803583*eta + 62863.79877094988*eta**3 - 26766.730897942085*eta2)*S3 + (-169.30909412804587 + 4900.706039920717*eta + 95314.99988114933*eta**3 - 41414.05689348732*eta2)*S4
    uneqSpin = 390.5443469721231*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    total = noSpin + eqSpin + uneqSpin
    return total

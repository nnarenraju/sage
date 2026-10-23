#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : phenomxhm_fits_inspiral.py
Description   : Batched parameter-space fits for the IMRPhenomXHM inspiral
                region: amplitude collocation points, phase lambda fits and
                the analytic PN phase correction.

                Mirrors lalsimulation-v6.1.0
                LALSimIMRPhenomXHM_inspiral.c (vendored in
                sage/data/waveform/lalsim_src/):
                  IMRPhenomXHM_Insp_Amp_{21,33,32,44}_iv{1,2,3}
                                                  c:48-537, amplitude
                                                  collocation-point fits,
                                                  version 122022 branch only
                                                  (the v6.1.0 release default)
                  IMRPhenomXHM_Insp_Phase_{21,33,32,44}_lambda
                                                  c:956-1046, phase fits,
                                                  version 122019 branch only
                                                  (the v6.1.0 release default)
                  IMRPhenomXHM_Insp_Phase_LambdaPN
                                                  c:1051-1081, analytic
                                                  linear-in-f PN phase term,
                                                  all four mode cases

                Every function takes a single dict pWF whose keys are the
                IMRPhenomXWaveformStruct member names used by the C body
                (eta, delta, chiPNHat, dchi_half, STotR, dchi, chi1L, chi2L),
                each a (B, 1) float64 torch tensor, and returns a (B, 1)
                tensor.  IMRPhenomXHM_Insp_Phase_LambdaPN takes (eta, modeTag)
                with eta a (B, 1) tensor and modeTag a Python int.

                All numeric literals are transcribed verbatim from the C
                source; local variable names and evaluation order mirror the
                C bodies.

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
Fits data     : supplementary material of https://dcc.ligo.org/P2000011-v2
"""

import math

import torch

# ----------------------------------------------------------------------
# AMPLITUDE
# ----------------------------------------------------------------------
# Fits of the collocation points in the inspiral across parameter space.
# 3 for each mode: iv1, iv2, iv3
# [iv1,iv2,iv3]=[0.5*f^Ins_lm,0.75*f^Ins_lm,f^Ins_lm]
# For more details about the reconstruction procedure, see Sec. IV.A
#
# These parameter-space fits are documented in the supplementary material
# of https://dcc.ligo.org/P2000011-v2.
#
# Spin parameter S = chi_PN = chi_eff - 38/113 * eta * (chi1 + chi2)
# chi_eff = (m1*chi1 + m2*chi2)/(m1 + m2)


def IMRPhenomXHM_Insp_Amp_21_iv1(pWF):
    """IMRPhenomXHM_Insp_Amp_21_iv1, LALSimIMRPhenomXHM_inspiral.c:48-86, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = torch.abs(chidiff1*eta5*(-3962.5020052272976 + 987.635855365408*S1 - 134.98527058315528*S2) + delta*(19.30531354642419 + 16.6640319856064*eta1 - 120.58166037019478*eta2 + 220.77233521626252*eta3)*sqroot + chidiff1*delta*(31.364509907424765*eta1 - 843.6414532232126*eta2 + 2638.3077554662905*eta3)*sqroot + chidiff1*delta*(32.374226994179054*eta1 - 202.86279451816662*eta2 + 347.1621871204769*eta3)*S1*sqroot + delta*S1*(-16.75726972301224*(1.1787350890261943 - 7.812073811917883*eta1 + 99.47071002831267*eta2 - 500.4821414428368*eta3 + 876.4704270866478*eta4) + 2.3439955698372663*(0.9373952326655807 + 7.176140122833879*eta1 - 279.6409723479635*eta2 + 2178.375177755584*eta3 - 4768.212511142035*eta4)*S1)*sqroot)
    return total


def IMRPhenomXHM_Insp_Amp_21_iv2(pWF):
    """IMRPhenomXHM_Insp_Amp_21_iv2, LALSimIMRPhenomXHM_inspiral.c:88-126, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = torch.abs(chidiff1*eta5*(-2898.9172078672705 + 580.9465034962822*S1 + 22.251142639924076*S2) + delta*(chidiff2*(-18.541685007214625*eta1 + 166.7427445020744*eta2 - 417.5186332459383*eta3) + chidiff1*(41.61457952037761*eta1 - 779.9151607638761*eta2 + 2308.6520892707795*eta3))*sqroot + delta*(11.414934585404561 + 30.883118528233638*eta1 - 260.9979123967537*eta2 + 1046.3187137392433*eta3 - 1556.9475493549746*eta4)*sqroot + delta*S1*(-10.809007068469844*(1.1408749895922659 - 18.140470190766937*eta1 + 368.25127088896744*eta2 - 3064.7291458207815*eta3 + 11501.848278358668*eta4 - 16075.676528787526*eta5) + 1.0088254664333147*(1.2322739396680107 - 192.2461213084741*eta1 + 4257.760834055382*eta2 - 35561.24587952242*eta3 + 130764.22485304279*eta4 - 177907.92440833704*eta5)*S1)*sqroot + delta*(chidiff1*(36.88578491943111*eta1 - 321.2569602623214*eta2 + 748.6659668096737*eta3)*S1 + chidiff1*(-95.42418611585117*eta1 + 1217.338674959742*eta2 - 3656.192371615541*eta3)*S2)*sqroot)
    return total


def IMRPhenomXHM_Insp_Amp_21_iv3(pWF):
    """IMRPhenomXHM_Insp_Amp_21_iv3, LALSimIMRPhenomXHM_inspiral.c:128-163, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = torch.abs(chidiff1*eta5*(-2282.9983216879655 + 157.94791186394787*S1 + 16.379731479465033*S2) + chidiff1*delta*(21.935833431534224*eta1 - 460.7130131927895*eta2 + 1350.476411541137*eta3)*sqroot + delta*(5.390240326328237 + 69.01761987509603*eta1 - 568.0027716789259*eta2 + 2435.4098320959706*eta3 - 3914.3390484239667*eta4)*sqroot + chidiff1*delta*(29.731007410186827*eta1 - 372.09609843131386*eta2 + 1034.4897198648962*eta3)*S1*sqroot + delta*S1*(-7.1976397556450715*(0.7603360145475428 - 6.587249958654174*eta1 + 120.87934060776237*eta2 - 635.1835857158857*eta3 + 1109.0598539312573*eta4) - 0.0811847192323969*(7.951454648295709 + 517.4039644814231*eta1 - 9548.970156895082*eta2 + 52586.63520999897*eta3 - 93272.17990295641*eta4)*S1 - 0.28384547935698246*(-0.8870770459576875 + 180.0378964169756*eta1 - 2707.9572896559484*eta2 + 14158.178124971111*eta3 - 24507.800226675925*eta4)*S2)*sqroot)
    return total


def IMRPhenomXHM_Insp_Amp_33_iv1(pWF):
    """IMRPhenomXHM_Insp_Amp_33_iv1, LALSimIMRPhenomXHM_inspiral.c:165-200, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = chidiff1*eta5*(155.1434307076563 + 26.852777193715088*S1 + 1.4157230717300835*S2) + chidiff1*delta*(6.296698171560171*eta1 + 15.81328761563562*eta2 - 141.85538063933927*eta3)*sqroot + delta*(20.94372147101354 + 68.14577638017842*eta1 - 898.470298591732*eta2 + 4598.64854748635*eta3 - 8113.199260593833*eta4)*sqroot + chidiff1*delta*(29.221863857271703*eta1 - 348.1658322276406*eta2 + 965.4670353331536*eta3)*S1*sqroot + delta*S1*(-9.753610761811967*(1.7819678168496158 - 44.07982999150369*eta1 + 750.8933447725581*eta2 - 5652.44754829634*eta3 + 19794.855873435758*eta4 - 26407.40988450443*eta5) + 0.014210376114848208*(-196.97328616330392 + 7264.159472864562*eta1 - 125763.47850622259*eta2 + 1.1458022059130718e6*eta3 - 4.948175330328345e6*eta4 + 7.911048294733888e6*eta5)*S1 - 0.26859293613553986*(-8.029069605349488 + 888.7768796633982*eta1 - 16664.276483466252*eta2 + 128973.72291098491*eta3 - 462437.2690007375*eta4 + 639989.1197424605*eta5)*S2)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_33_iv2(pWF):
    """IMRPhenomXHM_Insp_Amp_33_iv2, LALSimIMRPhenomXHM_inspiral.c:202-241, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = chidiff1*eta5*(161.62678370819597 + 37.141092711336846*S1 - 0.16889712161410445*S2) + chidiff1*delta*(3.4895829486899825*eta1 + 51.07954458810889*eta2 - 249.71072528701757*eta3)*sqroot + delta*(12.501397517602173 + 35.75290806646574*eta1 - 357.6437296928763*eta2 + 1773.8883882162215*eta3 - 3100.2396041211605*eta4)*sqroot + chidiff1*delta*(13.854211287141906*eta1 - 135.54916401086845*eta2 + 327.2467193417936*eta3)*S1*sqroot + delta*S1*(-5.2580116732827085*(1.7794900975289085 - 48.20753331991333*eta1 + 861.1650630146937*eta2 - 6879.681319382729*eta3 + 25678.53964955809*eta4 - 36383.824902258915*eta5) + 0.028627002336747746*(-50.57295946557892 + 734.7581857539398*eta1 - 2287.0465658878725*eta2 + 15062.821881048358*eta3 - 168311.2370167227*eta4 + 454655.37836367317*eta5)*S1 - 0.15528289788512326*(-12.738184090548508 + 1129.44485109116*eta1 - 25091.14888164863*eta2 + 231384.03447562453*eta3 - 953010.5908118751*eta4 + 1.4516597366230418e6*eta5)*S2)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_33_iv3(pWF):
    """IMRPhenomXHM_Insp_Amp_33_iv3, LALSimIMRPhenomXHM_inspiral.c:243-285, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    eta7 = eta1 * eta6
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = chidiff1*delta*(-0.5869777957488564*eta1 + 32.65536124256588*eta2 - 110.10276573567405*eta3) + chidiff1*delta*(3.524800489907584*eta1 - 40.26479860265549*eta2 + 113.77466499598913*eta3)*S1 + delta*S1*(-1.2846335585108297*(0.09991079016763821 + 1.37856806162599*eta1 + 23.26434219690476*eta2 - 34.842921754693386*eta3 - 70.83896459998664*eta4) - 0.03496714763391888*(-0.230558571912664 + 188.38585449575902*eta1 - 3736.1574640444287*eta2 + 22714.70643022915*eta3 - 43221.0453556626*eta4)*S1) + chidiff1*eta7*(2667.3441342894776 + 47.94869769580204*chidiff2 + 793.5988192446642*S1 + 293.89657731755483*S2) + delta*(5.148353856800232 + 148.98231189649468*eta1 - 2774.5868652930294*eta2 + 29052.156454239772*eta3 - 162498.31493332976*eta4 + 460912.76402476896*eta5 - 521279.50781871413*eta6)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_32_iv1(pWF):
    """IMRPhenomXHM_Insp_Amp_32_iv1, LALSimIMRPhenomXHM_inspiral.c:287-329, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = S
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = (chidiff1*delta*(-0.739317114582042*eta1 - 47.473246070362634*eta2 + 278.9717709112207*eta3 - 566.6420939162068*eta4) + chidiff2*(-0.5873680378268906*eta1 + 6.692187014925888*eta2 - 24.37776782232888*eta3 + 23.783684827838247*eta4))*sqroot + (3.2940434453819694 + 4.94285331708559*eta1 - 343.3143244815765*eta2 + 3585.9269057886418*eta3 - 19279.186145681153*eta4 + 51904.91007211022*eta5 - 55436.68857586653*eta6)*sqroot + chidiff1*delta*(12.488240781993923*eta1 - 209.32038774208385*eta2 + 1160.9833883184604*eta3 - 2069.5349737049073*eta4)*S1*sqroot + S1*(0.6343034651912586*(-2.5844888818001737 + 78.98200041834092*eta1 - 1087.6241783616488*eta2 + 7616.234910399297*eta3 - 24776.529123239357*eta4 + 30602.210950069973*eta5) - 0.062088720220899465*(6.5586380356588565 + 36.01386705325694*eta1 - 3124.4712274775407*eta2 + 33822.437731298516*eta3 - 138572.93700180828*eta4 + 198366.10615196894*eta5)*S1)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_32_iv2(pWF):
    """IMRPhenomXHM_Insp_Amp_32_iv2, LALSimIMRPhenomXHM_inspiral.c:331-371, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = S
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = (chidiff2*(-0.03940151060321499*eta1 + 1.9034209537174116*eta2 - 8.78587250202154*eta3) + chidiff1*delta*(-1.704299788495861*eta1 - 4.923510922214181*eta2 + 0.36790005839460627*eta3))*sqroot + (2.2911849711339123 - 5.1846950040514335*eta1 + 60.10368251688146*eta2 - 1139.110227749627*eta3 + 7970.929280907627*eta4 - 25472.73682092519*eta5 + 30950.67053883646*eta6)*sqroot + S1*(0.7718201508695763*(-1.3012906461000349 + 26.432880113146012*eta1 - 186.5001124789369*eta2 + 712.9101229418721*eta3 - 970.2126139442341*eta4) + 0.04832734931068797*(-5.9999628512498315 + 78.98681284391004*eta1 + 1.8360177574514709*eta2 - 2537.636347529708*eta3 + 6858.003573909322*eta4)*S1)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_32_iv3(pWF):
    """IMRPhenomXHM_Insp_Amp_32_iv3, LALSimIMRPhenomXHM_inspiral.c:373-412, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = S
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = (chidiff2*(-0.6358511175987503*eta1 + 5.555088747533164*eta2 - 14.078156877577733*eta3) + chidiff1*delta*(0.23205448591711159*eta1 - 19.46049432345157*eta2 + 36.20685853857613*eta3))*sqroot + (1.1525594672495008 + 7.380126197972549*eta1 - 17.51265776660515*eta2 - 976.9940395257111*eta3 + 8880.536804741967*eta4 - 30849.228936891763*eta5 + 38785.53683146884*eta6)*sqroot + chidiff1*delta*(1.904350804857431*eta1 - 25.565242391371093*eta2 + 80.67120303906654*eta3)*S1*sqroot + S1*(0.785171689871352*(-0.4634745514643032 + 18.70856733065619*eta1 - 167.9231114864569*eta2 + 744.7699462372949*eta3 - 1115.008825153004*eta4) + 0.13469300326662165*(-2.7311391326835133 + 72.17373498208947*eta1 - 483.7040402103785*eta2 + 1136.8367114738041*eta3 - 472.02962341590774*eta4)*S1)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_44_iv1(pWF):
    """IMRPhenomXHM_Insp_Amp_44_iv1, LALSimIMRPhenomXHM_inspiral.c:414-453, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = (chidiff1*delta*(0.5697308729057493*eta1 + 8.895576813118867*eta2 - 34.98399465240273*eta3) + chidiff2*(1.6370346538130884*eta1 - 14.597095790380884*eta2 + 33.182723737396294*eta3))*sqroot + (5.2601381002242595 - 3.557926105832778*eta1 - 138.9749850448088*eta2 + 603.7453704122706*eta3 - 923.5495700703648*eta4)*sqroot + S1*(-0.41839636169678796*(5.143510231379954 + 104.62892421207803*eta1 - 4232.508174045782*eta2 + 50694.024801783446*eta3 - 283097.33358214336*eta4 + 758333.2655404843*eta5 - 788783.0559069642*eta6) - 0.05653522061311774*(5.605483124564013 + 694.00652410087*eta1 - 17551.398321516353*eta2 + 165236.6480734229*eta3 - 761661.9645651339*eta4 + 1.7440315410044065e6*eta5 - 1.6010489769238676e6*eta6)*S1 - 0.023693246676754775*(16.437107575918503 - 2911.2154288136217*eta1 + 89338.32554683842*eta2 - 1.0803340811860575e6*eta3 + 6.255666490084672e6*eta4 - 1.7434160932177313e7*eta5 + 1.883460394974573e7*eta6)*S2)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_44_iv2(pWF):
    """IMRPhenomXHM_Insp_Amp_44_iv2, LALSimIMRPhenomXHM_inspiral.c:455-495, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = S
    S2 = S1 * S1
    S3 = S1 * S2
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = (chidiff2*(-0.8318312659717388*eta1 + 7.6541168007977864*eta2 - 16.648660653220123*eta3) + chidiff1*delta*(2.214478316304753*eta1 - 7.028104574328955*eta2 + 5.56587823143958*eta3))*sqroot + (3.173191054680422 + 6.707695566702527*eta1 - 155.22519772642607*eta2 + 604.0067075996933*eta3 - 876.5048298377644*eta4)*sqroot + chidiff1*delta*(4.749663394334708*eta1 - 42.62996105525792*eta2 + 97.01712147349483*eta3)*S1*sqroot + S1*(-0.2627203100303006*(6.460396349297595 - 52.82425783851536*eta1 - 552.1725902144143*eta2 + 12546.255587592654*eta3 - 81525.50289542897*eta4 + 227254.37897941095*eta5 - 234487.3875219032*eta6) - 0.008424003742397579*(-109.26773035716548 + 15514.571912666677*eta1 - 408022.6805482195*eta2 + 4.620165968920881e6*eta3 - 2.6446950627957724e7*eta4 + 7.539643948937692e7*eta5 - 8.510662871580401e7*eta6)*S1 - 0.008830881730801855*(-37.49992494976597 + 1359.7883958101172*eta1 - 23328.560285901796*eta2 + 260027.4121353132*eta3 - 1.723865744472182e6*eta4 + 5.858455766230802e6*eta5 - 7.756341721552802e6*eta6)*S2 - 0.027167813927224657*(34.281932237450256 - 3312.7658728016568*eta1 + 84126.14531363266*eta2 - 956052.0170024392*eta3 + 5.570748509263883e6*eta4 - 1.6270212243584689e7*eta5 + 1.8855858173287075e7*eta6)*S3)*sqroot
    return total


def IMRPhenomXHM_Insp_Amp_44_iv3(pWF):
    """IMRPhenomXHM_Insp_Amp_44_iv3, LALSimIMRPhenomXHM_inspiral.c:497-537, case 122022."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    sqroot = torch.sqrt(eta)
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    eta6 = eta1 * eta5
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = (chidiff1*delta*(1.4739380748149558*eta1 + 0.06541707987699942*eta2 - 9.473290540936633*eta3) + chidiff2*(-0.3640838331639651*eta1 + 3.7369795937033756*eta2 - 8.709159662885131*eta3))*sqroot + (1.7335503724888923 + 12.656614578053683*eta1 - 139.6610487470118*eta2 + 456.78649322753824*eta3 - 599.2709938848282*eta4)*sqroot + chidiff1*delta*(2.3532739003216254*eta1 - 21.37216554136868*eta2 + 53.35003268489743*eta3)*S1*sqroot + S1*(-0.15782329022461472*(6.0309399412954345 - 229.16361598098678*eta1 + 3777.477006415653*eta2 - 31109.307191210424*eta3 + 139319.8239886073*eta4 - 324891.4001578353*eta5 + 307714.3954026392*eta6) - 0.03050157254864058*(4.232861441291087 + 1609.4251694451375*eta1 - 51213.27604422822*eta2 + 612317.1751155312*eta3 - 3.5589766538499263e6*eta4 + 1.0147654212772278e7*eta5 - 1.138861230369246e7*eta6)*S1 - 0.026407497690308382*(-17.184685557542196 + 744.4743953122965*eta1 - 10494.512487701073*eta2 + 66150.52694069289*eta3 - 184787.79377504133*eta4 + 148102.4257785174*eta5 + 128167.89151782403*eta6)*S2)*sqroot
    return total


# ----------------------------------------------------------------------
# PHASE
# ----------------------------------------------------------------------
# Spin parameter S = (m1^2*chi1 + m2^2*chi2)/(m1^2 + m2^2)
#
# Below we give paramater-space fits for the weighted difference between
# each mode's phase and the 22-phase: phi_lm-m/2 phi_22(2/m f),
# see Eqs. (4.10-4.12)


def IMRPhenomXHM_Insp_Phase_21_lambda(pWF):
    """IMRPhenomXHM_Insp_Phase_21_lambda, LALSimIMRPhenomXHM_inspiral.c:956-977, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    S2 = S**2
    noSpin = 13.664473636545068 - 170.08866400251395*eta + 3535.657736681598*eta2 - 26847.690494515424*eta3 + 96463.68163125668*eta4 - 133820.89317471132*eta5
    eqSpin = (S*(18.52571430563905 - 41.55066592130464*S + eta3*(83493.24265292779 + 16501.749243703132*S - 149700.4915210766*S2) + eta*(3642.5891077598003 + 1198.4163078715173*S - 6961.484805326852*S2) + 33.8697137964237*S2 + eta2*(-35031.361998480075 - 7233.191207000735*S + 62149.00902591944*S2)))/(6.880288191574696 + 1.*S)
    uneqSpin = -134.27742343186577*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    total = noSpin + eqSpin + uneqSpin
    return total


def IMRPhenomXHM_Insp_Phase_33_lambda(pWF):
    """IMRPhenomXHM_Insp_Phase_33_lambda, LALSimIMRPhenomXHM_inspiral.c:979-998, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    delta = torch.sqrt(1.-4.*eta)
    eta2 = eta**2
    eta3 = eta2*eta
    noSpin = 4.1138398568400705 + 9.772510519809892*eta - 103.92956504520747*eta2 + 242.3428625556764*eta3
    eqSpin = ((-0.13253553909611435 + 26.644159828590055*eta - 105.09339163109497*eta2)*S)/(1. + 0.11322426762297967*S)
    uneqSpin = -19.705359163581168*pWF["dchi"]*eta2*delta
    total = noSpin + eqSpin + uneqSpin
    return total


def IMRPhenomXHM_Insp_Phase_32_lambda(pWF):
    """IMRPhenomXHM_Insp_Phase_32_lambda, LALSimIMRPhenomXHM_inspiral.c:1000-1022, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    S2 = S**2
    S3 = S**3
    S4 = S**4
    noSpin = (9.913819875501506 + 18.424900617803107*eta - 574.8672384388947*eta2 + 2671.7813055097877*eta3 - 6244.001932443913*eta4)/(1. - 0.9103118343073325*eta)
    eqSpin = (-4.367632806613781 + 245.06757304950986*eta - 2233.9319708029775*eta2 + 5894.355429022858*eta3)*S + (-1.375112297530783 - 1876.760129419146*eta + 17608.172965575013*eta2 - 40928.07304790013*eta3)*S2 + (-1.28324755577382 - 138.36970336658558*eta + 708.1455154504333*eta2 - 273.23750933544176*eta3)*S3 + (1.8403161863444328 + 2009.7361967331492*eta - 18636.271414571278*eta2 + 42379.205045791656*eta3)*S4
    uneqSpin = pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2*(-105.34550407768225 - 1566.1242344157668*pWF["chi1L"]*eta + 1566.1242344157668*pWF["chi2L"]*eta + 2155.472229664981*eta*S)
    total = noSpin + eqSpin + uneqSpin
    return total


def IMRPhenomXHM_Insp_Phase_44_lambda(pWF):
    """IMRPhenomXHM_Insp_Phase_44_lambda, LALSimIMRPhenomXHM_inspiral.c:1025-1046, case 122019."""
    eta = pWF["eta"]
    S = pWF["STotR"]
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    S2 = S**2
    noSpin = 5.254484747463392 - 21.277760168559862*eta + 160.43721442910618*eta2 - 1162.954360723399*eta3 + 1685.5912722190276*eta4 - 1538.6661348106031*eta5
    eqSpin = (0.007067861615983771 - 10.945895160727437*eta + 246.8787141453734*eta2 - 810.7773268493444*eta3)*S + (0.17447830920234977 + 4.530539154777984*eta - 176.4987316167203*eta2 + 621.6920322846844*eta3)*S2
    uneqSpin = -8.384066369867833*pWF["dchi"]*torch.sqrt(1.-4.*eta)*eta2
    total = noSpin + eqSpin + uneqSpin
    return total


# Returns linear-in-f term coming from the complex part of each mode's
# amplitude that will be added to the orbital phase, this is an expansion
# of Eq. (4.9) truncated at linear order
def IMRPhenomXHM_Insp_Phase_LambdaPN(eta, modeTag):
    """IMRPhenomXHM_Insp_Phase_LambdaPN, LALSimIMRPhenomXHM_inspiral.c:1051-1081, all mode cases (21, 33, 32, 44)."""
    PI = math.pi  # powers_of_lalpiHM.itself (LAL_PI)
    if modeTag == 21:
        # 2 f \[Pi] (-(1/2) - Log[16]/2)
        output = (2.*PI*(-0.5-2.*math.log(2))) * torch.ones_like(eta)
    elif modeTag == 33:
        # 2/3 f \[Pi] (-(21/5) + 6 Log[3/2])
        output = 2./3.*PI*(-21./5.+6.*math.log(1.5)) * torch.ones_like(eta)
    elif modeTag == 32:
        # -((2376 f \[Pi] (-5 + 22 \[Eta]))/(-3960 + 11880 \[Eta]))
        output = -((2376.*PI*(-5.+22.*eta))/(-3960.+11880*eta))
    elif modeTag == 44:
        # (45045 f \[powers_of_lalpiHM.itself] (336 - 1193 \[Eta] +320 (-1 + 3 \[Eta]) Log[2]))/(2 (-1801800 + 5405400 \[Eta]))
        output = 45045.*PI*(336.-1193.*eta+320.*(-1.+3.*eta)*math.log(2))/(2.*(-1801800. + 5405400.*eta))
    else:
        output = torch.zeros_like(eta)
    return -1.*output

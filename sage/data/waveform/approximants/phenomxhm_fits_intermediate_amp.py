#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : phenomxhm_fits_intermediate_amp.py
Description   : Batched PyTorch port of the IMRPhenomXHM intermediate-region
                amplitude collocation-point parameter-space fits.

                Mirrors LALSimIMRPhenomXHM_intermediate.c from
                lalsimulation-v6.1.0 (vendored in
                sage/data/waveform/lalsim_src/):
                  c:49-349   IMRPhenomXHM_Inter_Amp_{21,33,32,44}_{int1,int2}
                  c:514-711  IMRPhenomXHM_Inter_Amp_{21,33,32,44}_{int3,int4}
                             (extra collocation points of release 122022)
                Only the `case 122022` branch of each switch is ported — the
                amplitude release executed by the v6.1.0 default path.  The
                EMR extra-collocation-point fits
                IMRPhenomXHM_Inter_Amp_lm_int0/dint0 (c:351-511) exist only
                for the earlier release (case 122018, used by the 122019
                amplitude-version path) and are never reached by the 122022
                amplitude path, so they are NOT ported.

                Each function takes a single dict pWF whose keys are the
                IMRPhenomXWaveformStruct member names consumed by the C body
                (eta, delta, STotR, chiPNHat, dchi_half), all (B,1) float64
                torch tensors, and returns a (B,1) float64 tensor.

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
Fits          : supplementary material of https://dcc.ligo.org/P2000011-v2
"""

import torch

# Fits of the intermediate collocation points over parameter space.

# These parameter-space fits are documented in the supplementary material of
# https://dcc.ligo.org/P2000011-v2.  There are 2 Mathematica notebooks (one for
# amplitude and one for phase) that read the fits data and automatically
# generate the C-code below.  For more information read
# https://git.ligo.org/waveforms/reviews/imrphenomx/blob/master/documentation/ParspaceFits/README
# and the documentation in the notebooks.

# IMRPhenomXHM_Inter_Amp_lm_intX returns the value of the amplitude at the
# collocation point intX.
# for a definition of the frequencies corresponding to int0/1/2, see Sec V.A.

# The dominant spin parameter is S = (m1^2*chi1 + m2^2*chi2)/(m1^2 + m2^2)

# Start of Amp Parameter Space Fits


def IMRPhenomXHM_Inter_Amp_21_int1(pWF):
    """IMRPhenomXHM_Inter_Amp_21_int1, LALSimIMRPhenomXHM_intermediate.c:49-84, case 122022 (c:64-80)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
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
    total = torch.abs(delta*eta1*(chidiff2*(5.159755997682368*eta1 - 30.293198248154948*eta2 + 63.70715919820867*eta3) + chidiff1*(8.262642080222694*eta1 - 415.88826990259116*eta2 + 1427.5951158851076*eta3)) + delta*eta1*(18.55363583212328 - 66.46950491124205*eta1 + 447.2214642597892*eta2 - 1614.178472020212*eta3 + 2199.614895727586*eta4) + chidiff1*eta5*(-1698.841763891122 - 195.27885562092342*S1 - 1.3098861736238572*S2) + delta*eta1*(chidiff1*(34.17829404207186*eta1 - 386.34587928670015*eta2 + 1022.8553774274128*eta3)*S1 + chidiff1*(56.76554600963724*eta1 - 491.4593694689354*eta2 + 1016.6019654342113*eta3)*S2) + delta*eta1*S1*(-8.276366844994188*(1.0677538075697492 - 24.12941323757896*eta1 + 516.7886322104276*eta2 - 4389.799658723288*eta3 + 16770.447637953577*eta4 - 23896.392706809565*eta5) - 1.6908277400304084*(3.4799140066657928 - 29.00026389706585*eta1 + 114.8330693231833*eta2 - 184.13091281984674*eta3 + 592.300353344717*eta4 - 2085.0821513466053*eta5)*S1 - 0.46006975902558517*(-2.1663474937625975 + 826.026625945615*eta1 - 17333.549622759732*eta2 + 142904.08962903373*eta3 - 528521.6231015554*eta4 + 731179.456702448*eta5)*S2))
    return total


def IMRPhenomXHM_Inter_Amp_21_int2(pWF):
    """IMRPhenomXHM_Inter_Amp_21_int2, LALSimIMRPhenomXHM_intermediate.c:86-118, case 122022 (c:99-114)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = torch.abs(delta*eta1*(13.757856231617446 - 12.783698329428516*eta1 + 12.048194546899204*eta2) + chidiff1*delta*eta1*(15.107530092096438*eta1 - 416.811753638553*eta2 + 1333.6181181686939*eta3) + chidiff1*eta5*(-1549.6199518612063 - 102.34716990474509*S1 - 3.3637011939285015*S2) + delta*eta1*(chidiff1*(36.358142200869295*eta1 - 384.2123173145321*eta2 + 984.6826660818275*eta3)*S1 + chidiff1*(4.159271594881928*eta1 + 105.10911749116399*eta2 - 639.190132707115*eta3)*S2) + delta*eta1*S1*(-8.097876227116853*(0.6569459700232806 + 9.861355377849485*eta1 - 116.88834714736281*eta2 + 593.8035334117192*eta3 - 1063.0692862578455*eta4) - 1.0546375154878165*(0.745557030602097 + 65.25215540635162*eta1 - 902.5751736558435*eta2 + 4350.442990924205*eta3 - 7141.611333893155*eta4)*S1 - 0.5006664599166409*(10.289020582277626 - 212.00728173197498*eta1 + 2334.0029399672358*eta2 - 11939.621138801092*eta3 + 21974.8201355744*eta4)*S2))
    return total


def IMRPhenomXHM_Inter_Amp_33_int1(pWF):
    """IMRPhenomXHM_Inter_Amp_33_int1, LALSimIMRPhenomXHM_intermediate.c:120-156, case 122022 (c:137-152)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = chidiff1*delta*eta1*(-0.3516244197696068*eta1 + 40.425151307421416*eta2 - 148.3162618111991*eta3) + delta*eta1*(26.998512565991778 - 146.29035440932105*eta1 + 914.5350366065115*eta2 - 3047.513201789169*eta3 + 3996.417635728702*eta4) + chidiff1*delta*eta1*(5.575274516197629*eta1 - 44.592719238427094*eta2 + 99.91399033058927*eta3)*S1 + delta*eta1*S1*(-0.5383304368673182*(-7.456619067234563 + 129.36947401891433*eta1 - 843.7897535238325*eta2 + 3507.3655567272644*eta3 - 9675.194644814854*eta4 + 11959.83533107835*eta5) - 0.28042799223829407*(-6.212827413930676 + 266.69059813274475*eta1 - 4241.537539226717*eta2 + 32634.43965039936*eta3 - 119209.70783201039*eta4 + 166056.27237509796*eta5)*S1) + chidiff1*eta5*(199.6863414922219 + 53.36849263931051*S1 + 7.650565415855383*S2)
    return total


def IMRPhenomXHM_Inter_Amp_33_int2(pWF):
    """IMRPhenomXHM_Inter_Amp_33_int2, LALSimIMRPhenomXHM_intermediate.c:158-192, case 122022 (c:173-188)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = delta*eta1*(17.42562079069636 - 28.970875603981295*eta1 + 50.726220750178435*eta2) + chidiff1*delta*eta1*(-7.861956897615623*eta1 + 93.45476935080045*eta2 - 273.1170921735085*eta3) + chidiff1*delta*eta1*(-0.3265505633310564*eta1 - 9.861644053348053*eta2 + 60.38649425562178*eta3)*S1 + chidiff1*eta5*(234.13476431269862 + 51.2153901931183*S1 - 10.05114600643587*S2) + delta*eta1*S1*(0.3104472390387834*(6.073591341439855 + 169.85423386969634*eta1 - 4964.199967099143*eta2 + 42566.59565666228*eta3 - 154255.3408672655*eta4 + 205525.13910847943*eta5) + 0.2295327944679772*(19.236275867648594 - 354.7914372697625*eta1 + 1876.408148917458*eta2 + 2404.4151687877525*eta3 - 41567.07396803811*eta4 + 79210.33893514868*eta5)*S1 + 0.30983324991828787*(11.302200127272357 - 719.9854052004307*eta1 + 13278.047199998868*eta2 - 104863.50453518033*eta3 + 376409.2335857397*eta4 - 504089.07690692553*eta5)*S2)
    return total


def IMRPhenomXHM_Inter_Amp_32_int1(pWF):
    """IMRPhenomXHM_Inter_Amp_32_int1, LALSimIMRPhenomXHM_intermediate.c:194-232, case 122022 (c:212-228)."""
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
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = (chidiff2*(-0.2341404256829785*eta1 + 2.606326837996192*eta2 - 8.68296921440857*eta3) + chidiff1*delta*(0.5454562486736877*eta1 - 25.19759222940851*eta2 + 73.40268975811729*eta3))*sqroot + chidiff1*delta*(0.4422257616009941*eta1 - 8.490112284851655*eta2 + 32.22238925527844*eta3)*S1*sqroot + S1*(0.7067243321652764*(0.12885110296881636 + 9.608999847549535*eta1 - 85.46581740280585*eta2 + 325.71940024255775*eta3 + 175.4194342269804*eta4 - 1929.9084724384807*eta5) + 0.1540566313813899*(-0.3261041495083288 + 45.55785402900492*eta1 - 827.591235943271*eta2 + 7184.647314370326*eta3 - 28804.241518798244*eta4 + 43309.69769878964*eta5)*S1)*sqroot + (480.0434256230109*eta1 + 25346.341240810478*eta2 - 99873.4707358776*eta3 + 106683.98302194536*eta4)*sqroot*(1 + 1082.6574834474493*eta1 + 10083.297670051445*eta2)**(-1)
    return total


def IMRPhenomXHM_Inter_Amp_32_int2(pWF):
    """IMRPhenomXHM_Inter_Amp_32_int2, LALSimIMRPhenomXHM_intermediate.c:234-271, case 122022 (c:251-266)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
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
    total = eta1*(chidiff2*(-4.175680729484314*eta1 + 47.54281549129226*eta2 - 128.88334273588077*eta3) + chidiff1*delta*(-0.18274358639599947*eta1 - 71.01128541687838*eta2 + 208.07105580635888*eta3)) + eta1*(4.760999387359598 - 38.57900689641654*eta1 + 456.2188780552874*eta2 - 4544.076411013166*eta3 + 24956.9592553473*eta4 - 69430.10468748478*eta5 + 77839.74180254337*eta6) + chidiff1*delta*eta1*(1.2198776533959694*eta1 - 26.816651899746475*eta2 + 68.72798751937934*eta3)*S1 + eta1*S1*(1.5098291294292217*(0.4844667556328104 + 9.848766999273414*eta1 - 143.66427232396376*eta2 + 856.9917885742416*eta3 - 1633.3295758142904*eta4) + 0.32413108737204144*(2.835358206961064 - 62.37317183581803*eta1 + 761.6103793011912*eta2 - 3811.5047139343505*eta3 + 6660.304740652403*eta4)*S1)
    return total


def IMRPhenomXHM_Inter_Amp_44_int1(pWF):
    """IMRPhenomXHM_Inter_Amp_44_int1, LALSimIMRPhenomXHM_intermediate.c:273-308, case 122022 (c:289-303)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = eta1*(chidiff1*delta*(1.5378890240544967*eta1 - 3.4499418893734903*eta2 + 16.879953490422782*eta3) + chidiff2*(1.720226708214248*eta1 - 11.87925165364241*eta2 + 23.259283336239545*eta3)) + eta1*(8.790173464969538 - 64.95499142822892*eta1 + 324.1998823562892*eta2 - 1111.9864921907126*eta3 + 1575.602443847111*eta4) + eta1*S1*(-0.062333275821238224*(-21.630297087123807 + 137.4395894877131*eta1 + 64.92115530780129*eta2 - 1013.1110639471394*eta3) - 0.11014697070998722*(4.149721483857751 - 108.6912882442823*eta1 + 831.6073263887092*eta2 - 1828.2527520190122*eta3)*S1 - 0.07704777584463054*(4.581767671445529 - 50.35070009227704*eta1 + 344.9177692251726*eta2 - 858.9168637051405*eta3)*S2)
    return total


def IMRPhenomXHM_Inter_Amp_44_int2(pWF):
    """IMRPhenomXHM_Inter_Amp_44_int2, LALSimIMRPhenomXHM_intermediate.c:310-346, case 122022 (c:326-341)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
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
    total = eta1*(chidiff1*delta*(2.3123974306694057*eta1 - 12.237594841284904*eta2 + 44.78225529547671*eta3) + chidiff2*(2.9282931698944292*eta1 - 25.624210264341933*eta2 + 61.05270871360041*eta3)) + eta1*(6.98072197826729 - 46.81443520117986*eta1 + 236.76146303619544*eta2 - 920.358408667518*eta3 + 1478.050456337336*eta4) + eta1*S1*(-0.07801583359561987*(-28.29972282146242 + 752.1603553640072*eta1 - 10671.072606753183*eta2 + 83447.0461509547*eta3 - 350025.2112501252*eta4 + 760889.6919776166*eta5 - 702172.2934567826*eta6) + 0.013159545629626014*(91.1469833190294 - 3557.5003799977294*eta1 + 52391.684517955284*eta2 - 344254.9973814295*eta3 + 1.0141877915334814e6*eta4 - 1.1505186449682908e6*eta5 + 268756.85659532435*eta6)*S1)
    return total


# Extra collocation points of the 122022 amplitude release.


def IMRPhenomXHM_Inter_Amp_21_int3(pWF):
    """IMRPhenomXHM_Inter_Amp_21_int3, LALSimIMRPhenomXHM_intermediate.c:514-536, case 122022 (c:517-531)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = torch.abs(delta*eta1*(13.318990196097973 - 21.755549987331054*eta1 + 76.14884211156267*eta2 - 127.62161159798488*eta3) + chidiff1*delta*eta1*(17.704321326939414*eta1 - 434.4390350012534*eta2 + 1366.2408490833282*eta3) + chidiff1*delta*eta1*(11.877985158418596*eta1 - 131.04937626836355*eta2 + 343.79587860999874*eta3)*S1 + chidiff1*eta5*(-1522.8543551416456 - 16.639896279650678*S1 + 3.0053086651515843*S2) + delta*eta1*S1*(-8.665646058245033*(0.7862132291286934 + 8.293609541933655*eta1 - 111.70764910503321*eta2 + 576.7172598056907*eta3 - 1001.2370065269745*eta4) - 0.9459820574514348*(1.309016452198605 + 48.94077040282239*eta1 - 817.7854010574645*eta2 + 4331.56002883546*eta3 - 7518.309520232795*eta4)*S1 - 0.4308267743835775*(9.970654092010587 - 302.9708323417439*eta1 + 3662.099161055873*eta2 - 17712.883990278668*eta3 + 29480.158198408903*eta4)*S2))
    return total


def IMRPhenomXHM_Inter_Amp_21_int4(pWF):
    """IMRPhenomXHM_Inter_Amp_21_int4, LALSimIMRPhenomXHM_intermediate.c:538-560, case 122022 (c:541-555)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = torch.abs(delta*eta1*(13.094382343446163 - 22.831152256559523*eta1 + 83.20619262213437*eta2 - 139.25546924151664*eta3) + chidiff1*delta*eta1*(20.120192352555357*eta1 - 458.2592421214168*eta2 + 1430.3698681181*eta3) + chidiff1*delta*eta1*(12.925363020014743*eta1 - 126.87194512915104*eta2 + 280.6003655502327*eta3)*S1 + chidiff1*eta5*(-1528.956015503355 + 74.44462583487345*S1 - 2.2456928156392197*S2) + delta*eta1*S1*(-9.499741513411829*(0.912120958549489 + 2.400945118514037*eta1 - 33.651192908287236*eta2 + 166.04254881175257*eta3 - 248.5050377498615*eta4) - 0.7850652143322492*(1.534131218043425 + 60.81773903539479*eta1 - 1032.1319480683567*eta2 + 5381.481380750608*eta3 - 9077.037917192794*eta4)*S1 - 0.21540359093306097*(9.42805409480658 - 109.06544597367301*eta1 + 385.8345793110262*eta2 + 1889.9613367802453*eta3 - 9835.416414460055*eta4)*S2))
    return total


def IMRPhenomXHM_Inter_Amp_33_int3(pWF):
    """IMRPhenomXHM_Inter_Amp_33_int3, LALSimIMRPhenomXHM_intermediate.c:562-584, case 122022 (c:565-579)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    eta5 = eta1 * eta4
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    total = delta*eta1*(14.555522136327964 - 12.799844096694798*eta1 + 16.79500349318081*eta2) + chidiff1*delta*eta1*(-16.292654447108134*eta1 + 190.3516012682791*eta2 - 562.0936797781519*eta3) + chidiff1*delta*eta1*(-7.048898856045782*eta1 + 49.941617405768135*eta2 - 73.62033985436068*eta3)*S1 + chidiff1*eta5*(263.5151703818307 + 44.408527093031566*S1 + 10.457035444964653*S2) + delta*eta1*S1*(0.4590550434774332*(3.0594364612798635 + 207.74562213604057*eta1 - 5545.0086137386525*eta2 + 50003.94075934942*eta3 - 195187.55422847517*eta4 + 282064.174913521*eta5) + 0.657748992123043*(5.57939137343977 - 124.06189543062042*eta1 + 1276.6209573025596*eta2 - 6999.7659193505915*eta3 + 19714.675715229736*eta4 - 20879.999628681435*eta5)*S1 + 0.3695850566805098*(6.077183107132255 - 498.95526910874986*eta1 + 10426.348944657859*eta2 - 91096.64982858274*eta3 + 360950.6686625352*eta4 - 534437.8832860565*eta5)*S2)
    return total


def IMRPhenomXHM_Inter_Amp_33_int4(pWF):
    """IMRPhenomXHM_Inter_Amp_33_int4, LALSimIMRPhenomXHM_intermediate.c:586-609, case 122022 (c:589-604)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
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
    total = delta*eta1*(13.312095699772305 - 7.449975618083432*eta1 + 17.098576301150125*eta2) + delta*eta1*(chidiff1*(-31.171150896110156*eta1 + 371.1389274783572*eta2 - 1103.1917047361735*eta3) + chidiff2*(32.78644599730888*eta1 - 395.15713118955387*eta2 + 1164.9282236341376*eta3)) + chidiff1*delta*eta1*(-46.85669289852532*eta1 + 522.3965959942979*eta2 - 1485.5134187612182*eta3)*S1 + chidiff1*eta5*(287.90444670305715 - 21.102665129433042*chidiff2 + 7.635582066682054*S1 - 29.471275170013012*S2) + delta*eta1*S1*(0.6893003654021495*(3.1014226377197027 - 44.83989278653052*eta1 + 565.3767256471909*eta2 - 4797.429130246123*eta3 + 19514.812242035154*eta4 - 27679.226582207506*eta5) + 0.7068016563068026*(4.071212304920691 - 118.51094098279343*eta1 + 1788.1730303291356*eta2 - 13485.270489656365*eta3 + 48603.96661003743*eta4 - 65658.74746265226*eta5)*S1 + 0.2181399561677432*(-1.6754158383043574 + 303.9394443302189*eta1 - 6857.936471898544*eta2 + 59288.71069769708*eta3 - 216137.90827404748*eta4 + 277256.38289831823*eta5)*S2)
    return total


def IMRPhenomXHM_Inter_Amp_44_int3(pWF):
    """IMRPhenomXHM_Inter_Amp_44_int3, LALSimIMRPhenomXHM_intermediate.c:611-633, case 122022 (c:614-628)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["chiPNHat"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    S1 = S
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = eta1*(chidiff1*delta*(-0.8765502142143329*eta1 + 22.806632458441996*eta2 - 43.675503209991184*eta3) + chidiff2*(0.48698617426180074*eta1 - 4.302527065360426*eta2 + 16.18571810759235*eta3)) + eta1*(6.379772583015967 - 44.10631039734796*eta1 + 269.44092930942793*eta2 - 1285.7635006711453*eta3 + 2379.538739132234*eta4) + eta1*S1*(-0.23316184683282615*(-1.7279023138971559 - 23.606399143993716*eta1 + 409.3387618483284*eta2 - 1115.4147472977265*eta3) - 0.09653777612560172*(-5.310643306559746 - 2.1852511802701264*eta1 + 541.1248219096527*eta2 - 1815.7529908827103*eta3)*S1 - 0.060477799540741804*(-14.578189130145661 + 175.6116682068523*eta1 - 569.4799973930861*eta2 + 426.0861915646515*eta3)*S2)
    return total


def IMRPhenomXHM_Inter_Amp_44_int4(pWF):
    """IMRPhenomXHM_Inter_Amp_44_int4, LALSimIMRPhenomXHM_intermediate.c:635-658, case 122022 (c:638-653)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
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
    total = eta1*(chidiff1*delta*(-2.461738962276138*eta1 + 45.3240543970684*eta2 - 112.2714974622516*eta3) + chidiff2*(0.9158352037567031*eta1 - 8.724582331021695*eta2 + 28.44633544874233*eta3)) + eta1*(6.098676337298138 - 45.42463610529546*eta1 + 350.97192927929433*eta2 - 2002.2013283876834*eta3 + 4067.1685640401033*eta4) + eta1*S1*(-0.36068516166901304*(-2.120354236840677 - 47.56175350408845*eta1 + 1618.4222330016048*eta2 - 14925.514654896673*eta3 + 60287.45399959349*eta4 - 91269.3745059139*eta5) - 0.09635801207669747*(-11.824692837267394 + 371.7551657959369*eta1 - 4176.398139238679*eta2 + 16655.87939259747*eta3 - 4102.218189945819*eta4 - 67024.98285179552*eta5)*S1 - 0.06565232123453196*(-26.15227471380236 + 1869.0168486099005*eta1 - 33951.35186039629*eta2 + 253694.6032002248*eta3 - 845341.6001856657*eta4 + 1.0442282862506858e6*eta5)*S2)
    return total


def IMRPhenomXHM_Inter_Amp_32_int3(pWF):
    """IMRPhenomXHM_Inter_Amp_32_int3, LALSimIMRPhenomXHM_intermediate.c:660-684, case 122022 (c:663-679)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
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
    total = 3.881450518842405*eta1 - 12.580316392558837*eta2 + 1.7262466525848588*eta3 + chidiff2*(-7.065118823041031*eta2 + 77.97950589523865*eta3 - 203.65975422378446*eta4) - 58.408542930248046*eta4 + chidiff1*delta*(1.924723094787216*eta2 - 90.92716917757797*eta3 + 387.00162600306226*eta4) + 403.5748987560612*eta5 + chidiff1*delta*(-0.2566958540737833*eta2 + 14.488550203412675*eta3 - 26.46699529970884*eta4)*S1 + S1*(0.3650871458400108*(71.57390929624825*eta2 - 994.5272351916166*eta3 + 6734.058809060536*eta4 - 18580.859291282686*eta5 + 16001.318492586077*eta6) + 0.0960146077440495*(451.74917589707513*eta2 - 9719.470997418284*eta3 + 83403.5743434538*eta4 - 318877.43061174755*eta5 + 451546.88775684836*eta6)*S1 - 0.03985156529181297*(-304.92981902871617*eta2 + 3614.518459296278*eta3 - 7859.4784979916085*eta4 - 46454.57664737511*eta5 + 162398.81483375572*eta6)*S2)
    return total


def IMRPhenomXHM_Inter_Amp_32_int4(pWF):
    """IMRPhenomXHM_Inter_Amp_32_int4, LALSimIMRPhenomXHM_intermediate.c:686-707, case 122022 (c:689-702)."""
    eta = pWF["eta"]
    delta = pWF["delta"]
    S = pWF["STotR"]
    chidiff = pWF["dchi_half"]
    eta1 = eta
    eta2 = eta1 * eta1
    eta3 = eta1 * eta2
    eta4 = eta1 * eta3
    S1 = S
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = eta1*(chidiff2*(-8.572797326909152*eta1 + 92.95723645687826*eta2 - 236.2438921965621*eta3) + chidiff1*delta*(6.674358856924571*eta1 - 171.4826985994883*eta2 + 645.2760206304703*eta3)) + eta1*(3.921660532875504 - 16.57299637423352*eta1 + 25.254017911686333*eta2 - 143.41033155133266*eta3 + 692.926425981414*eta4) + chidiff1*delta*eta1*(-3.582040878719185*eta1 + 57.75888914133383*eta2 - 144.21651114700492*eta3)*S1 + eta1*S1*(1.242750265695504*(-0.522172424518215 + 25.168480118950065*eta1 - 303.5223688400309*eta2 + 1858.1518762309654*eta3 - 3797.3561904195085*eta4) + 0.2927045241764365*(0.5056957789079993 - 15.488754837330958*eta1 + 471.64047356915603*eta2 - 3131.5783196211587*eta3 + 6097.887891566872*eta4)*S1)
    return total


# End of Amp Parameter Space Fits

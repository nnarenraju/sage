#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : phenomxhm_fits_ringdown_amp.py
Description   : Batched PyTorch port of the IMRPhenomXHM ringdown amplitude
                parameter-space fits.

                Mirrors LALSimIMRPhenomXHM_ringdown.c (lalsimulation-v6.1.0,
                vendored in sage/data/waveform/lalsim_src/), amplitude fits
                section c:43-722 ("Start of Amp Parameter Space Fits" to
                "End of Amp Parameter Space Fits").  Only the case-122022
                branches are ported: the v6.1.0 default release (122022,
                LALSimIMRPhenomXHM_internals.c:296-345) sets
                IMRPhenomXHMRingdownAmpFitsVersion = 122022 and
                IMRPhenomXHMRingdownAmpVersion = 2 (21, 33, 44 modes) or
                1 (32 mode), so IMRPhenomXHM_RD_Amp_Coefficients
                (ringdown.c:727-816, called from internals.c:1231) reaches
                exactly:

                  32 mode (RingdownAmpVersion 1, RingdownAmpFits[6..8]):
                    IMRPhenomXHM_RD_Amp_32_{alambda, lambda, sigma}
                  21/33/44 modes (RingdownAmpVersion 2,
                                  RingdownAmpFits[12+3*modeInt ..]):
                    IMRPhenomXHM_RD_Amp_{21,33,44}_rdcp{1,2,3}
                  32 mode auxiliary region (MixingOn, RingdownAmpFits[24,25],
                                            ringdown.c:818-892):
                    IMRPhenomXHM_RD_Amp_32_rdaux{1,2}

                Each function takes a dict pWF of IMRPhenomXWaveformStruct
                members as (B, 1) float64 tensors and returns a (B, 1)
                tensor.  Keys consumed: eta, delta, STotR, chiPNHat,
                dchi_half.

                The spin parameter S = (m1^2*chi1 + m2^2*chi2)/(m1^2 + m2^2)

                alambda, lambda and sigma are the coefficients of the
                ringdown ansatz, see Eq. (6.2)

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
IMRPhenomXHM  : Garcia-Quiros et al. (2020), arXiv:2001.10914
Parameter-space fits : supplementary material of https://dcc.ligo.org/P2000011-v2
"""

import torch

# Fits over parameter space for the ringdown amplitude coefficients
# (alambda, lambda, sigma) for each mode.


def IMRPhenomXHM_RD_Amp_32_alambda(pWF):
    """IMRPhenomXHM_RD_Amp_32_alambda, LALSimIMRPhenomXHM_ringdown.c:154-193, case 122022."""
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
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = chidiff2*(-3.4614418482110163*eta3 + 35.464117772624164*eta4 - 85.19723511005235*eta5) + chidiff1*delta*(2.0328561081997463*eta3 - 46.18751757691501*eta4 + 170.9266105597438*eta5) + chidiff2*(-0.4600401291210382*eta3 + 12.23450117663151*eta4 - 42.74689906831975*eta5)*S1 + chidiff1*delta*(5.786292428422767*eta3 - 53.60467819078566*eta4 + 117.66195692191727*eta5)*S1 + S1*(-0.0013330716557843666*(56.35538385647113*eta1 - 1218.1550992423377*eta2 + 16509.69605686402*eta3 - 102969.88022112886*eta4 + 252228.94931931415*eta5 - 150504.2927996263*eta6) + 0.0010126460331462495*(-33.87083889060834*eta1 + 502.6221651850776*eta2 - 1304.9210590188136*eta3 - 36980.079328277505*eta4 + 295469.28617550555*eta5 - 597155.7619486618*eta6)*S1 - 0.00043088431510840695*(-30.014415072587354*eta1 - 1900.5495690280086*eta2 + 76517.21042363928*eta3 - 870035.1394696251*eta4 + 3.9072674134789007e6*eta5 - 6.094089675611567e6*eta6)*S2) + (0.08408469319155859*eta1 - 1.223794846617597*eta2 + 6.5972460654253515*eta3 - 15.707327897569396*eta4 + 14.163264397061505*eta5)*(1 - 8.612447115134758*eta1 + 18.93655612952139*eta2)**(-1)
    return total


def IMRPhenomXHM_RD_Amp_32_lambda(pWF):
    """IMRPhenomXHM_RD_Amp_32_lambda, LALSimIMRPhenomXHM_ringdown.c:195-229, case 122022."""
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
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = 0.978510781593996 + 0.36457571743142897*eta1 - 12.259851752618998*eta2 + 49.19719473681921*eta3 + chidiff1*delta*(-188.37119473865533*eta3 + 2151.8731700399308*eta4 - 6328.182823770599*eta5) + chidiff2*(115.3689949926392*eta3 - 1159.8596972989067*eta4 + 2657.6998831179444*eta5) + S1*(0.22358643406992756*(0.48943645614341924 - 32.06682257944444*eta1 + 365.2485484044132*eta2 - 915.2489655397206*eta3) + 0.0792473022309144*(1.877251717679991 - 103.65639889587327*eta1 + 1202.174780792418*eta2 - 3206.340850767219*eta3)*S1)
    return total


def IMRPhenomXHM_RD_Amp_32_sigma(pWF):
    """IMRPhenomXHM_RD_Amp_32_sigma, LALSimIMRPhenomXHM_ringdown.c:324-353, case 122022."""
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
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = 1.3353917551819414 + 0.13401718687342024*eta1 + chidiff1*delta*(144.37065005786636*eta3 - 754.4085447486738*eta4 + 123.86194078913776*eta5) + chidiff2*(209.09202210427972*eta3 - 1769.4658099037918*eta4 + 3592.287297392387*eta5) + S1*(-0.012086025709597246*(-6.230497473791485 + 600.5968613752918*eta1 - 6606.1009717965735*eta2 + 17277.60594350428*eta3) - 0.06066548829900489*(-0.9208054306316676 + 142.0346574366267*eta1 - 1567.249168668069*eta2 + 4119.373703246675*eta3)*S1)
    return total


def IMRPhenomXHM_RD_Amp_21_rdcp1(pWF):
    """IMRPhenomXHM_RD_Amp_21_rdcp1, LALSimIMRPhenomXHM_ringdown.c:371-394, case 122022."""
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
    total = torch.abs(delta*eta1*(12.880905080761432 - 23.5291063016996*eta1 + 92.6090002736012*eta2 - 175.16681482428694*eta3) + chidiff1*delta*eta1*(26.89427230731867*eta1 - 710.8871223808559*eta2 + 2255.040486907459*eta3) + chidiff1*delta*eta1*(21.402708785047853*eta1 - 232.07306353130417*eta2 + 591.1097623278739*eta3)*S1 + delta*eta1*S1*(-10.090867481062709*(0.9580746052260011 + 5.388149112485179*eta1 - 107.22993216128548*eta2 + 801.3948756800821*eta3 - 2688.211889175019*eta4 + 3950.7894052628735*eta5 - 1992.9074348833092*eta6) - 0.42972412296628143*(1.9193131231064235 + 139.73149069609775*eta1 - 1616.9974609915555*eta2 - 3176.4950303461164*eta3 + 107980.65459735804*eta4 - 479649.75188253267*eta5 + 658866.0983367155*eta6)*S1) + chidiff1*eta5*(-1512.439342647443 + 175.59081294852444*S1 + 10.13490934572329*S2))
    return total


def IMRPhenomXHM_RD_Amp_21_rdcp2(pWF):
    """IMRPhenomXHM_RD_Amp_21_rdcp2, LALSimIMRPhenomXHM_ringdown.c:396-418, case 122022."""
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
    total = torch.abs(delta*(9.112452928978168 - 7.5304766811877455*eta1)*eta1 + chidiff1*delta*eta1*(16.236533863306132*eta1 - 500.11964987628926*eta2 + 1618.0818430353293*eta3) + chidiff1*delta*eta1*(2.7866868976718226*eta1 - 0.4210629980868266*eta2 - 20.274691328125606*eta3)*S1 + chidiff1*eta5*(-1116.4039232324135 + 245.73200219767514*S1 + 21.159179960295855*S2) + delta*eta1*S1*(-8.236485576091717*(0.8917610178208336 + 5.1501231412520285*eta1 - 87.05136337926156*eta2 + 519.0146702141192*eta3 - 997.6961311502365*eta4) + 0.2836840678615208*(-0.19281297100324718 - 57.65586769647737*eta1 + 586.7942442434971*eta2 - 1882.2040277496196*eta3 + 2330.3534917059906*eta4)*S1 + 0.40226131643223145*(-3.834742668014861 + 190.42214703482531*eta1 - 2885.5110686004946*eta2 + 16087.433824017446*eta3 - 29331.524552164105*eta4)*S2))
    return total


def IMRPhenomXHM_RD_Amp_21_rdcp3(pWF):
    """IMRPhenomXHM_RD_Amp_21_rdcp3, LALSimIMRPhenomXHM_ringdown.c:420-442, case 122022."""
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
    total = torch.abs(delta*(2.920930733198033 - 3.038523690239521*eta1)*eta1 + chidiff1*delta*eta1*(6.3472251472354975*eta1 - 171.23657247338042*eta2 + 544.1978232314333*eta3) + chidiff1*delta*eta1*(1.9701247529688362*eta1 - 2.8616711550845575*eta2 - 0.7347258030219584*eta3)*S1 + chidiff1*eta5*(-334.0969956136684 + 92.91301644484749*S1 - 5.353399481074393*S2) + delta*eta1*S1*(-2.7294297839371824*(1.148166706456899 - 4.384077347340523*eta1 + 36.120093043420326*eta2 - 87.26454353763077*eta3) + 0.23949142867803436*(-0.6931516433988293 + 33.33372867559165*eta1 - 307.3404155231787*eta2 + 862.3123076782916*eta3)*S1 + 0.1930861073906724*(3.7735099269174106 - 19.11543562444476*eta1 - 78.07256429516346*eta2 + 485.67801863289293*eta3)*S2))
    return total


def IMRPhenomXHM_RD_Amp_33_rdcp1(pWF):
    """IMRPhenomXHM_RD_Amp_33_rdcp1, LALSimIMRPhenomXHM_ringdown.c:444-467, case 122022."""
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
    total = delta*eta1*(12.439702602599235 - 4.436329538596615*eta1 + 22.780673360839497*eta2) + delta*eta1*(chidiff1*(-41.04442169938298*eta1 + 502.9246970179746*eta2 - 1524.2981907688634*eta3) + chidiff2*(32.23960072974939*eta1 - 365.1526474476759*eta2 + 1020.6734178547847*eta3)) + chidiff1*delta*eta1*(-52.85961155799673*eta1 + 577.6347407795782*eta2 - 1653.496174539196*eta3)*S1 + chidiff1*eta5*(257.33227387984863 - 34.5074027042393*chidiff2 - 21.836905132600755*S1 - 15.81624534976308*S2) + 13.499999999999998*delta*eta1*S1*(-0.13654149379906394*(2.719687834084113 + 29.023992126142304*eta1 - 742.1357702210267*eta2 + 4142.974510926698*eta3 - 6167.08766058184*eta4 - 3591.1757995710486*eta5) - 0.06248535354306988*(6.697567446351289 - 78.23231700361792*eta1 + 444.79350113344543*eta2 - 1907.008984765889*eta3 + 6601.918552659412*eta4 - 10056.98422430965*eta5)*S1)*(-3.9329308614837704 + S1)**(-1)
    return total


def IMRPhenomXHM_RD_Amp_33_rdcp2(pWF):
    """IMRPhenomXHM_RD_Amp_33_rdcp2, LALSimIMRPhenomXHM_ringdown.c:469-492, case 122022."""
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
    total = delta*eta1*(8.425057692276933 + 4.543696144846763*eta1) + chidiff1*delta*eta1*(-32.18860840414171*eta1 + 412.07321398189293*eta2 - 1293.422289802462*eta3) + chidiff1*delta*eta1*(-17.18006888428382*eta1 + 190.73514518113845*eta2 - 636.4802385540647*eta3)*S1 + delta*eta1*S1*(0.1206817303851239*(8.667503604073314 - 144.08062755162752*eta1 + 3188.189172446398*eta2 - 35378.156133055556*eta3 + 163644.2192178668*eta4 - 265581.70142471837*eta5) + 0.08028332044013944*(12.632478544060636 - 322.95832000179297*eta1 + 4777.45310151897*eta2 - 35625.58409457366*eta3 + 121293.97832549023*eta4 - 148782.33687815256*eta5)*S1) + chidiff1*eta5*(159.72371180117415 - 29.10412708633528*chidiff2 - 1.873799747678187*S1 + 41.321480132899524*S2)
    return total


def IMRPhenomXHM_RD_Amp_33_rdcp3(pWF):
    """IMRPhenomXHM_RD_Amp_33_rdcp3, LALSimIMRPhenomXHM_ringdown.c:494-517, case 122022."""
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
    total = delta*eta1*(2.485784720088995 + 2.321696430921996*eta1) + delta*eta1*(chidiff1*(-10.454376404653859*eta1 + 147.10344302665484*eta2 - 496.1564538739011*eta3) + chidiff2*(-5.9236399792925996*eta1 + 65.86115501723127*eta2 - 197.51205149250532*eta3)) + chidiff1*delta*eta1*(-10.27418232676514*eta1 + 136.5150165348149*eta2 - 473.30988537734174*eta3)*S1 + chidiff1*eta5*(32.07819766300362 - 3.071422453072518*chidiff2 + 35.09131921815571*S1 + 67.23189816732847*S2) + 13.499999999999998*delta*eta1*S1*(0.0011484326782460882*(4.1815722950796035 - 172.58816646768219*eta1 + 5709.239330076732*eta2 - 67368.27397765424*eta3 + 316864.0589150127*eta4 - 517034.11171277676*eta5) - 0.009496797093329243*(0.9233282181397624 - 118.35865186626413*eta1 + 2628.6024206791726*eta2 - 23464.64953722729*eta3 + 94309.57566199072*eta4 - 140089.40725211444*eta5)*S1)*(0.09549360183532198 - 0.41099904730526465*S1 + S2)**(-1)
    return total


def IMRPhenomXHM_RD_Amp_44_rdcp1(pWF):
    """IMRPhenomXHM_RD_Amp_44_rdcp1, LALSimIMRPhenomXHM_ringdown.c:519-543, case 122022."""
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
    S2 = S1 * S1
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = eta1*(chidiff1*delta*(-8.51952446214978*eta1 + 117.76530248141987*eta2 - 297.2592736781142*eta3) + chidiff2*(-0.2750098647982238*eta1 + 4.456900599347149*eta2 - 8.017569928870929*eta3)) + eta1*(5.635069974807398 - 33.67252878543393*eta1 + 287.9418482197136*eta2 - 3514.3385364216438*eta3 + 25108.811524802128*eta4 - 98374.18361532023*eta5 + 158292.58792484726*eta6) + eta1*S1*(-0.4360849737360132*(-0.9543114627170375 - 58.70494649755802*eta1 + 1729.1839588870455*eta2 - 16718.425586396803*eta3 + 71236.86532610047*eta4 - 111910.71267453219*eta5) - 0.024861802943501172*(-52.25045490410733 + 1585.462602954658*eta1 - 15866.093368857853*eta2 + 35332.328181283*eta3 + 168937.32229060197*eta4 - 581776.5303770923*eta5)*S1 + 0.005856387555754387*(186.39698091707513 - 9560.410655118145*eta1 + 156431.3764198244*eta2 - 1.0461268207440731e6*eta3 + 3.054333578686424e6*eta4 - 3.2369858387064277e6*eta5)*S2)
    return total


def IMRPhenomXHM_RD_Amp_44_rdcp2(pWF):
    """IMRPhenomXHM_RD_Amp_44_rdcp2, LALSimIMRPhenomXHM_ringdown.c:545-568, case 122022."""
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
    total = eta1*(chidiff1*delta*(-2.861653255976984*eta1 + 50.50227103211222*eta2 - 123.94152825700999*eta3) + chidiff2*(2.9415751419018865*eta1 - 28.79779545444817*eta2 + 72.40230240887851*eta3)) + eta1*(3.2461722686239307 + 25.15310593958783*eta1 - 792.0167314124681*eta2 + 7168.843978909433*eta3 - 30595.4993786313*eta4 + 49148.57065911245*eta5) + eta1*S1*(-0.23311779185707152*(-1.0795711755430002 - 20.12558747513885*eta1 + 1163.9107546486134*eta2 - 14672.23221502075*eta3 + 73397.72190288734*eta4 - 127148.27131388368*eta5) + 0.025805905356653*(11.929946153728276 + 350.93274421955806*eta1 - 14580.02701600596*eta2 + 174164.91607515427*eta3 - 819148.9390278616*eta4 + 1.3238624538095295e6*eta5)*S1 + 0.019740635678180102*(-7.046295936301379 + 1535.781942095697*eta1 - 27212.67022616794*eta2 + 201981.0743810629*eta3 - 696891.1349708183*eta4 + 910729.0219043035*eta5)*S2)
    return total


def IMRPhenomXHM_RD_Amp_44_rdcp3(pWF):
    """IMRPhenomXHM_RD_Amp_44_rdcp3, LALSimIMRPhenomXHM_ringdown.c:570-592, case 122022."""
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
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = eta1*(chidiff1*delta*(2.4286414692113816*eta1 - 23.213332913737403*eta2 + 66.58241012629095*eta3) + chidiff2*(3.085167288859442*eta1 - 31.60440418701438*eta2 + 78.49621016381445*eta3)) + eta1*(0.861883217178703 + 13.695204704208976*eta1 - 337.70598252897696*eta2 + 2932.3415281149432*eta3 - 12028.786386004691*eta4 + 18536.937955014455*eta5) + eta1*S1*(-0.048465588779596405*(-0.34041762314288154 - 81.33156665674845*eta1 + 1744.329802302927*eta2 - 16522.343895064576*eta3 + 76620.18243090731*eta4 - 133340.93723954144*eta5) + 0.024804027856323612*(-8.666095805675418 + 711.8727878341302*eta1 - 13644.988225595187*eta2 + 112832.04975245205*eta3 - 422282.0368440555*eta4 + 584744.0406581408*eta5)*S1)
    return total


def IMRPhenomXHM_RD_Amp_32_rdaux1(pWF):
    """IMRPhenomXHM_RD_Amp_32_rdaux1, LALSimIMRPhenomXHM_ringdown.c:594-617, case 122022."""
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
    total = chidiff2*(-4.188795724777721*eta2 + 53.39200466700963*eta3 - 131.19660856923554*eta4) + chidiff1*delta*(14.284921364132623*eta2 - 321.26423637658746*eta3 + 1242.865584938088*eta4) + S1*(-0.022968727462555794*(83.66854837403105*eta1 - 3330.6261333413177*eta2 + 77424.12614733395*eta3 - 710313.3016672594*eta4 + 2.6934917075009225e6*eta5 - 3.572465179268999e6*eta6) + 0.0014795114305436387*(-1672.7273629876313*eta1 + 90877.38260964208*eta2 - 1.6690169155105734e6*eta3 + 1.3705532554135624e7*eta4 - 5.116110998398143e7*eta5 + 7.06066766311127e7*eta6)*S1) + (4.45156488896258*eta1 - 77.39303992494544*eta2 + 522.5070635563092*eta3 - 1642.3057499049708*eta4 + 2048.333892310575*eta5)*(1 - 9.611489164758915*eta1 + 24.249594730050312*eta2)**(-1)
    return total


def IMRPhenomXHM_RD_Amp_32_rdaux2(pWF):
    """IMRPhenomXHM_RD_Amp_32_rdaux2, LALSimIMRPhenomXHM_ringdown.c:619-643, case 122022."""
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
    eta7 = eta1 * eta6
    S1 = S
    chidiff1 = chidiff
    chidiff2 = chidiff1 * chidiff1
    total = chidiff2*(-18.550171209458394*eta2 + 188.99161055445936*eta3 - 440.26516625611*eta4) + chidiff1*delta*(13.132625215315063*eta2 - 340.5204040505528*eta3 + 1327.1224176812448*eta4) + S1*(-0.16707403272774676*(6.678916447469937*eta1 + 1331.480396625797*eta2 - 41908.45179140144*eta3 + 520786.0225074669*eta4 - 3.1894624909922685e6*eta5 + 9.51553823212259e6*eta6 - 1.1006903622406831e7*eta7) + 0.015205286051218441*(108.10032279461095*eta1 - 16084.215590200103*eta2 + 462957.5593513407*eta3 - 5.635028227588545e6*eta4 + 3.379925277713386e7*eta5 - 9.865815275452062e7*eta6 + 1.1201307979786257e8*eta7)*S1) + (3.902154247490771*eta1 - 55.77521071924907*eta2 + 294.9496843041973*eta3 - 693.6803787318279*eta4 + 636.0141528226893*eta5)*(1 - 8.56699762573719*eta1 + 19.119341007236955*eta2)**(-1)
    return total

# End of Amp Parameter Space Fits

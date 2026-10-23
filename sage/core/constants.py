#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : constants.py
Description     : Short description of the file

Created on 2026-01-22 10:04:13

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


# ALL CONSTANTS
PI = 3.141592653589793238462643383279502884
# LAL_MSUN_SI: solar mass derived from the tabulated heliocentric gravitational
# parameter GM_sun (known to ~1e-10) divided by G, per LAL/IAU. With G == LAL_G_SI
# below, this makes the geometrised GM = G*MSUN/C^3 land EXACTLY on LAL_MTSUN_SI
# (4.925490947641267e-6 s); the previous value (1.988409902e30) left GM off by
# ~1.6e-8, a uniform stretch on every Mf = f*M_s mapping and the absolute strain.
MSUN = 1.988409870698050731911960804878414216e30  # kg (== LAL_MSUN_SI)
G = 6.67430e-11  # m^3 / kg / s^2  (== LAL_G_SI)
C = 299792458.0  # m / s
GM = G * MSUN / (C**3.0)  # s  (== LAL_MTSUN_SI = 4.925490947641267e-6)
EulerGamma = 0.577215664901532860606512090082402431
Mpc = 3.085677581491367278913937957796471611e22  # m
SIDEREAL_DAY = 86164.09053083288  # s
TWOPI = 2.0 * PI

# ALL METADATA
CONST_METADATA = {
    "PI": "Constant ratio of a circle's circumference to its diameter",
    "MSUN": "1 solar mass in Kg",
    "G": "Newton's gravitational constant in m^3 / Kg / s^2",
    "C": "Speed of light in m/s",
    "GM": "Geometrised mass G * MSUN / C^3 in s",
    "EulerGamma": "Euler–Mascheroni constant",
    "Mpc": "1 Megaparsec in m",
    "SIDEREAL_DAY": "1 Earth rotation relative to the stars in s",
    "TWOPI": "This is PI, but twice!",
}

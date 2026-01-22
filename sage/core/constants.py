#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : constants.py
Description     : Short description of the file

Created on 2026-01-22 10:04:13

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


# ALL CONSTANTS
PI = 3.141592653589793238462643383279502884
MSUN = 1.988409902147041637325262574352366540e30  # kg
G = 6.67430e-11  # m^3 / kg / s^2
C = 299792458.0  # m / s
GM = G * MSUN / (C**3.0)  # s
EulerGamma = 0.577215664901532860606512090082402431
Mpc = 3.085677581491367278913937957796471611e22  # m

# ALL METADATA
CONST_METADATA = {
    "PI": "Constant ratio of a circle's circumference to its diameter",
    "MSUN": "1 solar mass in Kg",
    "G": "Newton's gravitational constant in m^3 / Kg / s^2",
    "C": "Speed of light in m/s",
    "GM": "Geometrised mass G * MSUN / C^3 in s",
    "EulerGamma": "Euler–Mascheroni constant",
    "Mpc": "1 Megaparsec in m",
}

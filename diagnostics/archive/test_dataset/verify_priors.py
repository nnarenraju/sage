# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Dec 17 15:33:53 2021

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# IN-BUILT
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})

def _figure(name):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, 1, figsize=(22.0, 22.0))
    fig.suptitle(f"{name}", fontsize=20, y=0.95)
    return axs

def _plot(ax, data, xlabel="unknown", c=None, save_dir=None):
    ax.hist(data, bins='auto', color=c)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Occurences")
    plt.savefig(os.path.join(save_dir, f"{xlabel}_verify_prior.png"))
    plt.close()

def verify(injection_file, check):
    """
    Verify the prior distributions used to create dataset with observed distribution
    Plotting all relevant plots alongside their intended priors

    Parameters
    ----------
    injection_file : str
        Absolute location of injections file of given dataset
    check : dict
        contains - save_dir, tc_llimit, tc_ulimit, segment_length, gap

    Returns
    -------
    None.

    """
    
    with h5py.File(injection_file, "r") as foo:
        # All Fields in injtable
        fields = list(foo.keys())
        
        # Plot the prior distributions as histograms
        for field in fields:
            # Adding labels for known priors
            if field == "chirp_distance":
                name = f"{field} with Uniform Prior " + r"dc^2 = (130^2, 350^2) Mpc^2"
            elif field == "ra":
                name = f"{field} with Uniform Prior " + r"$\phi$ = (-$\pi$, +$\pi$)"
            elif field == "dec":
                name = f"{field} with Uniform Prior " + r"sin$\theta$ = (-1, +1)"
            elif field == "inclination":
                name = f"{field} with Uniform Prior " + r"cos$\iota$ = (-1, +1)"
            elif field == "polarization":
                name = f"{field} with Uniform Prior " + r"$\psi$ = (0, 2$\pi$)"
            elif field == "coa_phase":
                name = f"{field} with Uniform Prior " + r"$\Phi_0$ = (0, 2$\pi$)"
            elif field == "mass1":
                name = f"{field} with Uniform Prior " + r"m_1 = (7.0, 50.0)"
            elif field == "mass2":
                name = f"{field} with Uniform Prior " + r"m_2 = (7.0, 50.0)"
            elif field == "spin1_a":
                name = f"{field} with Uniform Prior " + r"spin1_a = (0.0, 0.99)"
            elif field == "spin2_a":
                name = f"{field} with Uniform Prior " + r"spin2_a = (0.0, 0.99)"
            else:
                name = f"{field} with undetermined prior"
            
            # Set figure
            ax = _figure(name)
            # Plotting
            _plot(ax, np.array(foo[field]), xlabel=field, c="r", save_dir=check['save_dir'])
            
        # Distance b/w adjacent 'tc'
        # See https://github.com/gwastro/ml-mock-data-challenge-1/issues/8
        sorted_tc = np.sort(foo['tc'])
        diff = np.sort(sorted_tc[1:] - sorted_tc[:-1])
        worst_diff = check['segment_length']-check['tc_ulimit']+check['gap']+check['tc_llimit']
        if min(diff) < worst_diff:
            raise ValueError(f"Minimum difference b/w adjacent 'tc' is lower than expected\n \
                             Expected = {worst_diff}, Observed = {min(diff)}")

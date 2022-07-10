#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Apr 21 19:41:42 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plotter(title, xlabel, ylabel, hists, labels, save_path):
    plt.figure(figsize=(9.0, 6.0))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for data, label in zip(hists, labels):
        plt.hist(data, bins=100, alpha=0.8, label=label)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def debug_plotter(debug_dir):

    distance_new = np.loadtxt(debug_dir + '/distance_new.txt')
    distance_old = np.loadtxt(debug_dir + '/distance_old.txt')
    title = 'Comparing the old and new distance priors'
    xlabel = 'Distance'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'compare_distance.png')
    plotter(title, xlabel, ylabel, [distance_old, distance_new], ['Original', 'Augmented'], save_path)
    
    pol = np.loadtxt(debug_dir + '/pol.txt')
    title = 'Polarisation for 20 epochs of approx 1e4 signals each'
    xlabel = 'Polarisation Angle'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'pol_angle.png')
    plotter(title, xlabel, ylabel, [pol], ['Pol Angle'], save_path)
    
    ra = np.loadtxt(debug_dir + '/ra.txt')
    title = 'RA for 20 epochs of approx 1e4 signals each'
    xlabel = 'RA'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'ra.png')
    plotter(title, xlabel, ylabel, [ra], ['ra'], save_path)
    
    dec = np.loadtxt(debug_dir + '/dec.txt')
    title = 'DEC for 20 epochs of approx 1e4 signals each'
    xlabel = 'DEC'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'dec.png')
    plotter(title, xlabel, ylabel, [dec], ['dec'], save_path)
    
    noise_slide_1 = np.loadtxt(debug_dir + '/noise_slide_1.txt')
    title = 'Noise Slide 1 for 20 epochs of approx 1e4 signals each'
    xlabel = 'Noise Slide 1'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'noise_slide_1.png')
    plotter(title, xlabel, ylabel, [noise_slide_1], ['noise_slide_1'], save_path)
    
    noise_slide_2 = np.loadtxt(debug_dir + '/noise_slide_2.txt')
    title = 'Noise Slide 2 for 20 epochs of approx 1e4 signals each'
    xlabel = 'Noise Slide 2'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'noise_slide_2.png')
    plotter(title, xlabel, ylabel, [noise_slide_2], ['noise_slide_2'], save_path)
    
    dchirp = np.loadtxt(debug_dir + '/dchirp.txt')
    title = 'Chirp Distance 2 for 20 epochs of approx 1e4 signals each'
    xlabel = 'Chirp Distance'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'dchirp.png')
    plotter(title, xlabel, ylabel, [dchirp], ['dchirp'], save_path)

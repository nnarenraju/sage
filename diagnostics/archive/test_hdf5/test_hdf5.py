# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Dec  4 12:40:41 2021

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence

Github Repository: NULL

"""

# Modules
import h5py
import numpy as np
import pandas as pd

directory = ""

foreground_path = directory + "/background.hdf"
background_path = directory + "/background.hdf"
injections_path = directory + "/injections.hdf"
segments_path = directory + "/segments.csv"


def get_data(paths):
    # Get information about foreground and background
    for file_name, file_path in paths.items():
        with h5py.File(file_path, "r") as foo:
            # Attributes of file
            print("\n================ START ================\n")
            print(f"{file_name} Metadata:")
            for attr in list(foo.attrs.keys()):
                print(f"{attr}: {foo.attrs[attr]}")
            
            print(f"\n{file_name} Data Details")
            # Detectors
            print("Detectors: {} and {}".format(*list(foo.keys())))
            H1, L1 = list(foo.keys())
            detector_group = foo[H1]
            # Time Segments
            print("Time segments: {}".format(list(detector_group.keys())))
            time_segments = list(detector_group.keys())
            time_group = detector_group[time_segments[0]]
            # Segment Attributes
            print("Segment Attributes: {}".format(list(time_group.attrs.keys())))
            time_attrs = list(time_group.attrs.keys())
            for attr in time_attrs:
                print("'{}' attr = {}".format(attr, time_group.attrs[attr]))
            # Get Dataset
            data = np.array(time_group)
            print("Length of dataset = {} at 2048 Hz sampling rate".format(len(data)/2048.0))
            print("\n================ EOF ================\n\n")
            
def get_injections(path):
    # Get information about added injections
    file_name, file_path = list(path.items())[0]
    with h5py.File(file_path, "r") as foo:
        # Attributes of file
        print("\n================ START ================\n")
        print(f"{file_name} Metadata:")
        for attr in list(foo.attrs.keys()):
            print(f"{attr}: {foo.attrs[attr]}")
        
        print("\nGroups = {}".format(list(foo.keys())))
        lendata = len(np.array(foo['tc']))
        
        print(np.array(foo['tc']))
        print(f"\n{file_name} Data - sample from list of length {lendata}\n")
        for field in list(foo.keys()):
            print(f"{field} = {foo[field][0]}")
        print("\nTotal number of injections present = {}".format(lendata))
        print("\nDifferences between times of coalescence in injections (first 5):")
        sorted_tc = np.sort(foo['tc'])
        diff = np.sort(sorted_tc[1:] - sorted_tc[:-1])
        for n, tdiff in enumerate(diff[:150]):
            print(f"tc diff b/w signal {n+2} and {n+1} is {tdiff}")
        
        print("\n================ EOF ================\n\n")

def get_segments(path):
    # Analyse the segments.csv data
    file_name, file_path = list(path.items())[0]
    data = pd.read_csv(file_path)
    # Distance between segments
    start = np.array(data['start'][1:])
    end = np.array(data['end'][:-1])
    between = start - end
    # Distance within segments
    start = np.array(data['start'])
    end = np.array(data['end'])
    diff = end - start
    print("Analysing segments.csv")
    print("\nTotal number of segments = {}".format(len(data)))
    print("Maximum duration of segments = {} seconds".format(max(diff)))
    print("Minimum duration of segments = {} seconds".format(min(diff)))
    print("Maximum distance b/w segments = {} seconds".format(max(between)))
    print("Minimum distance b/w segments = {} seconds".format(min(between)))


if __name__ == "__main__":
    
    data_paths = {'Foregound':foreground_path, 'Background':background_path}
    inj_path = {'Injection': injections_path}
    seg_path = {'Segments':segments_path}
    # Get Metadata
    get_data(data_paths)
    get_injections(inj_path)
    get_segments(seg_path)

# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Dec 11 20:25:55 2021

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
import numpy as np

class Segments:
    """
    
    """
    
    def __init__(self):
        # Times
        self.GPS_start_time = 0.0
        self.duration = 2.0e+5
        # No correlation b/w segments
        self.gap = 1.0
        # Segment details
        # equal_segments: duration is a soft limit, segment_length is strict
        # unequal_segments: duration is hard limit, segment_length is soft
        self.equal_segments = True
        # For equal_segments==True
        self.segment_length = 20.0
        # For equal_segments==False
        self.unequal_segments_llimit = 10.0
        self.unequal_segments_ulimit = 30.0
        # Output
        self.output_filepath = "./segments.csv"
    
    def make_segments(self):
        # Make segments CSV file used by ligo.segments
        # Necessary fields = (id, GPS_start_time, GPS_end_time)
        if self.equal_segments:
            # Consider nsegments as lower limit
            offset = self.duration%self.segment_length
            start_time = self.GPS_start_time
            end_time = self.GPS_start_time + self.duration + offset
            nsegments = int((self.duration+offset)//self.segment_length)
            # Last point will be end time (ignore)
            start_times = np.linspace(start_time, end_time, nsegments, endpoint=False)
            gap = np.full(len(start_times[1:]), self.gap)*np.arange(1, len(start_times[1:])+1)
            start_times = np.concatenate(([start_times[0]], start_times[1:] + gap))
            start_times = start_times.astype(np.int16)
            # End-points
            end_times = (start_times + self.segment_length).astype(np.int16)
            segments = np.column_stack((range(len(start_times)), start_times, end_times))
            #print(max(segments.flatten()))
            np.savetxt("segments.csv", segments, delimiter=",", fmt="%d")
        else:
            raise ValueError("make_segments: unequal lengths under construction!")

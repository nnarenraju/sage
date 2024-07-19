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
import os
import numpy as np
    
def make_segments(GPS_start_time: int,
                  gap: float,
                  segment_length: float,
                  ninjections: int,
                  output_filepath: str,
                  force: bool) -> None:

    """
    Description
    -----------
    Make segments CSV file used by ligo.segments
    Necessary fields = (id, GPS_start_time, GPS_end_time)
    
    Parameters
    ----------
    
    
    
    Returns
    -------
    None.
    
    """
    duration = segment_length*ninjections
    # Consider nsegments as lower limit
    offset = duration%segment_length
    start_time = GPS_start_time
    end_time = GPS_start_time + duration + offset
    nsegments = int((duration+offset)//segment_length)
    # Last point will be end time (ignore)
    start_times = np.linspace(start_time, end_time, nsegments, endpoint=False)
    gap = np.full(len(start_times[1:]), gap)*np.arange(1, len(start_times[1:])+1)
    start_times = np.concatenate(([start_times[0]], start_times[1:] + gap))
    start_times = start_times.astype(np.int16)
    # End-points
    end_times = (start_times + segment_length).astype(np.int16)
    segments = np.column_stack((range(len(start_times)), start_times, end_times))
    # Savedata (segment.csv)
    if force:
        if os.path.exists(output_filepath):
            print(f"--force: {output_filepath} exists. Overwriting.")
            os.remove(output_filepath)
    
    np.savetxt(output_filepath, segments, delimiter=",", fmt="%d")

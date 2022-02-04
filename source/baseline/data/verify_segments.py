# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Dec 17 15:33:01 2021

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
import glob
import h5py
import numpy as np

# NOTE: Temporary global variable to store the global absolute maximum
abs_max_1 = 0.0
abs_max_2 = 0.0


def _common_(gname, gfile, check):
    # All common verifications between foreground and background files
    ## Reading all data parameters
    # Detectors
    dets = list(gfile.keys())
    # Groups within detectors (times as dict)
    detector_group_1 = gfile[dets[0]]
    detector_group_2 = gfile[dets[1]]
    # Times as list
    times_1 = list(detector_group_1.keys())
    times_2 = list(detector_group_2.keys())
    # Noise data within each detector
    data_1 = detector_group_1[times_1[0]]
    data_2 = detector_group_2[times_2[0]]
    
    
    # NOTE: Check the absolute maximum of the given data in both detectors
    global abs_max_1
    global abs_max_2
    current_max_1 = np.max(np.abs(data_1))
    current_max_2 = np.max(np.abs(data_2))
    if current_max_1 > abs_max_1:
        abs_max_1 = current_max_1
    if current_max_2 > abs_max_2:
        abs_max_2 = current_max_2
    
    attrs = dict(gfile.attrs)
    # [7] Check dataset type of given bg file
    if attrs['dataset'] != check['dataset_type']:
        raise NameError(f"{gname} does not belong to correct dataset type!")
    # [16] Data should be a two detector configuration
    assert len(dets) == 2
    # [17] Number of segments is same and one for training data
    if len(times_1) != 1 or len(times_2) != 1:
        raise ValueError(f"{gname} has more than one segment stored!\n \
                         Expected=1, Observed={times_1} and {times_2}")
    # [8] Segment time of detectors
    if times_1 != times_2:
        raise ValueError(f"{gname} contains different segtimes for detectors!")
    # [10] All segments should have the same seed
    if attrs['seed'] != check['seed']:
        raise ValueError(f"{gname} does not have the correct seed!\n \
                         Expected = {check['seed']}, observed = {attrs['seed']}")
    # [3] All segments should have the same unique dataset ID
    if attrs['unique_dataset_id'] != check['uqid']:
        raise ValueError(f"{gname} does not have expected unique dataset id")
    # [2] Sample rate check
    if attrs['sample_rate'] != check['sample_rate']:
        raise ValueError(f"{gname} does not have the correct sample rate\n \
                         Expected = {check['sample_rate']}, observed = {attrs['sample_rate']}")
    # [1] Verify the length of the segment
    expected_length = check['segment_length']*check['sample_rate']
    obs_length_D1 = len(data_1)
    obs_length_D2 = len(data_2)
    if obs_length_D1 != expected_length or obs_length_D2 != expected_length:
        raise ValueError(f"{gname} segment length is not as expected\n \
                         Expected = {expected_length}\n \
                         Observed = {obs_length_D1} and {obs_length_D2}")
    
    # Returning data for bg/fg specific checks
    return (data_1, data_2), times_1[0]

def _check_overlap(segs, name):
    # Check overlap between segments
    for n, (start, end) in enumerate(segs[:-1]):
        x = range(start, end)
        y = range(segs[n+1][0], segs[n+1][1])
        # If no overlap, len == 0
        if len(range(max(x[0], y[0]), min(x[-1], y[-1])+1)) != 0:
            raise ValueError(f"There is overlap between one or more segments in {name} files")

def verify(dirs, check):
    """
    Verification
    ------------
    1. Length/duration of the segment in each foreground and background file (FIN) \n
    2. Sample rate of each of the segments (FIN) \n
    3. All segments should have the same unique dataset ID (FIN) \n
    4. Foreground segment should be connected to the correct background segment (FIN) \n
    5. Presence of all requested segments via segments.csv (FIN) \n
    6. Collisions/overlap between two segments based on start and end times (FIN) \n
    7. All segments should be from the same dataset type (FIN) \n
    8. Segment time of timeseries from both detectors should be the same (FIN) \n
    9. All segments should use the same random seed (FIN) \n
    10. Distance b/w adjacent segments should be verified (FIN) \n
    11. Check whether h5py version in venv is *NOT* v3.4.0 (memory leak issue) (FIN) \n
    12. Check whether the noise generated is different for different detectors (FIN) \n
    13. Same number of background and foreground files (FIN) \n
    14. Assert that files should be of .hdf format (FIN) \n
    15. Data should contain ts of two detectors exactly (FIN) \n
    16. Number of segments is same and one for training data in given file (FIN)
    
    Parameters
    ----------
    dirs : dict
        Foreground and background directory paths
    check : dict
        contains - dataset_type, seed, uqid, sample_rate, segment_length, segments_path, gap
    
    Returns
    -------
    None.

    """
    
    # [12] Check the h5py version for memory leak issue
    if h5py.__version__ == "3.4.0":
        raise ValueError("System uses h5py version 3.4.0.\n \
                         See issue https://github.com/gwastro/ml-mock-data-challenge-1/issues/18")
    
    # Read all bg and fg files using glob
    bgfiles = glob.glob(dirs['background'] + "/background_*")
    fgfiles = glob.glob(dirs['foreground'] + "/foreground_*")
    # [15] Same number of background and foreground files
    if len(bgfiles) != len(fgfiles):
        msg =  "Number of foreground files not equal to number of background files\n"
        msg += "generate_data script should produce an equal number of both\n"
        msg += "This is asserted for metadata check. The ML pipeline can use an unequal number."
        raise ValueError(msg)
    
    global abs_max_1
    global abs_max_2
    
    times_bg = []
    times_fg = []
    for n, (bgname, fgname) in enumerate(zip(bgfiles, fgfiles)):
        # Read background and foreground data segments
        _, bg_extension = os.path.splitext(bgname)
        _, fg_extension = os.path.splitext(fgname)
        # [16] All segment files should be HDF5 format
        assert bg_extension == ".hdf"
        assert fg_extension == ".hdf"
        
        # Get all data from HDF5 dataset
        with h5py.File(bgname, 'r') as bgfile:
            (noise_1, noise_2), segtime_bg = _common_(bgname, bgfile, check) 
            # [13] Check whether noise generated is different b/w detectors
            # Tolerance should be << than typical GW strain
            if np.allclose(noise_1, noise_2, rtol=1e-46, atol=1e-46):
                raise ValueError(f"Noise b/w two detectors in {bgname} is the same!")
            # [5] Storing times to check for requested segments
            times_bg.append(segtime_bg)
        
        with h5py.File(fgname, 'r') as fgfile:
            (signal_1, signal_2), segtime_fg = _common_(fgname, fgfile, check)
            # [4] Foregound file should be connected to the correct background file
            attrs_fg = dict(fgfile.attrs)
            expected_bgfile = f"background_{n}.hdf"
            if attrs_fg['background-file'] == expected_bgfile:
                raise ValueError(f"Foreground file {fgname} not connected to correct bgfile!")
            # [5] Storing times to check for requested segments
            times_fg.append(segtime_fg)
    
    print(abs_max_1, abs_max_2)
    
    # Converting all lists to np arrays for convenience
    times_bg = np.sort(np.array(times_bg, dtype=np.int32))
    times_fg = np.sort(np.array(times_fg, dtype=np.int32))
    # [5] Checking for requested segments in segment.csv
    segdata = np.loadtxt(check['segments_path'], delimiter=",")
    # fields: (idx, start_times, end_times)
    start_times = segdata[:,1]
    start_times = np.sort(start_times.astype(np.int32))
    
    if len(times_bg) != len(start_times) or len(times_fg) != len(start_times):
        raise ValueError("Total number of segments observed not the same as in segments.csv")
    if not np.allclose(times_bg, start_times):
        raise ValueError("Times in background files is not the same as segments.csv")
    if not np.allclose(times_fg, start_times):
        raise ValueError("Times in foreground files is not the same as segments.csv")
    
    # [6] Check for overlap between segments
    end_times_fg = times_fg + check['segment_length']
    end_times_bg = times_bg + check['segment_length']
    # Since times are always integers, we can use range to check overlap
    segs = list(zip(times_fg, end_times_fg))
    _check_overlap(segs, "foreground")
    segs = list(zip(times_bg, end_times_bg))
    _check_overlap(segs, "background")
    
    # [10] Checking the gap between different segments
    gap_fg = times_fg[1:] - end_times_fg[:-1]
    gap_bg = times_bg[1:] - end_times_bg[:-1]
    if np.count_nonzero(gap_fg==gap_fg[0]) != len(gap_fg) or gap_fg[0] != check['gap']:
        raise ValueError("Foreground gap b/w segments not equal OR gap value is unexpected")
    if np.count_nonzero(gap_bg==gap_bg[0]) != len(gap_bg) or gap_bg[0] != check['gap']:
        raise ValueError("Background gap b/w segments not equal OR gap value is unexpected")

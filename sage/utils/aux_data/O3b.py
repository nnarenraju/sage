import os
import sys
import glob
import time
import h5py
import argparse
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import signal

from gwpy.timeseries import TimeSeries
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.types import TimeSeries as TS

from pycbc import DYN_RANGE_FAC

# Download files
import urllib.request


def get_sage_abspath():
    # Get Sage abspath
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')
    return repo_abspath

def downsample(strain, sample_rate=2048., crop=2.5, noise_low_freq_cutoff=15.0):
    res = resample_to_delta_t(strain, 1./sample_rate)
    ret = highpass(res, noise_low_freq_cutoff).astype(np.float32)
    ret = ret.time_slice(float(ret.start_time) + crop,
                         float(ret.end_time) - crop)
    return ret

def get_detector_noise(args):
    n, left_boundary, right_boundary, detector = args
    success = False
    while not success:
        try:
            noise_data = TimeSeries.fetch_open_data(detector, left_boundary, right_boundary, cache=1)
            success = True
        except:
            duration = right_boundary - left_boundary
            left_boundary += 0.1 * duration
            right_boundary -= 0.1 * duration
            if (right_boundary - left_boundary) < 3600:
                break

    if success:
        data = TS(noise_data.value, delta_t=noise_data.dt.value)
        data = downsample(data).numpy() * DYN_RANGE_FAC
        data = data.astype(np.float32)
        return (n, data)
    else:
        return (n, None)

def fetcher(GPS_boundaries, num_workers=4, parent_dir="", detector=""):
    detector_noise = []
    print('Fetching GWOSC data for detector {} using {} cores'.format(detector, num_workers))
    # Download data at each GPS range
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(GPS_boundaries)) as pbar:
            pbar.set_description("MP-DET_NOISE GWOSC")
            for out in pool.imap_unordered(get_detector_noise, [(n, foo[0], foo[1], detector) for n, foo in enumerate(GPS_boundaries)]):
                n, data = out
                if isinstance(data, np.ndarray):
                    with h5py.File('{}/{}/noise_{}_O3b_chunk_{}.hdf'.format(parent_dir, detector, detector, n), 'a') as hf:
                        hf.create_dataset('data', data=data, compression="gzip", chunks=True)
                pbar.update()

def get_valid_segments(all_events_GPS, det_O3b_ranges, event_buffer, minimum_segment_duration):
    det_O3b_GPS_boundaries = []

    for n, (det_O3b_start, det_O3b_end) in enumerate(det_O3b_ranges):
        if det_O3b_end - det_O3b_start <= minimum_segment_duration:
            continue 
        event_idxs = np.array([det_O3b_start <= GPS_time <= det_O3b_end for GPS_time in all_events_GPS]).nonzero()[0]
        if len(event_idxs):
            print('Event present between GPS times {} and {} with duration = {} s'.format(det_O3b_start, det_O3b_end, det_O3b_end-det_O3b_start))
            # All event times present in the given segment
            invalid_GPS_times = all_events_GPS[event_idxs]
            # Removing 30 seconds on either side of an event
            windows = [[iGPStime-event_buffer, iGPStime+event_buffer] for iGPStime in invalid_GPS_times]

            if len(windows) > 1:
                # combine overlapping windows and remove bad boundaries
                # Remember: the event GPS sorted are sorted
                boundaries = [det_O3b_start]
                for n in np.arange(1, len(windows)):
                    # if left boundary of first window is greater than segment start time
                    # Segment start time already added to combined windows
                    if n==1 and windows[n-1][0] > det_O3b_start:
                        boundaries.append(windows[n-1][0])
                    # if left boundary of current window overlaps with right boundary of previous window
                    if windows[n][0] > windows[n-1][1]:
                        boundaries.extend([windows[n-1][1], windows[n][0]])
                    # if right boundary of last window is less than segment end time
                    if n == len(windows)-1 and windows[n][1] < det_O3b_end:
                        boundaries.append(windows[n][1])
            else:
                boundaries = [det_O3b_start]
                if windows[0][0] > det_O3b_start:
                    boundaries.append(windows[0][0])
                if windows[0][1] < det_O3b_end:
                    boundaries.append(windows[0][1])
            
            # Add segment boundary
            boundaries.append(det_O3b_end)
            boundaries = np.array(boundaries)

            # Check if any mini-segment within the given segment is below a threshold in duration
            # Every two subsequent boundaries must now be valid minisegments within the given segment
            # There should always be an equal number of boundaries within the variable (if it works as intended)
            assert len(boundaries) % 2 == 0
            split_boundaries = np.array_split(boundaries, len(boundaries) / 2)
            minimum_minisegment_duration = 60 # seconds
            split_boundaries = np.array([foo for foo in split_boundaries if foo[1]-foo[0] > minimum_minisegment_duration])

            # All valid boundaries
            det_O3b_GPS_boundaries.extend(split_boundaries)
        
        else:
            # If there are no events in the given segment
            boundary = np.array([H1_O3b_start, H1_O3b_end])
            det_O3b_GPS_boundaries.append(boundary)

    total_valid_duration = 0
    for boundary in det_O3b_GPS_boundaries:
        total_valid_duration += boundary[1]-boundary[0]
    print("Total valid duration = {} days".format(total_valid_duration/(3600*24)))
    return det_O3b_GPS_boundaries

def _sanity_check(all_events_GPS, det_O3b_GPS_boundaries):
    # Sanity check to see if any events made its way through
    for n, (det_O3b_start, det_O3b_end) in enumerate(det_O3b_GPS_boundaries):
        event_idxs = np.array([det_O3b_start <= GPS_time <= det_O3b_end for GPS_time in all_events_GPS]).nonzero()[0]
        if len(event_idxs):
            raise ValueError('GWOSC event present within noise segment!')

def get_o3b_data(raw_args=None):

    # Args
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers used to retrieve O3b data from GWOSC")
    parser.add_argument("--event-buffer", type=int, default=30,
                        help="Seconds before and after event time to ignore")
    parser.add_argument("--minimum-segment-duration", type=int, default=3600,
                        help="Minimum noise segment duration allowed from O3b")
    parser.add_argument('--data-dir', type=str, required=True,
                        help=("Directory to save all downloaded data (approx 140 GB)"))
    
    opts = parser.parse_args(raw_args)

    repo_abspath = get_sage_abspath()
    tmp_save_dir = os.path.join(repo_abspath, 'tmp')

    # Make directories if needed
    h1_data_folder = os.path.join(opts.data_dir, 'H1')
    l1_data_folder = os.path.join(opts.data_dir, 'L1')
    if not os.path.exists(h1_data_folder):
        os.makedirs(h1_data_folder)
    if not os.path.exists(l1_data_folder):
        os.makedirs(l1_data_folder)
    
    # Link: https://gwosc.org/eventapi/html/allevents/
    # Getting ASCII file of all events provided in GWOSC (Last verified: July 19th, 2024)
    event_list = os.path.join(tmp_save_dir, "all_events_list.txt")
    urllib.request.urlretrieve("https://gwosc.org/eventapi/ascii/allevents/", event_list)
    # Read all events
    all_events_list = np.genfromtxt(event_list, skip_header=1)

    # Link: https://gwosc.org/timeline/show/O3b_4KHZ_R1/H1_DATA*L1_DATA*V1_DATA/1256655618/12708000/
    # DATA flag was included and none of the flags {CBC_CAT1, CBC_CAT2, CBC_HW_INJ, or BURST_HW_INJ} were included
    # Data collected only for O3b (Last verified: July 19th, 2024)
    H1_O3b_path = os.path.join(tmp_save_dir, "H1_O3b_GPS.txt")
    urllib.request.urlretrieve("https://gwosc.org/timeline/segments/O3b_4KHZ_R1/H1_DATA/1256655618/12708000/", H1_O3b_path)
    L1_O3b_path = os.path.join(tmp_save_dir, "L1_O3b_GPS.txt")
    urllib.request.urlretrieve("https://gwosc.org/timeline/segments/O3b_4KHZ_R1/L1_DATA/1256655618/12708000/", L1_O3b_path)
    # Read O3b segment data
    H1_O3b_GPS = np.loadtxt(H1_O3b_path)
    L1_O3b_GPS = np.loadtxt(L1_O3b_path)

    all_events_GPS = np.sort(all_events_list[:,4])
    H1_O3b_ranges = list(zip(H1_O3b_GPS[:,0], H1_O3b_GPS[:,1]))
    L1_O3b_ranges = list(zip(L1_O3b_GPS[:,0], L1_O3b_GPS[:,1]))

    # Get valid noise segments for H1
    H1_O3b_GPS_boundaries = get_valid_segments(all_events_GPS, H1_O3b_ranges, opts.event_buffer, opts.minimum_segment_duration)
    _sanity_check(all_events_GPS, H1_O3b_GPS_boundaries)
    # Get valid noise segments for L1
    L1_O3b_GPS_boundaries = get_valid_segments(all_events_GPS, L1_O3b_ranges, opts.event_buffer, opts.minimum_segment_duration)
    _sanity_check(all_events_GPS, L1_O3b_GPS_boundaries)

    # Get all valid O3b data for the H1 detector
    fetcher(H1_O3b_GPS_boundaries, num_workers=opts.num_workers, parent_dir=h1_data_folder, detector='H1')
    # Get all valid O3b data for the L1 detector
    fetcher(L1_O3b_GPS_boundaries, num_workers=opts.num_workers, parent_dir=l1_data_folder, detector='L1')

    ## Save durations of each segment in O3b
    # Downloaded noise data
    noise_files_H1_O3b = [h5py.File(fname) for fname in glob.glob(os.path.join(h1_data_folder, "*.hdf"))]
    duration_H1_O3b = [len(np.array(hf['data'][:])) for hf in noise_files_H1_O3b]
    np.save(os.path.join(tmp_save_dir, "durs_H1_O3b_all_noise.npy"), duration_H1_O3b)

    noise_files_L1_O3b = [h5py.File(fname) for fname in glob.glob(os.path.join(l1_data_folder, "*.hdf"))]
    duration_L1_O3b = [len(np.array(hf['data'][:])) for hf in noise_files_L1_O3b]
    np.save(os.path.join(tmp_save_dir, "durs_L1_O3b_all_noise.npy"), duration_L1_O3b)


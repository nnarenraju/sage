import pycbc
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector

import numpy as np

signal_length = 15.0
sample_length_in_s = 20.0
whiten_padding = 5.0
signal_low_freq_cutoff = 20.0
sample_rate = 2048.
error_padding_in_s = 0.5
error_padding_in_num = int(error_padding_in_s * sample_rate)
sample_length_in_num = int(sample_length_in_s * sample_rate)


def optimise_fmin(h_pol, waveform_kwargs):
    # Use self.waveform_kwargs to calculate the fmin for given params
    # Such that the length of the sample is atleast 20s by the time it reaches fmin
    # This DOES NOT mean we produce signals that are exactly 20s long
    current_start_time = -1*h_pol.get_sample_times()[0]
    req_start_time = signal_length - h_pol.get_sample_times()[-1]
    fmin = signal_low_freq_cutoff*(current_start_time/req_start_time)**(3./8.)
    
    while True:
        # fmin_new is the fmin required for the current params to produce 20.0s signal
        waveform_kwargs['f_lower'] = fmin
        h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)
        # Sanity check to verify the new signal length
        new_signal_length = len(h_plus)/2048.
        if new_signal_length > signal_length:
            break
        else:
            fmin = fmin - 3.0
        
    # Return new signal
    return h_plus, h_cross


def get_ht(h_plus, h_cross, waveform_kwargs):
    # If the signal is smaller than 20s, we change fmin such that it is atleast 20s
    if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] < signal_length:
        # Pass h_plus or h_cross
        h_plus, h_cross = optimise_fmin(h_plus)

    # If it is longer than signal_length, slice out the required region
    if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] > signal_length and False:
        new_end = h_plus.get_sample_times()[-1]
        new_start = -1*(signal_length - new_end)
        h_plus = h_plus.time_slice(start=new_start, end=new_end)
        h_cross = h_cross.time_slice(start=new_start, end=new_end)

    ## Properly time and project the waveform
    start_time = waveform_kwargs['injection_time'] + h_plus.get_sample_times()[0]
    end_time = waveform_kwargs['injection_time'] + h_plus.get_sample_times()[-1]
    
    # Calculate the number of zeros to append or prepend
    # Whitening padding will be corrupt and removed in whiten transformation
    start_samp = waveform_kwargs['tc'] + (whiten_padding/2.0)
    start_interval = waveform_kwargs['injection_time'] - start_samp
    # subtract delta value for length error (0.001 if needed)
    end_padding = whiten_padding/2.0
    post_merger = signal_length - waveform_kwargs['tc']
    end_interval = waveform_kwargs['injection_time'] + post_merger + end_padding
    
    # Calculate the difference (if any) between two time sets
    diff_start = start_time - start_interval
    diff_end = end_interval - end_time
    # Convert num seconds to num samples
    diff_end_num = int(diff_end * sample_rate)
    diff_start_num = int(diff_start * sample_rate)
    
    expected_length = ((end_interval-start_interval) + error_padding_in_s*2.0) * sample_rate
    observed_length = len(h_plus) + (diff_start_num + diff_end_num + error_padding_in_num*2.0)
    diff_length = expected_length - observed_length
    if diff_length != 0:
        diff_end_num += diff_length

    # If any positive difference exists, add padding on that side
    # Pad h_plus and h_cross with zeros on both end for slicing
    if diff_end > 0.0:
        # Append zeros if we need samples after signal ends
        h_plus.append_zeros(int(diff_end_num + error_padding_in_num))
        h_cross.append_zeros(int(diff_end_num + error_padding_in_num))
    
    if diff_start > 0.0:
        # Prepend zeros if we need samples before signal begins
        # prepend_zeros arg must be an integer
        h_plus.prepend_zeros(int(diff_start_num + error_padding_in_num))
        h_cross.prepend_zeros(int(diff_start_num + error_padding_in_num))

    elif diff_start < 0.0:
        h_plus = h_plus.crop(left=-1*((diff_start_num + error_padding_in_num)/2048.), right=0.0)
        h_cross = h_cross.crop(left=-1*((diff_start_num + error_padding_in_num)/2048.), right=0.0)

    #assert len(h_plus) == sample_length_in_num + error_padding_in_num*2.0
    #assert len(h_cross) == sample_length_in_num + error_padding_in_num*2.0
    
    # Setting the start_time, sets epoch and end_time as well within the TS
    # Set the start time of h_plus and h_plus after accounting for prepended zeros
    h_plus.start_time = start_interval - error_padding_in_s
    h_cross.start_time = start_interval - error_padding_in_s
    # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    dets = [det_h1, det_l1]
    strains = [det.project_wave(h_plus, h_cross, waveform_kwargs['ra'], waveform_kwargs['dec'], 
                                waveform_kwargs['polarization'], method='constant') for det in dets]
    # Put both strains together
    time_interval = (start_interval, end_interval)
    signals = np.array([strain.time_slice(*time_interval, mode='nearest') for strain in strains])
    return signals


def generate(waveform_kwargs):
    """ Read sample and return necessary training params """
    h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)
    signals = get_ht(h_plus, h_cross, waveform_kwargs)
    return (signals)


# BUILT-IN
import os
import uuid
import numpy as np

from scipy.signal import get_window

# PyCBC
import pycbc
from pycbc.types import TimeSeries

# LALSimulation Packages
import lalsimulation as lalsim


class PyCBCGenerateWaveform:
    def __init__(self, 
                 rwrap = 3.0, 
                 beta_taper = 8, 
                 pad_duration_estimate = 1.1, 
                 min_mass = 5.0,
                 debug_me = False
                ):

        # Generate the frequency grid (default values)
        self.f_lower = 0.0 # Hz
        self.f_upper = 0.0 # Hz
        self.delta_t = 0.0 # seconds
        self.f_ref = 0.0 # Hz
        self.sample_rate = 0.0 # Hz
        # Clean-up params
        self.rwrap = rwrap
        # Tapering params
        self.beta = beta_taper
        # Condition for optimising f_min
        self.duration_padfactor = pad_duration_estimate
        # Projection params
        self.signal_length = 0.0 # seconds
        self.whiten_padding = 0.0 # seconds
        self.error_padding_in_s = 0.0 # seconds
        # Other        
        self.min_mass = min_mass
        self.debug_me = debug_me

    def precompute_common_params(self):
        # Pick the longest waveform from priors to make some params static
        # Default params are for m1 = 5.01, m2 = 5.0 and with aligned spins s1z, s2z = 0.99
        # Minimum mass is chosen to be below prior minimum mass (just in case)
        ## Sanity check
        assert any([self.f_lower, self.f_upper, self.delta_t, \
                    self.f_ref, self.signal_length, self.whiten_padding])
        ## End
        _theta = {'mass1': self.min_mass+0.01, 'mass2': self.min_mass, 'spin1z': 0.99, 'spin2z': 0.99}
        self.tmp_f_lower, self.tmp_delta_f, self.fsize = self.optimise_fmin(_theta)
        # Get the fseries over which we get the waveform in FD
        self.fseries = np.arange(0.0, self.f_upper, self.tmp_delta_f)
        # self.fseries = np.arange(self.tmp_f_lower, self.f_upper, self.tmp_delta_f)
        fseries_trunc = self.fseries[:self.fsize]
        self.cshift = np.exp(-2j*np.pi*(-self.rwrap)*fseries_trunc)
        self.clean_idx = self.fseries < self.tmp_f_lower
        # Windowing params
        self.width = self.f_lower - self.tmp_f_lower
        self.winlen = int(2. * (self.width / self.tmp_delta_f))
        self.window = np.array(get_window(('kaiser', self.beta), self.winlen))
        self.kmin = int(self.tmp_f_lower / self.tmp_delta_f)
        self.kmax = self.kmin + self.winlen//2

    """ ONE-OFF FUNCTIONS """
    def get_imr_duration(self, theta, f_lower):
        # This function is applicable for IMRPhenomD and IMRPhenomPv2
        # Multiplying by a factor of 1.1 for overestimate of signal duration
        return 1.1 * lalsim.SimIMRPhenomDChirpTime(theta['mass1']*1.989e+30, theta['mass2']*1.989e+30, 
                                                   theta['spin1z'], theta['spin2z'], 
                                                   f_lower)
    
    def nearest_larger_binary_number(self, input_len):
        # Return the nearest binary number larger than input_len.
        return int(2**np.ceil(np.log2(input_len)))

    def optimise_fmin(self, theta):
        ## NOTE: We find that even for the longest duration waveform we deal with
        ## the value of f_lower is still 17.02 Hz (we can fix this value and remove this function)
        # determine the duration to use
        full_duration = duration = self.get_imr_duration(theta, self.f_lower)
        tmp_f_lower = self.f_lower
        while True:
            # This iteration is typically done 16 times
            full_duration = self.get_imr_duration(theta, tmp_f_lower)
            condition = duration * self.duration_padfactor
            if full_duration >= condition:
                break
            else:
                # We can change this to tmp_f_lower -= 3.0 to lower iterations
                # It will consequently increase the time taken for waveform generation process
                # But, we've already seen that this shouldn't matter much for Ripple
                # tmp_f_lower *= 0.99 is consistent with PyCBC docs
                tmp_f_lower *= 0.99

        # factor to ensure the vectors are all large enough. We don't need to
        # completely trust our duration estimator in this case, at a small
        # increase in computational cost
        fudge_duration = (full_duration + .1 + self.rwrap) * self.duration_padfactor
        fsamples = int(fudge_duration / self.delta_t)
        N = self.nearest_larger_binary_number(fsamples)
        fudge_duration = N * self.delta_t

        tmp_delta_f = 1.0 / fudge_duration
        tsize = int(1.0 / self.delta_t /  tmp_delta_f)
        fsize = tsize // 2 + 1

        return (tmp_f_lower, tmp_delta_f, fsize)
    
    """ JITTABLES """
    # Jitting these functions require the first argument (self) to be defined as static
    def convert_to_timeseries(self, hpol):
        ## Convert frequency series to time series
        return np.fft.irfft(hpol) * (1./self.delta_t)
    
    def fd_taper_left(self, out):
        # Apply Tapering
        out[self.kmin:self.kmax] = out[self.kmin:self.kmax] * self.window[:self.winlen//2]
        out[:self.kmin] = out[:self.kmin] * 0.
        # Convert frequency series to time series
        out = self.convert_to_timeseries(out)
        return out
    
    def cyclic_time_shift(self, hpol):
        return hpol * self.cshift

    def resize(self, hpol):
        return hpol[0:self.fsize]
    
    def get_theta_pycbc(self, theta):
        # Add required params to waveform kwargs
        theta['f_lower'] = self.tmp_f_lower
        theta['delta_f'] = self.tmp_delta_f
        theta['delta_t'] = self.delta_t
        theta['f_final'] = self.f_upper
        return theta
    
    def get_pycbc_hphc(self, theta):
        # Get frequency domain waveform
        return pycbc.waveform.get_fd_waveform(**theta)

    """ MAIN """
    def make_injection(self, hp, hc, params):
        # Get the required sample length and tc
        sample_length_in_s = len(hp)/self.sample_rate
        tc_obs = sample_length_in_s - self.rwrap
        tc_req = params['tc']
        start_time = tc_obs - tc_req
        end_time = tc_obs + (self.signal_length - tc_req)
        # Pad the start and end times for whitening and error padding
        start_time -= (self.whiten_padding/2.0 + self.error_padding_in_s)
        end_time += (self.whiten_padding/2.0 + self.error_padding_in_s)
        # Pad hpol with zeros (append or prepend) if necessary
        left_pad = int(-start_time * self.sample_rate) if start_time < 0.0 else 0
        right_pad = int((end_time-sample_length_in_s) * self.sample_rate) if end_time > sample_length_in_s else 0
        hp = np.pad(hp, (left_pad, right_pad), 'constant', constant_values=(0.0, 0.0))
        hc = np.pad(hc, (left_pad, right_pad), 'constant', constant_values=(0.0, 0.0))
        # Slice the required section out of hpol
        start_idx = int(start_time*self.sample_rate) if start_time > 0.0 else 0
        end_idx = int(end_time*self.sample_rate) + int(left_pad)
        slice_idx = slice(start_idx, end_idx)
        hp = hp[slice_idx]
        hc = hc[slice_idx]
        return (hp, hc)

    def project(self, hp, hc, special, params):
        # Get hp, hc in the time domain and convert to h(t)
        # Time of coalescence
        tc = params['tc']
        tc_gps = params['injection_time']
        ## Get random value (with a given prior) for polarisation angle, ra, dec
        np.random.seed(special['sample_seed'])
        # Polarisation angle
        pol_angle = special['distrs']['pol'].rvs()[0][0]
        # Right ascension, declination
        sky_pos = special['distrs']['sky'].rvs()[0]
        declination, right_ascension = sky_pos
        # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
        hp = TimeSeries(hp, delta_t=self.delta_t)
        hc = TimeSeries(hc, delta_t=self.delta_t)
        # Get start interval and end interval for time series
        # Start and end interval define slice of ts without error padding
        pre_coalescence = tc + (self.whiten_padding/2.0)
        start_interval = tc_gps - pre_coalescence
        post_merger = self.signal_length - tc
        end_interval = tc_gps + post_merger + (self.whiten_padding/2.0)
        # Setting the start time for hp and hc
        hp.start_time = hc.start_time = start_interval - self.error_padding_in_s
        # Returns
        time_interval = (start_interval, end_interval)
        return (hp, hc, time_interval)

    def generate(self, _theta):
        theta = _theta.copy()
        ## Generate waveform on the fly using GPU-accelerated Ripple
        # Convert theta to theta_ripple (jnp) (only required params)
        theta_pycbc = self.get_theta_pycbc(theta)
        # Get h_plus and h_cross from the given waveform parameters theta
        # Note than hp and hc are in the frequency domain
        hp, hc = self.get_pycbc_hphc(theta_pycbc)
        # Resizing (to the required sample rate)
        hp = self.resize(hp)
        hc = self.resize(hc)
        # Cyclic time-shift
        hp = self.cyclic_time_shift(hp)
        hc = self.cyclic_time_shift(hc)
        # Tapering and convert from freq domain to time domain (fd_to_td)
        hp_td = self.fd_taper_left(hp)
        hc_td = self.fd_taper_left(hc)

        return hp_td, hc_td
    
    def apply(self, params: dict, special: dict):
        # Set lal.Detector object as global as workaround for MP methods
        # Project wave does not work with DataLoader otherwise
        setattr(self, 'dets', special['dets'])
        # Augmentation on all params
        hp, hc = self.generate(params)
        # Make hp, hc into proper injection (adjust to tc and zero pad)
        hp, hc = self.make_injection(hp, hc, params)
        # Convert hp, hc into h(t) using antenna pattern (H1, L1 considered)
        out = self.project(hp, hc, special, params)
        # output: (h_plus, h_cross, time_interval)
        return out
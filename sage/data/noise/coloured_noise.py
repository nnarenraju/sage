#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : coloured_noise.py
Description   : Short description of the file

Created on 2026-01-19 16:18:07

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


class ColouredNoiseGenerator:
    """Generate Dataset 3 -like noise for Sage training"""

    def __init__(self, psds_dir: str = ""):
        self.psds_dir = psds_dir
        # H1 and L1 dirs expected inside psds parent directory
        H1_dir = os.path.join(self.psds_dir, "H1")
        L1_dir = os.path.join(self.psds_dir, "L1")
        # Get all .hdf files containing one psd each
        self.psd_options = {
            "H1": glob.glob(os.path.join(H1_dir, "*.hdf")),
            "L1": glob.glob(os.path.join(L1_dir, "*.hdf")),
        }
        # Other params
        self.sample_length = None
        self.delta_f = None
        self.noise_low_freq_cutoff = None
        self.sample_rate = None

    def precompute_common_params(self):
        # Compute ASD for chosen PSD
        self.complex_asds = {det: [] for det in self.psd_options.keys()}
        for i, det in enumerate(self.psd_options.keys()):
            # Read all detector PSDs as frequency series with appropriate delta_f
            for psd_det in self.psd_options[det]:
                psd = load_frequencyseries(psd_det)
                psd = interpolate(psd, 1.0 / self.sample_length)
                # Convert PSD's to ASD's for colouring the white noise
                foo = self.psd_to_asd(
                    psd,
                    0.0,
                    self.sample_length,
                    sample_rate=self.sample_rate,
                    low_frequency_cutoff=self.noise_low_freq_cutoff,
                    filter_duration=self.sample_length,
                )
                self.complex_asds[det].append(foo)

    def psd_to_asd(
        self,
        psd,
        start_time,
        end_time,
        sample_rate=2048.0,
        low_frequency_cutoff=15.0,
        filter_duration=128,
    ):

        psd = psd.copy()

        flen = int(sample_rate / psd.delta_f) // 2 + 1
        oldlen = len(psd)
        psd.resize(flen)

        # Want to avoid zeroes in PSD.
        max_val = psd.max()
        for i in range(len(psd)):
            if i >= (oldlen - 1):
                psd.data[i] = psd[oldlen - 2]
            if psd[i] == 0:
                psd.data[i] = max_val

        fil_len = int(filter_duration * sample_rate)
        wn_dur = int(end_time - start_time) + 2 * filter_duration
        if psd.delta_f >= 1.0 / (2.0 * filter_duration):
            # If the PSD is short enough, this method is less memory intensive than
            # resizing and then calling inverse_spectrum_truncation
            psd = pycbc.psd.interpolate(psd, 1.0 / (2.0 * filter_duration))
            # inverse_spectrum_truncation truncates the inverted PSD. To truncate
            # the non-inverted PSD we give it the inverted PSD to truncate and then
            # invert the output.
            psd = 1.0 / pycbc.psd.inverse_spectrum_truncation(
                1.0 / psd,
                fil_len,
                low_frequency_cutoff=low_frequency_cutoff,
                trunc_method="hann",
            )
            psd = psd.astype(complex_same_precision_as(psd))
            # Zero-pad the time-domain PSD to desired length. Zeroes must be added
            # in the middle, so some rolling between a resize is used.
            psd = psd.to_timeseries()
            psd.roll(fil_len)
            psd.resize(int(wn_dur * sample_rate))
            psd.roll(-fil_len)
            # As time series is still mirrored the complex frequency components are
            # 0. But convert to real by using abs as in inverse_spectrum_truncate
            psd = psd.to_frequencyseries()

        kmin = int(low_frequency_cutoff / psd.delta_f)
        psd[:kmin].clear()
        asd = (psd.squared_norm()) ** 0.25
        return asd

    def colored_noise(
        self,
        asd,
        start_time,
        end_time,
        seed=42,
        sample_rate=2048.0,
        filter_duration=128,
        det=None,
    ):
        """Create noise from a PSD

        Return noise from the chosen PSD. Note that if unique noise is desired
        a unique seed should be provided.

        Parameters
        ----------
        asd : pycbc.types.FrequencySeries
            ASD to color the noise
        start_time : int
            Start time in GPS seconds to generate noise
        end_time : int
            End time in GPS seconds to generate noise
        seed : {None, int}
            The seed to generate the noise.
        sample_rate: {16384, float}
            The sample rate of the output data. Keep constant if you want to
            ensure continuity between disjoint time spans.
        filter_duration : {128, float}
            The duration in seconds of the coloring filter

        Returns
        --------
        noise : TimeSeries
            A TimeSeries containing gaussian noise colored by the given psd.
        """

        white_noise = self.normal(
            start_time - filter_duration,
            end_time + filter_duration,
            seed=seed,
            sample_rate=sample_rate,
        )

        asd = interpolate(asd, 1.0 / (len(white_noise) / 2048.0))
        white_noise = white_noise.to_frequencyseries()
        # Here we color. Do not want to duplicate memory here though so use '*='
        white_noise *= asd
        colored = white_noise.to_timeseries(delta_t=1.0 / sample_rate)
        return colored.time_slice(start_time, end_time)

    def normal(self, start, end, sample_rate=2048.0, seed=0):
        """Generate data with a white Gaussian (normal) distribution

        Parameters
        ----------
        start_time : int
            Start time in GPS seconds to generate noise
        end_time : int
            End time in GPS seconds to generate noise
        sample-rate: float
            Sample rate to generate the data at. Keep constant if you want to
            ensure continuity between disjoint time spans.
        seed : {None, int}
            The seed to generate the noise.

        Returns
        --------
        noise : TimeSeries
            A TimeSeries containing gaussian noise
        """

        data = np.random.normal(
            size=int((end - start) * sample_rate), scale=(sample_rate / 2.0) ** 0.5
        )
        return TimeSeries(data, delta_t=1.0 / sample_rate)

    def choose_asd(self):
        # Choose asd for each detector randomly
        # Similar to D3 of MLGWSC-1
        H1_asd = random.choice(self.complex_asds["H1"])
        L1_asd = random.choice(self.complex_asds["L1"])
        return (H1_asd, L1_asd)

    def generate(self, asd, seed, det):
        # Create noise realisation with given ASD
        noise = self.colored_noise(
            asd,
            0.0,
            self.sample_length,
            seed=seed,
            sample_rate=self.sample_rate,
            filter_duration=1.0,
            det=det,
        )
        noise = noise.numpy()
        return noise

    def apply(self, special, det_only=""):
        # choose a random asd from precomputed set
        time_1 = time.time()
        H1_asd, L1_asd = self.choose_asd()
        # Generate coloured noise using random asd
        rs = np.random.RandomState(seed=special["sample_seed"])
        seeds = list(rs.randint(0, 2**32, 2))  # one for each detector
        H1_noise = self.generate(H1_asd, seeds[0], "H1")
        L1_noise = self.generate(L1_asd, seeds[1], "L1")
        noise = np.stack([H1_noise, L1_noise], axis=0)
        return noise

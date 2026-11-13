#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gaussian noise coloured by measured PSDs, as in MLGWSC-1 dataset 3,
with the colouring of PyCBC ``pycbc.noise.reproduceable``.

Filename      : coloured_noise.py
Description   : Short description of the file

Created on 2026-01-19 16:18:07

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


class ColouredNoiseGenerator:
    """
    H1 and L1 Gaussian noise coloured by randomly drawn measured PSDs.

    Each realisation draws one precomputed ASD per detector, as in
    MLGWSC-1 dataset 3. ``sample_length``, ``sample_rate`` and
    ``noise_low_freq_cutoff`` must be set before
    :meth:`precompute_common_params`.

    Parameters
    ----------
    psds_dir : str
        Directory with ``H1/`` and ``L1/`` subdirectories of ``.hdf``
        PSD files, one PSD per file.
    """

    def __init__(self, psds_dir: str = ""):
        self.psds_dir = psds_dir
        H1_dir = os.path.join(self.psds_dir, "H1")
        L1_dir = os.path.join(self.psds_dir, "L1")
        self.psd_options = {
            "H1": glob.glob(os.path.join(H1_dir, "*.hdf")),
            "L1": glob.glob(os.path.join(L1_dir, "*.hdf")),
        }
        self.sample_length = None
        self.delta_f = None
        self.noise_low_freq_cutoff = None
        self.sample_rate = None

    def precompute_common_params(self):
        """
        Load every PSD file and convert it to a colouring ASD.

        Needed before :meth:`generate` or :meth:`apply`; the ASDs are
        stored per detector in ``complex_asds``.
        """
        self.complex_asds = {det: [] for det in self.psd_options.keys()}
        for i, det in enumerate(self.psd_options.keys()):
            # PSDs interpolated to delta_f = 1 / sample_length
            for psd_det in self.psd_options[det]:
                psd = load_frequencyseries(psd_det)
                psd = interpolate(psd, 1.0 / self.sample_length)
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
        """
        Colouring ASD of a PSD, as in PyCBC ``colored_noise``.

        Zero bins are filled; if ``psd.delta_f >= 1 / (2 *
        filter_duration)`` the PSD is truncated to ``filter_duration``
        (Hann) and zero-padded to the noise length.

        Parameters
        ----------
        psd : pycbc.types.FrequencySeries
            Power spectral density.
        start_time, end_time : float
            GPS start and end (s); only their difference is used.
        sample_rate : float
            Sample rate (Hz).
        low_frequency_cutoff : float
            Frequency (Hz) below which the ASD is zero.
        filter_duration : float
            Duration (s) of the colouring filter.

        Returns
        -------
        pycbc.types.FrequencySeries
            The ASD, ``squared_norm(psd) ** 0.25``.
        """
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
            # If the PSD is short enough, this method is less memory
            # intensive than resizing and then calling
            # inverse_spectrum_truncation
            psd = pycbc.psd.interpolate(psd, 1.0 / (2.0 * filter_duration))
            # inverse_spectrum_truncation truncates the inverted PSD.
            # To truncate the non-inverted PSD we give it the inverted
            # PSD to truncate and then invert the output.
            psd = 1.0 / pycbc.psd.inverse_spectrum_truncation(
                1.0 / psd,
                fil_len,
                low_frequency_cutoff=low_frequency_cutoff,
                trunc_method="hann",
            )
            psd = psd.astype(complex_same_precision_as(psd))
            # Zero-pad the time-domain PSD to desired length. Zeroes
            # must be added in the middle, so some rolling between a
            # resize is used.
            psd = psd.to_timeseries()
            psd.roll(fil_len)
            psd.resize(int(wn_dur * sample_rate))
            psd.roll(-fil_len)
            # As time series is still mirrored the complex frequency
            # components are 0. But convert to real by using abs as in
            # inverse_spectrum_truncate
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
        """
        Gaussian noise coloured by an ASD (PyCBC ``colored_noise``).

        The ASD is interpolated to ``delta_f = 2048 / N`` for ``N``
        white-noise samples and multiplies the noise in the frequency
        domain.

        Parameters
        ----------
        asd : pycbc.types.FrequencySeries
            ASD that colours the noise.
        start_time : int
            Start time in GPS seconds.
        end_time : int
            End time in GPS seconds.
        seed : int
            Passed to :meth:`normal`, which does not use it.
        sample_rate : float
            Sample rate of the output.
        filter_duration : float
            White noise is drawn this many seconds beyond each end.
        det : str or None
            Not used.

        Returns
        -------
        pycbc.types.TimeSeries
            Coloured noise from ``start_time`` to ``end_time``.
        """

        white_noise = self.normal(
            start_time - filter_duration,
            end_time + filter_duration,
            seed=seed,
            sample_rate=sample_rate,
        )

        asd = interpolate(asd, 1.0 / (len(white_noise) / 2048.0))
        white_noise = white_noise.to_frequencyseries()
        # Here we color. Do not want to duplicate memory here though so
        # use '*='
        white_noise *= asd
        colored = white_noise.to_timeseries(delta_t=1.0 / sample_rate)
        return colored.time_slice(start_time, end_time)

    def normal(self, start, end, sample_rate=2048.0, seed=0):
        """
        White Gaussian noise from NumPy's global random state.

        Parameters
        ----------
        start : float
            Start time in GPS seconds; only the duration is used.
        end : float
            End time in GPS seconds.
        sample_rate : float
            Sample rate; the variance is ``sample_rate / 2``.
        seed : int
            Not used.

        Returns
        -------
        pycbc.types.TimeSeries
            White noise of ``(end - start) * sample_rate`` samples,
            starting at epoch 0.
        """

        data = np.random.normal(
            size=int((end - start) * sample_rate), scale=(sample_rate / 2.0) ** 0.5
        )
        return TimeSeries(data, delta_t=1.0 / sample_rate)

    def choose_asd(self):
        """
        Random precomputed ASD of each detector, as in MLGWSC-1 D3.

        Returns
        -------
        tuple of pycbc.types.FrequencySeries
            ``(H1 ASD, L1 ASD)``.
        """
        H1_asd = random.choice(self.complex_asds["H1"])
        L1_asd = random.choice(self.complex_asds["L1"])
        return (H1_asd, L1_asd)

    def generate(self, asd, seed, det):
        """
        One coloured noise series of ``sample_length`` seconds.

        Parameters
        ----------
        asd : pycbc.types.FrequencySeries
            ASD that colours the noise.
        seed : int
            Passed to :meth:`colored_noise`.
        det : str
            Passed to :meth:`colored_noise`; not used.

        Returns
        -------
        numpy.ndarray
            Coloured noise.
        """
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
        """
        Coloured noise of H1 and L1 with randomly drawn ASDs.

        Parameters
        ----------
        special : dict
            Holds ``"sample_seed"``, the seed of the per-detector seeds.
        det_only : str
            Not used.

        Returns
        -------
        numpy.ndarray, shape ``(2, N)``
            H1 and L1 noise.
        """
        time_1 = time.time()
        H1_asd, L1_asd = self.choose_asd()
        rs = np.random.RandomState(seed=special["sample_seed"])
        seeds = list(rs.randint(0, 2**32, 2))  # one per detector
        H1_noise = self.generate(H1_asd, seeds[0], "H1")
        L1_noise = self.generate(L1_asd, seeds[1], "L1")
        noise = np.stack([H1_noise, L1_noise], axis=0)
        return noise

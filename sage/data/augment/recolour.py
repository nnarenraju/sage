#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : recolour.py
Description   : Short description of the file

Created on 2026-01-19 16:16:50

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


class Recolour(NoiseWrapper):
    """Used to augment the PSD of given real noise segment (D4)"""

    # This method required extra sample length for noise (equal to whiten_padding).
    # Whitening module removes corrupted bits due to possible edge effects.
    def __init__(
        self,
        always_apply=True,
        use_precomputed=False,
        h1_psds_hdf="",
        l1_psds_hdf="",
        use_shifted=False,
        shift_up_factor=10,
        shift_down_factor=1,
        p_recolour=0.3,
        trunc_method="hann",
        debug_me=False,
        debug_dir="",
    ):

        super().__init__(always_apply)
        # Warnings
        # Using precomputed PSDs. Make sure that they do not include PSDs from testing dataset.
        # Unless using a robust set of PSDs, this method is bound by the limitations of supervised learning

        # Using precomputed (using 81 days of O3a noise)
        # WARNING: This is a cheaty method if this uses testing data PSDs for training.
        self.use_precomputed = use_precomputed
        self.h1_psds = h5py.File(h1_psds_hdf, "r")
        self.shape_h1_psds = dict(self.h1_psds.attrs)["shape"]
        self.l1_psds = h5py.File(l1_psds_hdf, "r")
        self.shape_l1_psds = dict(self.h1_psds.attrs)["shape"]
        # Using shifted PSD (shift along y axis)
        self.use_shifted = use_shifted
        self.shift_up_factor = shift_up_factor
        self.shift_down_factor = shift_down_factor
        # Probability of being recoloured
        self.p_recolour = p_recolour

        # Other params
        self.fs = 2048.0  # Hz
        self.sample_length_in_s = 0.0  # seconds
        self.noise_low_freq_cutoff = 0.0  # Hz
        self.signal_low_freq_cutoff = 0.0  # Hz
        self.whiten_padding = 0.0  # seconds
        self.trunc_method = trunc_method

        # DEBUGGER
        self.debug_me = debug_me
        if debug_me:
            # TODO: If this is fast enough, include in export dir
            self.debug_dir = debug_dir
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
            save_txt = os.path.join(debug_dir, "recolour.txt")
            self.tmp_debug = open(save_txt, "a")

    def estimate_psd(self, ts, DET):
        # Compute PSD using Welch's method
        if DET["is_recolour"]:
            freqs, psd = scipy_welch(
                ts, fs=self.fs, nperseg=4.0 * self.fs, average="median"
            )
            delta_f = freqs[1] - freqs[0]
            DET["old_psd"] = psd
            DET["old_delta_f"] = delta_f
        return DET

    def inv_spec_trunc(self, psd, max_filter_len):
        # Interpolate and smooth to the desired corruption length
        psd = inverse_spectrum_truncation(
            psd,
            max_filter_len=max_filter_len,
            low_frequency_cutoff=self.signal_low_freq_cutoff,
            trunc_method=self.trunc_method,
        )
        return psd

    def whiten(self, signal, psd, signal_delta_f):
        """Return a whitened time series"""
        # Convert signal to Timeseries object
        signal = TimeSeries(signal, delta_t=1.0 / self.fs)
        # Filter length for inverse spectrum truncation
        max_filter_len = int(round(self.whiten_padding * self.fs))
        ## Manipulate PSD for usage in whitening
        delta_f = signal_delta_f
        ## Whitening
        psd = self.inv_spec_trunc(psd, max_filter_len)
        # NOTE: Factor of dt not taken into account. Since normlayer takes care of standardisation,
        # we don't necessarily need to include this. After decorrelation, the diagonal matrix
        # values will not be 1's but some other value dependant on the signal input.
        white_frequency_series = signal.to_frequencyseries(delta_f=delta_f) / psd**0.5
        return (white_frequency_series, max_filter_len)

    def get_psd(self, H1, L1):
        ## Use pre-computed PSDs from HDF5 file
        idx_h1 = -1
        idx_l1 = -1
        # H1 - use different PSD
        if H1["is_diff_psd"] and H1["is_recolour"]:
            idx_h1 = np.random.randint(0, int(self.shape_h1_psds[0]))
            H1["new_psd"] = self.h1_psds["data"][idx_h1]
        else:
            H1["new_psd"] = None  # H1['old_psd']
        # L1 - use different PSD
        if L1["is_diff_psd"] and L1["is_recolour"]:
            idx_l1 = np.random.randint(0, int(self.shape_l1_psds[0]))
            L1["new_psd"] = self.l1_psds["data"][idx_l1]
        else:
            L1["new_psd"] = None  # L1['old_psd']

        # Debugger
        if self.debug_me:
            foo = "{}, {}, {}, {}".format(
                idx_h1, idx_l1, H1["is_recolour"], L1["is_recolour"]
            )
            self.tmp_debug.write(foo)

        return H1, L1

    def shift_psd(self, H1, L1):
        # Shift new PSD along y axis
        H1_shift_up_factor = np.random.uniform(1, self.shift_up_factor)
        H1_shift_down_factor = np.random.uniform(1, self.shift_down_factor) ** -1
        H1_up_or_down = 1 if np.random.random() < 0.5 else 0
        if H1["is_recolour"]:
            H1["new_psd"] *= (
                H1_shift_up_factor if H1_up_or_down else H1_shift_down_factor
            )
        L1_shift_up_factor = np.random.uniform(1, self.shift_up_factor)
        L1_shift_down_factor = np.random.uniform(1, self.shift_down_factor) ** -1
        L1_up_or_down = 1 if np.random.random() < 0.5 else 0
        if L1["is_recolour"]:
            L1["new_psd"] *= (
                L1_shift_up_factor if L1_up_or_down else L1_shift_down_factor
            )
        return (H1, L1)

    def debug_recolour(self, data, labels):
        # Plotting debug recoloured
        # NOTE to self: figsize is (width, height)
        fig, axs = plt.subplots(
            len(labels), 1, figsize=(9.0, 9.0 * len(labels)), squeeze=False
        )
        fig.suptitle("Debugging Recolour Module")
        for n, (d, l) in enumerate(zip(data, labels)):
            # Subplot top
            if "psd" in l:
                axs[n][0].loglog(d, label=l)
            else:
                axs[n][0].plot(d, label=l)
            axs[n][0].grid()
            axs[n][0].legend()
        # Other
        filename = "recolour_{}.png".format(uuid.uuid4().hex)
        save = os.path.join(self.debug_dir, filename)
        plt.savefig(save)
        plt.close()

    def resize_to_samplelen(self, ts):
        crop = slice(
            int(self.whiten_padding / 2.0 * self.fs),
            -int(self.whiten_padding / 2.0 * self.fs),
        )
        cropped = ts[crop]
        return cropped

    def recolour(self, ts, DET):
        ## Whiten the noise using old PSD and recolour using new PSD
        if not DET["is_recolour"]:
            cropped = self.resize_to_samplelen(ts)
            return cropped
        # Add a whiten padding to either side of the ts (will be corrupted)
        # padlen = int((self.whiten_padding/2.0)*self.fs)
        # ts = np.pad(ts, (padlen, padlen), 'constant', constant_values=(0, 0))
        # delta_f will have to be changed based on new length
        data_delta_f = 1.0 / (self.sample_length_in_s + self.whiten_padding)
        # Convert the PSD to new delta_f using PyCBC interpolate function
        old_psd = FrequencySeries(DET["old_psd"], delta_f=DET["old_delta_f"])
        old_psd = interpolate(old_psd, data_delta_f)
        # Whitening (Remove old PSD from data) still in fd
        whitened_fd, max_filter_len = self.whiten(ts, old_psd, data_delta_f)
        # Convert the new PSDs to have delta_f similar to data
        # new_psd = FrequencySeries(DET['new_psd'], delta_f=0.25)
        new_psd = FrequencySeries(DET["new_psd"], delta_f=DET["old_delta_f"])
        new_psd = interpolate(new_psd, data_delta_f)
        new_psd = self.inv_spec_trunc(new_psd, max_filter_len)
        # Recolour using new PSD and return to time domain
        recoloured = (whitened_fd * new_psd**0.5).to_timeseries()
        # NOTE: Removing 5 seconds of data here. Make sure sample length is set accordingly.
        recoloured = recoloured[
            int(max_filter_len / 2) : int(len(recoloured) - max_filter_len / 2)
        ].numpy()
        # debug plotter
        if self.debug_me:
            _, recovered = scipy_welch(
                recoloured, fs=self.fs, nperseg=4.0 * self.fs, average="median"
            )
            self.debug_recolour(
                [old_psd, new_psd, ts, recoloured, recovered],
                ["old_psd", "new_psd", "original", "recoloured", "recovered_psd"],
            )

        return recoloured

    def apply(self, y: np.ndarray, debug=""):
        # Apply given PSD augmentation technique
        ## Or not to recolour
        if np.random.rand() >= self.p_recolour:
            cropped_h1 = self.resize_to_samplelen(y[0])
            cropped_l1 = self.resize_to_samplelen(y[1])
            y = np.stack([cropped_h1, cropped_l1], axis=0)
            return y
        ## To Recolour
        # Is the detector noise going to be recoloured?
        # Is the detector PSD going to be shifted along y axis?
        # Is the detector PSD going to be blurred with Gaussian noise?
        always_recolour = True if self.p_recolour == 1.0 else False
        H1 = {
            "is_recolour": np.random.rand() < 0.5,
            "is_diff_psd": np.random.rand() < 0.5,
            "is_shifted": np.random.rand() < 0.5,
            "is_blurred": np.random.rand() < 0.5,
            "recoloured": y[0],
        }
        L1 = {
            "is_recolour": np.random.rand() < 0.5,
            "is_diff_psd": np.random.rand() < 0.5,
            "is_shifted": np.random.rand() < 0.5,
            "is_blurred": np.random.rand() < 0.5,
            "recoloured": y[1],
        }
        # Check for always recolour
        if always_recolour:
            H1["is_recolour"] = True
            L1["is_recolour"] = True
        # Sanity check (happens 25% of the time)
        if not H1["is_recolour"] and not L1["is_recolour"]:
            cropped_h1 = self.resize_to_samplelen(y[0])
            cropped_l1 = self.resize_to_samplelen(y[1])
            y = np.stack([cropped_h1, cropped_l1], axis=0)
            return y
        # shifted and blurred are not implemented yet
        # TODO: Remove this after implementation
        H1["is_diff_psd"] = H1["is_recolour"]
        L1["is_diff_psd"] = L1["is_recolour"]
        # Estimate the old PSD of each detector (as required)
        H1 = self.estimate_psd(y[0], H1)
        L1 = self.estimate_psd(y[1], L1)
        # Recolour and augment (as required)
        if self.use_precomputed:
            H1, L1 = self.get_psd(H1, L1)
        if self.use_shifted:
            H1, L1 = self.shift_psd(H1, L1)

        # Adjusting H1 and L1 for extra padding added (if needed)
        H1["recoloured"] = self.recolour(y[0], H1)
        L1["recoloured"] = self.recolour(y[1], L1)

        recoloured_noise = np.stack([H1["recoloured"], L1["recoloured"]], axis=0)
        return recoloured_noise

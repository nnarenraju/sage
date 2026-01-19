#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : whiten.py
Description   : Short description of the file

Created on 2026-01-19 16:26:37

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


class Whiten(TransformWrapperPerChannel):
    # PSDs can be different between the channels, so we use perChannel method
    def __init__(
        self,
        always_apply=True,
        trunc_method="hann",
        remove_corrupted=True,
        estimated=False,
        whitening_psd_dir=None,
    ):
        super().__init__(always_apply)
        self.trunc_method = trunc_method
        self.remove_corrupted = remove_corrupted
        self.estimated = estimated
        if whitening_psd_dir != None:
            # Store the PSD files here in RAM. This reduces the overhead when whitening.
            # Read all psds in the data_dir and store then as FrequencySeries
            psds = {}
            H1_dir = os.path.join(whitening_psd_dir, "H1")
            L1_dir = os.path.join(whitening_psd_dir, "L1")
            psd_file_H1 = glob.glob(os.path.join(H1_dir, "*.hdf"))
            psd_file_L1 = glob.glob(os.path.join(L1_dir, "*.hdf"))
            for psd_file in psd_file_H1:
                psd_data = load_frequencyseries(psd_file)
                # Store PSD data into lookup dict
                psds["H1"] = psd_data
            for psd_file in psd_file_L1:
                psd_data = load_frequencyseries(psd_file)
                # Store PSD data into lookup dict
                psds["L1"] = psd_data
            self.whitening_psd = [psds["H1"], psds["L1"]]
        else:
            self.whitening_psd = whitening_psd_dir

    def estimate_psd(self, data_cfg, delta_f, max_filter_len):
        ### Estimate the PSD
        delta_t = 1.0 / 2048.0
        seg_len = int(0.5 / delta_t)
        seg_stride = int(seg_len / 2)
        pure_noise = TimeSeries(pure_noise, delta_t=1.0 / data_cfg.sample_rate)
        psd = welch(pure_noise, seg_len=seg_len, seg_stride=seg_stride)
        psd = interpolate(psd, delta_f)
        psd = inverse_spectrum_truncation(
            psd,
            max_filter_len=max_filter_len,
            low_frequency_cutoff=data_cfg.signal_low_freq_cutoff,
            trunc_method=self.trunc_method,
        )
        return psd

    def whiten(self, signal, psd, data_cfg):
        """Return a whitened time series"""
        # Convert signal to Timeseries object
        signal = TimeSeries(signal, delta_t=1.0 / data_cfg.sample_rate)
        # Filter length for inverse spectrum truncation
        max_filter_len = int(round(data_cfg.whiten_padding * data_cfg.sample_rate))

        ## Manipulate PSD for usage in whitening
        ## Interpolation is probably not required as the psds are created based on signal len anyway
        # Calculating delta_f of signal and providing that to the PSD interpolation method
        delta_f = data_cfg.delta_f
        # Interpolate the PSD to the required delta_f
        # NOTE: This may or may not be required (enable if there is a discrepancy in delta_f)
        # Possible bug: It is possible that the sample lengths are not consistent in Dataloader
        psd = interpolate(psd, delta_f)

        ## Whitening
        # Whiten the data by the asd
        if self.estimated:
            raise NotImplementedError(
                "PSD Estimation method not working at the moment!"
            )

        # Interpolate and smooth to the desired corruption length
        psd = inverse_spectrum_truncation(
            psd,
            max_filter_len=max_filter_len,
            low_frequency_cutoff=data_cfg.signal_low_freq_cutoff,  # used to be signal low freq cutoff
            trunc_method=self.trunc_method,
        )

        # NOTE: Factor of dt not taken into account. Since layernorm takes care of standardisation,
        # we don't necessarily need to include this. After decorrelation, the diagonal matrix
        # values will not be 1 but some other value dependant on the signal input.
        white = (signal.to_frequencyseries(delta_f=delta_f) / psd**0.5).to_timeseries()

        if self.remove_corrupted:
            white = white[
                int(max_filter_len / 2) : int(len(white) - max_filter_len / 2)
            ]

        return white

    def apply(self, y: np.ndarray, channel: int, special: dict):
        ## Whitening using approximate PSD
        # Use a fixed provided PSD
        if self.whitening_psd != None:
            psd = self.whitening_psd[channel]
        else:
            psd = special["psds"][channel]
        # Whiten sample and return results
        return self.whiten(y, psd, special["data_cfg"])

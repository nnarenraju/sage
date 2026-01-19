#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : augmentation.py
Description   : Short description of the file

Created on 2026-01-19 16:04:01

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



class AugmentPolSky(SignalWrapper):
    """Used to augment polarisation angle, ra and dec (Sky position)"""

    def __init__(self, always_apply=True, augmentation=True):
        super().__init__(always_apply)
        self.augmentation = augmentation

    def augment(self, signals, params, special):
        ## Get random value (with a given prior) for polarisation angle, ra, dec
        # Polarisation angle
        pol_angle = (
            special["distrs"]["pol"].rvs()[0][0]
            if self.augmentation
            else params["polarization"]
        )
        # Right ascension, declination
        sky_pos = (
            special["distrs"]["sky"].rvs()[0]
            if self.augmentation
            else (params["dec"], params["ra"])
        )
        # Times
        time_interval = (params["interval_lower"], params["interval_upper"])
        start_time = params["start_time"]
        # h+ and hx
        h_plus = signals[0]
        h_cross = signals[1]

        declination, right_ascension = sky_pos

        # Using PyCBC project_wave to get h_t from h_plus and h_cross
        # Setting the start_time is important! (too late, too early errors are because of this)
        h_plus = TimeSeries(h_plus, delta_t=1.0 / params["sample_rate"])
        h_cross = TimeSeries(h_cross, delta_t=1.0 / params["sample_rate"])
        # Set start times
        h_plus.start_time = start_time
        h_cross.start_time = start_time

        # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
        strains = [
            det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle)
            for det in self.dets
        ]
        # Put both strains together
        augmented_signal = np.array(
            [strain.time_slice(*time_interval, mode="nearest") for strain in strains]
        )
        # Update params
        params["declination"] = declination
        params["right_ascension"] = right_ascension
        params["polarisation_angle"] = pol_angle

        return (augmented_signal, params)

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Set lal.Detector object as global as workaround for MP methods
        # Project wave does not work with DataLoader otherwise
        setattr(self, "dets", special["dets"])
        # Augmentation on polarisation and sky position
        out, params = self.augment(y, params, special)
        # Update params
        params.update(params)
        # Input: (h_plus, h_cross) --> output: (det1 h_t, det_2 h_t)
        # Shape remains the same, so reading in dataset object won't be a problem
        return (out, params, special)


class AugmentDistance(SignalWrapper):
    """Used to augment the distance parameter of the given signal"""

    def __init__(self, always_apply=True, uniform_dchirp=False):
        super().__init__(always_apply)
        self.uniform_dchirp = uniform_dchirp

    def get_augmented_signal(self, signal, params, distrs, debug):
        # Get old params
        distance_old = params["distance"]
        mchirp = params["mchirp"]
        # Getting new distance
        if self.uniform_dchirp:
            chirp_distance = np.random.uniform(130.0, 350.0, size=1)[0]
        else:
            chirp_distance = distrs["dchirp"].rvs()[0][0]
        # Producing the new distance with the required priors
        distance_new = chirp_distance * (2.0 ** (-1.0 / 5) * 1.4 / mchirp) ** (-5.0 / 6)

        ## Augmenting on the distance
        augmented_signal = (distance_old / distance_new) * signal
        # Update params
        params["distance"] = distance_new
        params["dchirp"] = chirp_distance
        return (augmented_signal, params)

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Augmenting on distance parameter
        # Unpack required elements from special for augmentation
        distrs = special["distrs"]
        norms = special["norm"]
        # Run through the augmentation procedure with given dist, mchirp
        out, params = self.get_augmented_signal(y, params, distrs, debug)
        # Update params
        params.update(params)
        # Update special
        special["norm_dist"] = norms["dist"].norm(params["distance"])
        special["norm_dchirp"] = norms["dchirp"].norm(params["dchirp"])
        # Send back the rescaled signal and updated dicts
        return (out, params, special)


class AugmentOptimalNetworkSNR(SignalWrapper):
    """Used to augment the SNR distribution of the dataset"""

    def __init__(
        self,
        always_apply=True,
        rescale=True,
        use_uniform=False,
        use_beta=False,
        a=2,
        b=5,
        use_add5=False,
        use_halfnorm=False,
        snr_lower_limit=5.0,
        snr_upper_limit=15.0,
        fix_snr=None,
        always_rescale_for_validation=True,
    ):

        super().__init__(always_apply)
        # If rescale is False, AUG method returns original network_snr, norm_snr and signal
        self.rescale = rescale
        # Applying a custom distributions for SNR PDFs
        self.use_uniform = use_uniform
        self.use_beta = use_beta
        self.use_add5 = use_add5
        self.a = a
        self.b = b
        self.use_halfnorm = use_halfnorm
        self.snr_lower_limit = snr_lower_limit
        self.snr_upper_limit = snr_upper_limit
        self.fix_snr = fix_snr
        self.always_rescale_for_validation = always_rescale_for_validation

    def _dchirp_from_dist(self, dist, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the chirp distance given the luminosity distance and chirp mass.
        return dist * (2.0 ** (-1.0 / 5) * ref_mass / mchirp) ** (5.0 / 6)

    def get_rescaled_signal(
        self, signal, psds, params, cfg, debug, training, aux, epoch, seed
    ):
        # params: This will contain params var found in __getitem__ method of custom dataset object
        # np.random.seed(seed) ----------------------------------------------------------------------------------- ???
        # Get original network SNR
        prelim_network_snr = get_network_snr(
            signal, psds, params, cfg.export_dir, debug
        )

        if self.rescale or (not training and self.always_rescale_for_validation):
            if aux == -1:
                # Rescaling the SNR to a uniform distribution within a given range
                rescaled_snr_lower = self.snr_lower_limit
                rescaled_snr_upper = self.snr_upper_limit

                # Uniform on SNR range
                if self.use_uniform:
                    target_snr = np.random.uniform(
                        rescaled_snr_lower, rescaled_snr_upper
                    )
                elif self.use_beta:
                    target_snr = beta.rvs(self.a, self.b)
                    target_snr *= rescaled_snr_upper
                    target_snr += rescaled_snr_lower
                elif self.use_add5:
                    # Make everything detectible
                    target_snr = prelim_network_snr + 5.0
                elif self.use_halfnorm:
                    target_snr = halfnorm.rvs() * 4.0 + 5.0

            elif aux in [0, 2]:
                target_snr = 5.0
            elif aux in [1, 3]:
                target_snr = 12.0
            else:
                raise ValueError("Unidentified value for cflag!")

            # Fix SNR for all input signals
            if self.fix_snr != None:
                target_snr = self.fix_snr

            rescaling_factor = target_snr / prelim_network_snr
            # Add noise to rescaled signal
            rescaled_signal = signal * rescaling_factor

            # Adjust distance parameter for signal according to the new rescaled SNR
            rescaled_distance = params["distance"] / rescaling_factor
            rescaled_dchirp = self._dchirp_from_dist(
                rescaled_distance, params["mchirp"]
            )
            # Update targets and params with new rescaled distance is not possible
            # We do not know the priors of network_snr properly
            if "parameter_estimation" in cfg.model_params.keys():
                parameter_estimation = cfg.model_params["parameter_estimation"]
                if (
                    "norm_dist" in parameter_estimation
                    or "norm_dchirp" in parameter_estimation
                ):
                    raise RuntimeError(
                        "rescale_snr option cannot be used with dist/dchirp PE!"
                    )
            # Update the params dictionary with new rescaled distances
            params["distance"] = rescaled_distance
            params["dchirp"] = rescaled_dchirp
            # Add network snr to params as well
            params["network_snr"] = target_snr
        else:
            # Default option returns only network snr
            params["network_snr"] = prelim_network_snr
            rescaled_signal = signal

        return (rescaled_signal, params)

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Unpack required elements from special for augmentation
        psds = special["psds"]
        cfg = special["cfg"]
        training = special["training"]
        aux = special["aux"]
        norms = special["norm"]
        epoch = special["epoch"]
        seed = special["sample_seed"]
        # Augmentation on optimal network SNR
        out, params = self.get_rescaled_signal(
            y, psds, params, cfg, debug, training, aux, epoch, seed
        )
        # Update params
        params.update(params)
        # Update special
        special["network_snr"] = params["network_snr"]
        special["norm_snr"] = norms["snr"].norm(params["network_snr"])
        # Send back the rescaled signal and updated dicts
        return (out, params, special)



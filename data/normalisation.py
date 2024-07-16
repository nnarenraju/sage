class Normalise:
    """
    Normalise the parameter using prior ranges
    
        For example, norm_tc = (tc - min_val)/(max_val - min_val)
        The values of max_val and min_val are provided
        to the class. self.get_norm can be called during
        data generation to get normalised values of tc, if needed.
    
    """
    
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def norm(self, val):
        # Return lambda to use for normalisation
        return (val - self.min_val)/(self.max_val - self.min_val)


def _dist_from_dchirp(chirp_distance, mchirp, ref_mass=1.4):
    # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
    # Returns the luminosity distance given a chirp distance and chirp mass.
    return chirp_distance * (2.**(-1./5) * ref_mass / mchirp)**(-5./6)


def get_normalisations(cfg, data_cfg):
    # Normalise time of coalescence
    norm_tc = Normalise(min_val=data_cfg.tc_inject_lower, max_val=data_cfg.tc_inject_upper)

    # Normalise chirp mass
    ml = data_cfg.prior_low_mass
    mu = data_cfg.prior_high_mass
    # m2 will always be slightly lower than m1, but (m, m) will give limit
    # that the mchirp will never reach but tends to as num_samples tends to inf.
    # Range for mchirp can be written as --> (min_mchirp, max_mchirp)
    min_mchirp = (ml*ml / (ml+ml)**2.)**(3./5) * (ml + ml)
    max_mchirp = (mu*mu / (mu+mu)**2.)**(3./5) * (mu + mu)
    norm_mchirp = Normalise(min_val=min_mchirp, max_val=max_mchirp)
    
    # Get distance ranges from chirp distance priors
    # mchirp present in numerator of self.distance_from_chirp_distance.
    # Thus, min_mchirp for dist_lower and max_mchirp for dist_upper
    dist_lower = _dist_from_dchirp(data_cfg.prior_low_chirp_dist, min_mchirp)
    dist_upper = _dist_from_dchirp(data_cfg.prior_high_chirp_dist, max_mchirp)
    # get normlised distance class
    norm_dist = Normalise(min_val=dist_lower, max_val=dist_upper)
    
    # Normalise chirp distance
    norm_dchirp = Normalise(min_val=data_cfg.prior_low_chirp_dist, 
                            max_val=data_cfg.prior_high_chirp_dist)

    # Normalise the mass ratio 'q'
    norm_q = Normalise(min_val=1.0, max_val=mu/ml)
    norm_invq = Normalise(min_val=0.0, max_val=1.0)
    
    # Normalise the SNR
    metadata = cfg.transforms['signal'](np.ndarray, {}, {}, return_metadata=True)
    snr_lower_limit = metadata['AugmentOptimalNetworkSNR']['snr_lower_limit']
    snr_upper_limit = metadata['AugmentOptimalNetworkSNR']['snr_upper_limit']
    print(snr_lower_limit, snr_upper_limit)
    raise
    norm_snr = Normalise(min_val=snr_lower_limit,
                         max_val=snr_upper_limit)
    
    # All normalisation variables
    norm = {}
    norm['dist'] = norm_dist
    norm['dchirp'] = norm_dchirp
    norm['mchirp'] = norm_mchirp
    norm['q'] = norm_q
    norm['invq'] = norm_invq
    norm['snr'] = norm_snr
    norm['tc'] = norm_tc

    limits = {}
    limits['mchirp'] = (min_mchirp, max_mchirp)

    return norm, limits

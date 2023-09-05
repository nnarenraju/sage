# Modules
import numpy as np
from pycbc import distributions


def get_distributions(data_cfg):
    # Used for obtaining random polarisation angle
    uniform_angle_distr = distributions.angular.UniformAngle(uniform_angle=(0., 2.0*np.pi))
    # Used for obtaining random ra and dec
    skylocation_distr = distributions.sky_location.UniformSky()
    # Used for obtaining random mass
    mass_distr = distributions.Uniform(mass=(data_cfg.prior_low_mass, data_cfg.prior_high_mass))
    # Used for obtaining random chirp distance
    dist_gen = distributions.power_law.UniformRadius
    chirp_distance_distr = dist_gen(distance=(data_cfg.prior_low_chirp_dist, 
                                              data_cfg.prior_high_chirp_dist))
    
    # Distributions object
    distrs = {'pol': uniform_angle_distr, 
              'sky': skylocation_distr,
              'mass': mass_distr, 
              'dchirp': chirp_distance_distr}
    
    return distrs

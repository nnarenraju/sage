# Packages
import numpy as np


def get_uniform_masses(mass_lower, mass_upper, num_samples):
    # Get uniform mass distribution
    x_mass = [np.random.uniform(mass_lower, mass_upper, num_samples) for _ in range(2)]
    # Apply the mass constraint (mass2 <= mass1)
    masses = np.column_stack((x_mass[0], x_mass[1]))
    masses = np.fliplr(np.sort(masses, axis=1))
    # Sanity check
    assert all(masses[:,0] > masses[:,1]), "Mass1 > Mass2 in mass priors!"
    # Assign mass1 and mass2
    mass1 = masses[:,0]
    mass2 = masses[:,1]
    return (mass1, mass2)

def q_from_uniform_mass1_mass2(mass1, mass2):
    # Calculate mass ratio (mass1/mass2) on bounds [1, +inf]
    return mass1/mass2

def mchirp_from_uniform_signal_duration(tau_lower, tau_upper, num_samples, signal_low_freq_cutoff):
    # Calculate chirp mass from uniform signal duration
    lf = signal_low_freq_cutoff
    G = 6.67e-11
    c = 3.0e8
    uniform_signal_duration = np.random.uniform(tau_lower, tau_upper, num_samples)
    chirp_mass_from_uniform_tau = ((uniform_signal_duration/5.) * (8.*np.pi*lf)**(8./3.))**(-3./5.) * (c**3./(G*1.989e30))
    return chirp_mass_from_uniform_tau

def mass1_from_mchirp_q(mchirp, q):
    """Returns the primary mass from the given chirp mass and mass ratio."""
    mass1 = q**(2./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass1

def mass2_from_mchirp_q(mchirp, q):
    """Returns the secondary mass from the given chirp mass and mass ratio."""
    mass2 = q**(-3./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass2

def mass1_mass2_from_mchirp_q(mchirp, q):
    # Get mass1 and mass2 from mchirp and q
    mass1 = mass1_from_mchirp_q(mchirp, q)
    mass2 = mass2_from_mchirp_q(mchirp, q)
    return (mass1, mass2)

def get_tau_priors(ml, mu, lf):
    # m2 will always be slightly lower than m1, but (m, m) will give limit
    # that the mchirp will never reach but tends to as num_samples tends to inf.
    # Range for mchirp can be written as --> (min_mchirp, max_mchirp)
    min_mchirp = (ml*ml / (ml+ml)**2.)**(3./5) * (ml + ml)
    max_mchirp = (mu*mu / (mu+mu)**2.)**(3./5) * (mu + mu)
    # Tau priors
    G = 6.67e-11
    c = 3.0e8
    tau = lambda mc: 5. * (8.*np.pi*lf)**(-8./3.) * (mc*1.989e30*G/c**3.)**(-5./3.)
    tau_lower = tau(max_mchirp)
    tau_upper = tau(min_mchirp)
    return (tau_lower, tau_upper)

def get_uniform_mchirp(ml, mu, num_samples, min_mchirp=None, max_mchirp=None, override=False):
    # m2 will always be slightly lower than m1, but (m, m) will give limit
    # that the mchirp will never reach but tends to as num_samples tends to inf.
    # Range for mchirp can be written as --> (min_mchirp, max_mchirp)
    if override:
        min_mchirp = min_mchirp
        max_mchirp = max_mchirp
    else:
        min_mchirp = (ml*ml / (ml+ml)**2.)**(3./5) * (ml + ml)
        max_mchirp = (mu*mu / (mu+mu)**2.)**(3./5) * (mu + mu)
    # Get uniform chirp mass
    uniform_mchirp = np.random.uniform(min_mchirp, max_mchirp, num_samples)
    return uniform_mchirp

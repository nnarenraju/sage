# Packages
import numpy as np
from scipy.stats import beta

from pycbc.conversions import tau0_from_mass1_mass2, tau3_from_mass1_mass2
from pycbc.conversions import mass1_from_tau0_tau3, mass2_from_tau0_tau3
from pycbc.conversions import mass2_from_mchirp_mass1, mchirp_from_tau0
from pycbc.conversions import mchirp_from_mass1_mass2
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q

import lal

""" Conversions """
def q_from_mass1_mass2(mass1, mass2):
    # Calculate mass ratio (mass1/mass2) on bounds [1, +inf]
    return mass1/mass2

def chirp_mass_from_signal_duration(tau, signal_low_freq_cutoff):
    # Calculate chirp mass from signal duration
    lf = signal_low_freq_cutoff # Hz
    G = 6.67e-11 # Nm^2/Kg^2
    c = 3.0e8 # ms^-1
    chirp_mass_from_tau = ((tau/5.) * (8.*np.pi*lf)**(8./3.))**(-3./5.) * (c**3./(G*1.989e30))
    return chirp_mass_from_tau

def signal_duration_from_chirp_mass(mchirp, signal_low_freq_cutoff):
    lf = signal_low_freq_cutoff # Hz
    G = 6.67e-11 # Nm^2/Kg^2
    c = 3.0e8 # ms^-1
    tau_from_chirp_mass = 5. * (8.*np.pi*lf)**(-8./3.) * (mchirp*1.989e30*G/c**3.)**(-5./3.)
    return tau_from_chirp_mass

def mass1_from_mchirp_q(mchirp, q):
    # Returns the primary mass from the given chirp mass and mass ratio.
    mass1 = q**(2./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass1

def mass2_from_mchirp_q(mchirp, q):
    # Returns the secondary mass from the given chirp mass and mass ratio.
    mass2 = q**(-3./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass2

def mass1_mass2_from_mchirp_q(mchirp, q):
    # Get mass1 and mass2 from mchirp and q
    mass1 = mass1_from_mchirp_q(mchirp, q)
    mass2 = mass2_from_mchirp_q(mchirp, q)
    return (mass1, mass2)


""" Prior bounds """
def get_mchirp_priors(ml, mu):
    # Range for mchirp
    min_mchirp = (ml*ml / (ml+ml)**2.)**(3./5) * (ml + ml)
    max_mchirp = (mu*mu / (mu+mu)**2.)**(3./5) * (mu + mu)
    return (min_mchirp, max_mchirp)

def get_tau_priors(ml, mu, lf):
    # Range for mchirp
    min_mchirp, max_mchirp = get_mchirp_priors(ml, mu)
    # Tau priors
    G = 6.67e-11 # Nm^2/Kg^2
    c = 3.0e8 # ms^-1
    tau = lambda mc: 5. * (8.*np.pi*lf)**(-8./3.) * (mc*1.989e30*G/c**3.)**(-5./3.)
    tau_lower = tau(max_mchirp)
    tau_upper = tau(min_mchirp)
    return (tau_lower, tau_upper)


""" Generation """
def get_uniform_masses_with_mass1_gt_mass2(mass_lower, mass_upper, num_samples):
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


class BoundedPriors:

    def __init__(self, mu, ml, lf):
        # Common
        self.lf = lf # Hz (signal low freq cutoff)
        self.mu = mu
        self.ml = ml
        # Constants
        G = 6.67e-11 # Nm^2/Kg^2
        c = 3.0e8 # ms^-1
        self.const = (5. * (8.*np.pi*lf)**(-8./3.)) * (1.989e30*G/c**3.)**(-5./3.)

        # Boundaries on tau0 and tau3
        # Placing templates uniform in the (tau0, tau3) space
        # The boundaries of tau3 are at m1=m2=m_max and m1=m2=m_min
        self.tau3_boundary_low = tau3_from_mass1_mass2(mu, mu, f_lower=lf)
        self.tau3_boundary_high = tau3_from_mass1_mass2(ml, ml, f_lower=lf)

        # The boundaries of tau0 are at also at the same locations
        self.tau0_boundary_low = tau0_from_mass1_mass2(mu, mu, f_lower=lf)
        self.tau0_boundary_high = tau0_from_mass1_mass2(ml, ml, f_lower=lf)

    def _intersection_with_m1_eq_m2(self, tau):
        # Where does the curve intersect with m1=m2?
        C_1dash = self.const * 2**(1./3.)
        intersc_diagonal = (tau/C_1dash)**(-3./5.)
        return intersc_diagonal
    
    def _intersection_with_m2_when_m1_is_mu(self, tau):
        # Where does the curve intersect with x or y axis?
        # Where does it intersect m2 when m1=50.0 Msun?
        C_2dash = self.const/self.mu
        C_3dash = (tau/C_2dash)**3.
        # We get one non-complex root for the value of m2
        m2_when_m1_is_mu = np.array([np.roots([c3d, 0.0, -1, -self.mu]) for c3d in C_3dash])
        m2_when_m1_is_mu = np.array([np.real(foo[np.isreal(foo)])[0] for foo in m2_when_m1_is_mu])
        return m2_when_m1_is_mu

    def _intersection_with_m1_when_m2_is_ml(self, tau):
        # Where does it intersect m1 when m2=7.0 Msun?
        C_4dash = self.const/self.ml
        C_5dash = (tau/C_4dash)**3.
        # We get one non-complex root for the value of m2
        m1_when_m2_is_ml = np.array([np.roots([c5d, 0.0, -1, -self.ml]) for c5d in C_5dash])
        m1_when_m2_is_ml = np.array([np.real(foo[np.isreal(foo)])[0] for foo in m1_when_m2_is_ml])
        return m1_when_m2_is_ml
    
    def _get_m1_upper_bounds(self, m2_when_m1_is_mu, m1_when_m2_is_ml):
        upper_bounds = m2_when_m1_is_mu
        # Checkl if bounds are correct
        idxs = m2_when_m1_is_mu < self.ml
        alt_idxs = m2_when_m1_is_mu >= self.ml
        upper_bounds[idxs] = m1_when_m2_is_ml[idxs]
        upper_bounds[alt_idxs] = np.full(sum(alt_idxs), self.mu)
        return upper_bounds
    
    def _get_m2_from_m1_tau(self, mass1, tau):
        C_6dash = (tau/(self.const/mass1))**3.
        mass2_roots = np.array([np.roots([c6d, 0.0, -1, -_m1]) for c6d, _m1 in zip(C_6dash, mass1)])
        mass2 = np.array([np.real(foo[np.isreal(foo)])[0] for foo in mass2_roots])
        return mass2

    def _common_umc_utau(self, tau):
        ## Step 2: Find the intersection of const Tau curve on P(m1, m2) where m1, m2 are uniform and m1>m2
        intersc_diagonal = self._intersection_with_m1_eq_m2(tau)
        m2_when_m1_is_mu = self._intersection_with_m2_when_m1_is_mu(tau)
        m1_when_m2_is_ml = self._intersection_with_m1_when_m2_is_ml(tau)
        ## Step 3: We can get an upper and lower bound on m1. We can now sample uniformly on m1.
        upper_bounds = self._get_m1_upper_bounds(m2_when_m1_is_mu, m1_when_m2_is_ml)
        lower_bounds = intersc_diagonal
        mass1 = np.array([np.random.uniform(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)])
        ## Step 4: Obtain m2 from m1 and const Tau.
        mass2 = self._get_m2_from_m1_tau(mass1, tau)
        ## Step 5: Obtain q and mchirp.
        q = mass1/mass2
        mchirp = (mass1*mass2 / (mass1+mass2)**2.)**(3./5) * (mass1 + mass2)
        return (mass1, mass2, q, mchirp)
    
    def tau3_lower_boundary_from_tau0(self, _tau0):
        # The lower boundary will always intersect the m1=m2 line
        # m1=m2 corresponds to eta=0.25
        A3 = np.pi / (8.0 * (np.pi*self.lf)**(5./3.))
        A0 = 5.0 / (256.0 * (np.pi*self.lf)**(8./3.))
        lower_boundary_tau3 = 4.0*A3 * (_tau0/(4.0*A0))**(2./5.)
        return lower_boundary_tau3
    
    def tau3_upper_boundary_from_tau0(self, _tau0):
        # The boundary here is chosen based on tau0 where m1=m_max and m2=m_min
        inflection_tau0 = tau0_from_mass1_mass2(self.mu, self.ml, f_lower=self.lf)
        if _tau0 > inflection_tau0:
            # m2=m_min and m1=[m_min, m_max]
            mchirp = mchirp_from_tau0(_tau0, f_lower=self.lf)
            # m1,m2 can be interchanged assuming the func can return m2>m1
            estimated_m1 = mass2_from_mchirp_mass1(mchirp, self.ml)
            upper_boundary_tau3 = tau3_from_mass1_mass2(estimated_m1, self.ml, f_lower=self.lf)
        elif _tau0 < inflection_tau0:
            # m1=m_max and m2=[m_min, m_max]
            mchirp = mchirp_from_tau0(_tau0, f_lower=self.lf)
            estimated_m2 = mass2_from_mchirp_mass1(mchirp, self.mu)
            upper_boundary_tau3 = tau3_from_mass1_mass2(self.mu, estimated_m2, f_lower=self.lf)
        elif _tau0 == inflection_tau0:
            # edge case: tau3 upper boundary where m1=m_max and m2=m_min
            upper_boundary_tau3 = tau3_from_mass1_mass2(self.mu, self.ml, f_lower=self.lf)
        return upper_boundary_tau3
    
    def q_upper_boundary_from_mchirp(self, _mchirp):
        # Maximum value of q (m1/m2) with m1>m2
        inflection_mchirp = mchirp_from_mass1_mass2(self.mu, self.ml)
        if _mchirp > inflection_mchirp:
            # perpendicular of the m1, m2 right triangle
            # m1=m_max and m2=[m_min, m_max]
            estimated_m2 = mass2_from_mchirp_mass1(_mchirp, self.mu)
            upper_boundary_q = self.mu/estimated_m2
        elif _mchirp < inflection_mchirp:
            # base of the m1, m2 right triangle
            # m2=m_min and m1=[m_min, m_max]
            # Assuming that the following function does not constrain m1>m2
            estimated_m1 = mass2_from_mchirp_mass1(_mchirp, self.ml)
            upper_boundary_q = estimated_m1/self.ml
        elif _mchirp == inflection_mchirp:
            # edge case: q upper boundary where m1=m_max and m2=m_min
            upper_boundary_q = self.mu/self.ml
        return upper_boundary_q
    
    # Sampling uniform on (tau0, tau3)
    def _check_tau0_tau3_(self, t0, t3):
        lb_tau3 = self.tau3_lower_boundary_from_tau0(t0)
        ub_tau3 = self.tau3_upper_boundary_from_tau0(t0)
        if lb_tau3 <= t3 <= ub_tau3:
            return True
        else:
            return False
    

    ### ALL METHODS

    def get_bounded_gwparams_from_uniform_tau(self):
        ## Step 1a: Get uniform Tau values within bounds provided by m1 and m2
        tau_lower, tau_upper = get_tau_priors(ml=self.ml, mu=self.mu, lf=self.lf)
        tau = np.random.uniform(tau_lower, tau_upper, 1)
        # Common steps
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)
    
    def get_bounded_gwparams_from_uniform_mchirp(self):
        ## Step 1b: Get uniform chirp mass and convert to tau
        mchirp_lower, mchirp_upper = get_mchirp_priors(ml=self.ml, mu=self.mu)
        expected_mchirp = np.random.uniform(mchirp_lower, mchirp_upper, 1)
        tau = signal_duration_from_chirp_mass(expected_mchirp, self.lf)
        # Common steps
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)
    
    def get_bounded_gwparams_from_powerlaw_mchirp(self):
        # Get tau from a beta distribution
        ## Step 1c: Get power law chirp mass and convert to tau
        mchirp_lower, mchirp_upper = get_mchirp_priors(ml=self.ml, mu=self.mu)
        plaw_mchirp = (np.random.power(0.5, 1) * (mchirp_upper-mchirp_lower)) + mchirp_lower
        G = 6.67e-11 # Nm^2kg^-2
        c = 3.0e8 # ms^-1
        lf = 20.0 # Hz
        tau = 5. * (8.*np.pi*lf)**(-8./3.) * (plaw_mchirp*1.989e30*G/c**3.)**(-5./3.)
        # Common steps
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)
    
    def get_bounded_gwparams_from_powerlaw_tau(self):
        # Get tau from a beta distribution
        ## Step 1d: Get power law tau
        # Go lower than 1.0 to bias toward lower values of tau
        tau_lower, tau_upper = get_tau_priors(ml=self.ml, mu=self.mu, lf=self.lf)
        tau = (np.random.power(0.5, 1) * (tau_upper-tau_lower)) + tau_lower
        # Common steps
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)
        
    def get_bounded_gwparams_from_uniform_mchirp_given_limits(self, mchirp_lower=None, mchirp_upper=None):
        ## Step 1b: Get uniform chirp mass and convert to tau
        expected_mchirp = np.random.uniform(mchirp_lower, mchirp_upper, 1)
        tau = signal_duration_from_chirp_mass(expected_mchirp, self.lf)
        # Common steps
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)

    def get_bounded_gwparams_from_template_placement_metric(self):
        # Rejection sampling method (13.37% efficiency)
        num_trials = 100 # hoping we would get at least one
        atau0 = None
        atau3 = None
        
        accepted = 0
        while not accepted:
            tau0 = np.random.uniform(self.tau0_boundary_low, self.tau0_boundary_high, num_trials)
            tau3 = np.random.uniform(self.tau3_boundary_low, self.tau3_boundary_high, num_trials)
            for t0, t3 in zip(tau0, tau3):
                if self._check_tau0_tau3_(t0, t3):
                    atau0 = t0
                    atau3 = t3
                    accepted = 1
                    break
        
        ## Get all required params
        # Converting (tau0, tau3) to (m1, m2)
        mass1 = mass1_from_tau0_tau3(atau0, atau3, f_lower=self.lf)
        mass2 = mass2_from_tau0_tau3(atau0, atau3, f_lower=self.lf)
        mchirp = mchirp_from_mass1_mass2(mass1, mass2)
        q = mass1/mass2
        return (mass1, mass2, q, mchirp, atau0)
    
    def get_bounded_gwparams_from_uniform_in_mchirp_q(self):
        # No cubic roots involved!
        # Get lines of const mchirp and find the boundaries on q
        # The lower boundary is already fixed at q=1
        min_mchirp, max_mchirp = get_mchirp_priors(self.ml, self.mu)
        umchirp = np.random.uniform(min_mchirp, max_mchirp)
        # Get mass ratio boundaries from const mchirp line
        lb_q = 1
        ub_q = self.q_upper_boundary_from_mchirp(umchirp)
        uq = np.random.uniform(lb_q, ub_q)
        # Get other params
        mass1_umcq = mass1_from_mchirp_q(umchirp, uq)
        mass2_umcq = mass2_from_mchirp_q(umchirp, uq)
        tau0 = signal_duration_from_chirp_mass(umchirp, self.lf)
        return (mass1_umcq, mass2_umcq, uq, umchirp, tau0)

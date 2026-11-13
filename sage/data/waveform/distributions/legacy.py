"""
Newtonian chirp-time conversions and mass priors bounded by
``ml <= mass2 <= mass1 <= mu``, drawn in chirp-time or chirp-mass
coordinates (NumPy).
"""

import numpy as np
from scipy.stats import beta

from pycbc.conversions import tau0_from_mass1_mass2, tau3_from_mass1_mass2
from pycbc.conversions import mass1_from_tau0_tau3, mass2_from_tau0_tau3
from pycbc.conversions import mass2_from_mchirp_mass1, mchirp_from_tau0
from pycbc.conversions import mchirp_from_mass1_mass2
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q

import lal

from sage.core.constants import GM


""" Conversions """


def q_from_mass1_mass2(mass1, mass2):
    """
    Mass ratio ``q = mass1 / mass2``.

    Parameters
    ----------
    mass1, mass2 : float or numpy.ndarray
        Component masses.

    Returns
    -------
    float or numpy.ndarray
        Mass ratio, >= 1 when ``mass1 >= mass2``.
    """
    return mass1 / mass2


def chirp_mass_from_signal_duration(tau, signal_low_freq_cutoff):
    """
    Chirp mass from the Newtonian signal duration.

    Parameters
    ----------
    tau : float or numpy.ndarray
        Signal duration in seconds.
    signal_low_freq_cutoff : float
        Lower frequency cutoff (Hz).

    Returns
    -------
    float or numpy.ndarray
        Chirp mass in solar masses.
    """
    lf = signal_low_freq_cutoff  # Hz
    chirp_mass_from_tau = ((tau / 5.0) * (8.0 * np.pi * lf) ** (8.0 / 3.0)) ** (
        -3.0 / 5.0
    ) * (1.0 / GM)
    return chirp_mass_from_tau


def signal_duration_from_chirp_mass(mchirp, signal_low_freq_cutoff):
    """
    Newtonian signal duration from the chirp mass.

    Parameters
    ----------
    mchirp : float or numpy.ndarray
        Chirp mass in solar masses.
    signal_low_freq_cutoff : float
        Lower frequency cutoff (Hz).

    Returns
    -------
    float or numpy.ndarray
        Signal duration in seconds.
    """
    lf = signal_low_freq_cutoff  # Hz
    tau_from_chirp_mass = (
        5.0
        * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
        * (mchirp * GM) ** (-5.0 / 3.0)
    )
    return tau_from_chirp_mass


def mass1_from_mchirp_q(mchirp, q):
    """
    Primary mass from the chirp mass and the mass ratio ``m1 / m2``.

    Parameters
    ----------
    mchirp : float or numpy.ndarray
        Chirp mass.
    q : float or numpy.ndarray
        Mass ratio ``m1 / m2``.

    Returns
    -------
    float or numpy.ndarray
        Primary mass.
    """
    mass1 = q ** (2.0 / 5.0) * (1.0 + q) ** (1.0 / 5.0) * mchirp
    return mass1


def mass2_from_mchirp_q(mchirp, q):
    """
    Secondary mass from the chirp mass and the mass ratio ``m1 / m2``.

    Parameters
    ----------
    mchirp : float or numpy.ndarray
        Chirp mass.
    q : float or numpy.ndarray
        Mass ratio ``m1 / m2``.

    Returns
    -------
    float or numpy.ndarray
        Secondary mass.
    """
    mass2 = q ** (-3.0 / 5.0) * (1.0 + q) ** (1.0 / 5.0) * mchirp
    return mass2


def mass1_mass2_from_mchirp_q(mchirp, q):
    """
    Component masses from the chirp mass and the mass ratio ``m1 / m2``.

    Parameters
    ----------
    mchirp : float or numpy.ndarray
        Chirp mass.
    q : float or numpy.ndarray
        Mass ratio ``m1 / m2``.

    Returns
    -------
    tuple
        ``(mass1, mass2)``.
    """
    mass1 = mass1_from_mchirp_q(mchirp, q)
    mass2 = mass2_from_mchirp_q(mchirp, q)
    return (mass1, mass2)


""" Prior bounds """


def get_mchirp_priors(ml, mu):
    """
    Chirp-mass range of the equal-mass binaries at the mass limits.

    Parameters
    ----------
    ml : float
        Lower component-mass limit.
    mu : float
        Upper component-mass limit.

    Returns
    -------
    tuple of float
        ``(min_mchirp, max_mchirp)``, the chirp masses at
        ``m1 = m2 = ml`` and at ``m1 = m2 = mu``.
    """
    min_mchirp = (ml * ml / (ml + ml) ** 2.0) ** (3.0 / 5) * (ml + ml)
    max_mchirp = (mu * mu / (mu + mu) ** 2.0) ** (3.0 / 5) * (mu + mu)
    return (min_mchirp, max_mchirp)


def get_tau_priors(ml, mu, lf):
    """
    Newtonian signal-duration range of the equal-mass binaries.

    Parameters
    ----------
    ml : float
        Lower component-mass limit.
    mu : float
        Upper component-mass limit.
    lf : float
        Lower frequency cutoff (Hz).

    Returns
    -------
    tuple of float
        ``(tau_lower, tau_upper)`` in seconds, at the largest and the
        smallest chirp mass of :func:`get_mchirp_priors`.
    """
    min_mchirp, max_mchirp = get_mchirp_priors(ml, mu)
    tau = (
        lambda mc: 5.0
        * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
        * (mc * GM) ** (-5.0 / 3.0)
    )
    tau_lower = tau(max_mchirp)
    tau_upper = tau(min_mchirp)
    return (tau_lower, tau_upper)


""" Generation """


def get_uniform_masses_with_mass1_gt_mass2(mass_lower, mass_upper, num_samples):
    """
    Component mass pairs uniform in a range, with ``mass1 > mass2``.

    Parameters
    ----------
    mass_lower : float
        Lower mass bound.
    mass_upper : float
        Upper mass bound.
    num_samples : int
        Number of pairs.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mass1, mass2)``, each of shape ``(num_samples,)``.

    Raises
    ------
    AssertionError
        If a pair has equal masses.
    """
    x_mass = [np.random.uniform(mass_lower, mass_upper, num_samples) for _ in range(2)]
    # Larger mass of each pair first
    masses = np.column_stack((x_mass[0], x_mass[1]))
    masses = np.fliplr(np.sort(masses, axis=1))
    assert all(masses[:, 0] > masses[:, 1]), "Mass1 > Mass2 in mass priors!"
    mass1 = masses[:, 0]
    mass2 = masses[:, 1]
    return (mass1, mass2)


class BoundedPriors:
    """
    Mass priors bounded by ``ml <= mass2 <= mass1 <= mu``.

    Draws in signal duration, chirp mass, (tau0, tau3) or (mchirp, q),
    using the boundaries of the mass region in those coordinates.

    Parameters
    ----------
    mu : float
        Upper component-mass limit (solar masses).
    ml : float
        Lower component-mass limit (solar masses).
    lf : float
        Signal low-frequency cutoff (Hz).
    """

    def __init__(self, mu, ml, lf):
        self.lf = lf  # Hz (signal low freq cutoff)
        self.mu = mu
        self.ml = ml
        # Newtonian duration is const * mchirp^(-5/3)
        self.const = (5.0 * (8.0 * np.pi * lf) ** (-8.0 / 3.0)) * (
            GM
        ) ** (-5.0 / 3.0)

        # (tau0, tau3) rectangle with corners at m1 = m2 = mu and ml
        self.tau3_boundary_low = tau3_from_mass1_mass2(mu, mu, f_lower=lf)
        self.tau3_boundary_high = tau3_from_mass1_mass2(ml, ml, f_lower=lf)

        self.tau0_boundary_low = tau0_from_mass1_mass2(mu, mu, f_lower=lf)
        self.tau0_boundary_high = tau0_from_mass1_mass2(ml, ml, f_lower=lf)

    def _intersection_with_m1_eq_m2(self, tau):
        """
        Component mass of the equal-mass binary with duration ``tau``.

        Parameters
        ----------
        tau : numpy.ndarray
            Newtonian signal durations in seconds.

        Returns
        -------
        numpy.ndarray
            ``m1 = m2`` on each constant-duration curve.
        """
        C_1dash = self.const * 2 ** (1.0 / 3.0)
        intersc_diagonal = (tau / C_1dash) ** (-3.0 / 5.0)
        return intersc_diagonal

    def _intersection_with_m2_when_m1_is_mu(self, tau):
        """
        ``m2`` where each constant-duration curve meets ``m1 = mu``.

        Parameters
        ----------
        tau : numpy.ndarray
            Newtonian signal durations in seconds.

        Returns
        -------
        numpy.ndarray
            The real root of the duration cubic in ``m2``.
        """
        C_2dash = self.const / self.mu
        C_3dash = (tau / C_2dash) ** 3.0
        # The cubic has one real root
        m2_when_m1_is_mu = np.array(
            [np.roots([c3d, 0.0, -1, -self.mu]) for c3d in C_3dash]
        )
        m2_when_m1_is_mu = np.array(
            [np.real(foo[np.isreal(foo)])[0] for foo in m2_when_m1_is_mu]
        )
        return m2_when_m1_is_mu

    def _intersection_with_m1_when_m2_is_ml(self, tau):
        """
        ``m1`` where each constant-duration curve meets ``m2 = ml``.

        Parameters
        ----------
        tau : numpy.ndarray
            Newtonian signal durations in seconds.

        Returns
        -------
        numpy.ndarray
            The real root of the duration cubic in ``m1``.
        """
        C_4dash = self.const / self.ml
        C_5dash = (tau / C_4dash) ** 3.0
        # The cubic has one real root
        m1_when_m2_is_ml = np.array(
            [np.roots([c5d, 0.0, -1, -self.ml]) for c5d in C_5dash]
        )
        m1_when_m2_is_ml = np.array(
            [np.real(foo[np.isreal(foo)])[0] for foo in m1_when_m2_is_ml]
        )
        return m1_when_m2_is_ml

    def _get_m1_upper_bounds(self, m2_when_m1_is_mu, m1_when_m2_is_ml):
        """
        Upper bound on ``m1`` along each constant-duration curve.

        Writes the bounds into ``m2_when_m1_is_mu`` in place.

        Parameters
        ----------
        m2_when_m1_is_mu : numpy.ndarray
            ``m2`` where each curve meets ``m1 = mu``.
        m1_when_m2_is_ml : numpy.ndarray
            ``m1`` where each curve meets ``m2 = ml``.

        Returns
        -------
        numpy.ndarray
            ``m1_when_m2_is_ml`` where ``m2_when_m1_is_mu < ml``, else
            ``mu``.
        """
        upper_bounds = m2_when_m1_is_mu
        idxs = m2_when_m1_is_mu < self.ml
        alt_idxs = m2_when_m1_is_mu >= self.ml
        upper_bounds[idxs] = m1_when_m2_is_ml[idxs]
        upper_bounds[alt_idxs] = np.full(sum(alt_idxs), self.mu)
        return upper_bounds

    def _get_m2_from_m1_tau(self, mass1, tau):
        """
        ``m2`` on the constant-duration curve of ``tau`` at ``mass1``.

        Parameters
        ----------
        mass1 : numpy.ndarray
            Primary masses.
        tau : numpy.ndarray
            Newtonian signal durations in seconds.

        Returns
        -------
        numpy.ndarray
            The real root of the duration cubic in ``m2``.
        """
        C_6dash = (tau / (self.const / mass1)) ** 3.0
        mass2_roots = np.array(
            [np.roots([c6d, 0.0, -1, -_m1]) for c6d, _m1 in zip(C_6dash, mass1)]
        )
        mass2 = np.array([np.real(foo[np.isreal(foo)])[0] for foo in mass2_roots])
        return mass2

    def _common_umc_utau(self, tau):
        """
        Masses on the constant-duration curves, uniform in m1.

        Parameters
        ----------
        tau : numpy.ndarray
            Newtonian signal durations in seconds.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mass1, mass2, q, mchirp)``.
        """
        # Bounds on m1 along each curve within the mass region
        intersc_diagonal = self._intersection_with_m1_eq_m2(tau)
        m2_when_m1_is_mu = self._intersection_with_m2_when_m1_is_mu(tau)
        m1_when_m2_is_ml = self._intersection_with_m1_when_m2_is_ml(tau)
        upper_bounds = self._get_m1_upper_bounds(m2_when_m1_is_mu, m1_when_m2_is_ml)
        lower_bounds = intersc_diagonal
        mass1 = np.array(
            [np.random.uniform(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
        )
        mass2 = self._get_m2_from_m1_tau(mass1, tau)
        q = mass1 / mass2
        mchirp = (mass1 * mass2 / (mass1 + mass2) ** 2.0) ** (3.0 / 5) * (mass1 + mass2)
        return (mass1, mass2, q, mchirp)

    def tau3_lower_boundary_from_tau0(self, _tau0):
        """
        Lower ``tau3`` boundary at ``tau0``: the equal-mass line.

        Parameters
        ----------
        _tau0 : float
            Chirp time ``tau0`` in seconds.

        Returns
        -------
        float
            ``tau3`` of the equal-mass binary (``eta = 0.25``).
        """
        A3 = np.pi / (8.0 * (np.pi * self.lf) ** (5.0 / 3.0))
        A0 = 5.0 / (256.0 * (np.pi * self.lf) ** (8.0 / 3.0))
        lower_boundary_tau3 = 4.0 * A3 * (_tau0 / (4.0 * A0)) ** (2.0 / 5.0)
        return lower_boundary_tau3

    def tau3_upper_boundary_from_tau0(self, _tau0):
        """
        Upper ``tau3`` boundary at ``tau0``.

        The edge ``m2 = ml`` above the ``tau0`` of ``(mu, ml)``, and the
        edge ``m1 = mu`` below it.

        Parameters
        ----------
        _tau0 : float
            Chirp time ``tau0`` in seconds.

        Returns
        -------
        float
            ``tau3`` on the boundary.
        """
        inflection_tau0 = tau0_from_mass1_mass2(self.mu, self.ml, f_lower=self.lf)
        if _tau0 > inflection_tau0:
            # m2 = ml and m1 in [ml, mu]
            mchirp = mchirp_from_tau0(_tau0, f_lower=self.lf)
            # Called with ml as m1, so it returns the other mass
            estimated_m1 = mass2_from_mchirp_mass1(mchirp, self.ml)
            upper_boundary_tau3 = tau3_from_mass1_mass2(
                estimated_m1, self.ml, f_lower=self.lf
            )
        elif _tau0 < inflection_tau0:
            # m1 = mu and m2 in [ml, mu]
            mchirp = mchirp_from_tau0(_tau0, f_lower=self.lf)
            estimated_m2 = mass2_from_mchirp_mass1(mchirp, self.mu)
            upper_boundary_tau3 = tau3_from_mass1_mass2(
                self.mu, estimated_m2, f_lower=self.lf
            )
        elif _tau0 == inflection_tau0:
            # Corner m1 = mu, m2 = ml
            upper_boundary_tau3 = tau3_from_mass1_mass2(
                self.mu, self.ml, f_lower=self.lf
            )
        return upper_boundary_tau3

    def q_upper_boundary_from_mchirp(self, _mchirp):
        """
        Largest mass ratio ``m1 / m2`` at chirp mass ``_mchirp``.

        Parameters
        ----------
        _mchirp : float
            Chirp mass.

        Returns
        -------
        float
            Mass ratio on the edge ``m1 = mu`` above the chirp mass of
            ``(mu, ml)``, and on the edge ``m2 = ml`` below it.
        """
        inflection_mchirp = mchirp_from_mass1_mass2(self.mu, self.ml)
        if _mchirp > inflection_mchirp:
            # m1 = mu and m2 in [ml, mu]
            estimated_m2 = mass2_from_mchirp_mass1(_mchirp, self.mu)
            upper_boundary_q = self.mu / estimated_m2
        elif _mchirp < inflection_mchirp:
            # m2 = ml and m1 in [ml, mu]; called with ml as m1, so it
            # returns the other mass
            estimated_m1 = mass2_from_mchirp_mass1(_mchirp, self.ml)
            upper_boundary_q = estimated_m1 / self.ml
        elif _mchirp == inflection_mchirp:
            # Corner m1 = mu, m2 = ml
            upper_boundary_q = self.mu / self.ml
        return upper_boundary_q

    def _check_tau0_tau3_(self, t0, t3):
        """
        Whether ``t3`` lies between the ``tau3`` boundaries at ``t0``.

        Parameters
        ----------
        t0 : float
            Chirp time ``tau0`` in seconds.
        t3 : float
            Chirp time ``tau3`` in seconds.

        Returns
        -------
        bool
            True if the point is inside the mass region.
        """
        lb_tau3 = self.tau3_lower_boundary_from_tau0(t0)
        ub_tau3 = self.tau3_upper_boundary_from_tau0(t0)
        if lb_tau3 <= t3 <= ub_tau3:
            return True
        else:
            return False

    def get_bounded_gwparams_from_uniform_tau(self):
        """
        One draw with the signal duration uniform in its range.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mass1, mass2, q, mchirp, tau)``, each of shape ``(1,)``.
        """
        tau_lower, tau_upper = get_tau_priors(ml=self.ml, mu=self.mu, lf=self.lf)
        tau = np.random.uniform(tau_lower, tau_upper, 1)
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)

    def get_bounded_gwparams_from_uniform_mchirp(self):
        """
        One draw with the chirp mass uniform in its range.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mass1, mass2, q, mchirp, tau)``, each of shape ``(1,)``.
        """
        mchirp_lower, mchirp_upper = get_mchirp_priors(ml=self.ml, mu=self.mu)
        expected_mchirp = np.random.uniform(mchirp_lower, mchirp_upper, 1)
        tau = signal_duration_from_chirp_mass(expected_mchirp, self.lf)
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)

    def get_bounded_gwparams_from_powerlaw_mchirp(self):
        """
        One draw with the chirp mass from ``np.random.power(0.5)``.

        The power-law draw is scaled to the chirp-mass range; its
        duration uses a fixed 20 Hz cutoff.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mass1, mass2, q, mchirp, tau)``, each of shape ``(1,)``.
        """
        mchirp_lower, mchirp_upper = get_mchirp_priors(ml=self.ml, mu=self.mu)
        plaw_mchirp = (
            np.random.power(0.5, 1) * (mchirp_upper - mchirp_lower)
        ) + mchirp_lower
        lf = 20.0  # Hz
        tau = (
            5.0
            * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
            * (plaw_mchirp * GM) ** (-5.0 / 3.0)
        )
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)

    def get_bounded_gwparams_from_powerlaw_tau(self):
        """
        One draw with the signal duration from ``np.random.power(0.5)``.

        The power-law draw is scaled to the duration range.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mass1, mass2, q, mchirp, tau)``, each of shape ``(1,)``.
        """
        # Exponent below 1 favours short durations
        tau_lower, tau_upper = get_tau_priors(ml=self.ml, mu=self.mu, lf=self.lf)
        tau = (np.random.power(0.5, 1) * (tau_upper - tau_lower)) + tau_lower
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)

    def get_bounded_gwparams_from_uniform_mchirp_given_limits(
        self, mchirp_lower=None, mchirp_upper=None
    ):
        """
        One draw with the chirp mass uniform in given limits.

        Parameters
        ----------
        mchirp_lower : float
            Lower chirp-mass limit.
        mchirp_upper : float
            Upper chirp-mass limit.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mass1, mass2, q, mchirp, tau)``, each of shape ``(1,)``.
        """
        expected_mchirp = np.random.uniform(mchirp_lower, mchirp_upper, 1)
        tau = signal_duration_from_chirp_mass(expected_mchirp, self.lf)
        mass1, mass2, q, mchirp = self._common_umc_utau(tau)
        return (mass1, mass2, q, mchirp, tau)

    def get_bounded_gwparams_from_template_placement_metric(self):
        """
        One draw uniform in ``(tau0, tau3)`` within the mass region.

        Rejection sampling from the bounding ``(tau0, tau3)`` rectangle.

        Returns
        -------
        tuple of float
            ``(mass1, mass2, q, mchirp, tau0)``.
        """
        num_trials = 100  # draws per round
        atau0 = None
        atau3 = None

        accepted = 0
        while not accepted:
            tau0 = np.random.uniform(
                self.tau0_boundary_low, self.tau0_boundary_high, num_trials
            )
            tau3 = np.random.uniform(
                self.tau3_boundary_low, self.tau3_boundary_high, num_trials
            )
            for t0, t3 in zip(tau0, tau3):
                if self._check_tau0_tau3_(t0, t3):
                    atau0 = t0
                    atau3 = t3
                    accepted = 1
                    break

        mass1 = mass1_from_tau0_tau3(atau0, atau3, f_lower=self.lf)
        mass2 = mass2_from_tau0_tau3(atau0, atau3, f_lower=self.lf)
        mchirp = mchirp_from_mass1_mass2(mass1, mass2)
        q = mass1 / mass2
        return (mass1, mass2, q, mchirp, atau0)

    def get_bounded_gwparams_from_uniform_in_mchirp_q(self):
        """
        One draw uniform in chirp mass, then uniform in mass ratio.

        The mass ratio is uniform between 1 and its largest value at the
        drawn chirp mass.

        Returns
        -------
        tuple of float
            ``(mass1, mass2, q, mchirp, tau0)``.
        """
        min_mchirp, max_mchirp = get_mchirp_priors(self.ml, self.mu)
        umchirp = np.random.uniform(min_mchirp, max_mchirp)
        lb_q = 1
        ub_q = self.q_upper_boundary_from_mchirp(umchirp)
        uq = np.random.uniform(lb_q, ub_q)
        mass1_umcq = mass1_from_mchirp_q(umchirp, uq)
        mass2_umcq = mass2_from_mchirp_q(umchirp, uq)
        tau0 = signal_duration_from_chirp_mass(umchirp, self.lf)
        return (mass1_umcq, mass2_umcq, uq, umchirp, tau0)

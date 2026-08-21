#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : population.py
Description   : GWTC-3 Power-Law + Peak intrinsic-parameter sampling.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The population injections are drawn from: GWTC-3's Power-Law + Peak mass model with low-
mass smoothing, a power-law mass ratio, a power-law-in-(1+z) merger-rate redshift model,
an i.i.d. Beta spin-magnitude distribution and the Gaussian-plus-isotropic spin-tilt
mixture. Each is evaluated on a grid from one hyperposterior sample and sampled by
inverting its CDF.

**Ported verbatim from sgwc-1.** Source:
``~/research/sgwc-1/notebooks/gwtc3_analyses/{gpu,cpu}_pp_intrinsic_params.py``. The
bodies below are that code, unchanged; only the module framing, the lazy import and these
notes are new. Sampling the population is not a method this search invents, and a
re-derivation would be a second implementation to keep in agreement with the first.

Both variants are kept because they are not equivalent, and which one ran matters:

- The **torch** functions are what ``injection_study.ipynb`` actually calls
  (``gpu_pp_intrinsic_params.sample_intrinsic``), so they produced the injection set that
  defined ``p(x|signal)`` for sgwc-1's p_astro.
- The **numpy** functions are the reference implementation. Its ``sample_intrinsic``
  recomputes ``p(q|m1)`` inside a python loop over all N primary masses and draws N
  samples on each pass, so at the production N = 100,000 it is O(N^2) and cannot have
  been the path that ran.

.. warning::

   **Known defect, present in both variants and deliberately preserved here.** Every
   injection's mass ratio is drawn from a single primary mass's conditional distribution
   rather than its own: the torch version indexes ``cdfs.T[:, 0]``, which is row 0, and
   the numpy version keeps only the last iteration of its loop. ``get_p_q_vec`` builds the
   full ``(N, n_q)`` matrix correctly -- the vectorisation is there -- and the sampling
   line discards every row but one. Since ``p(q|m1)`` in this model is truncated at
   ``q_min = mmin/m1``, the shapes genuinely differ with ``m1``, so this is not a
   normalisation subtlety.

   It is left in place so that this port can be checked against sgwc-1's own output. The
   fix is one line and lands separately, with a test that each injection's ``q`` follows
   its own conditional.

One deliberate difference from the ported bodies: ``np.trapz`` is spelled
``np.trapezoid``. It is the same function under its current name -- ``trapz`` is deprecated
in numpy 2 and emits a warning on every one of the 1000 iterations of :func:`CDF` -- and it
is the spelling sgwc-1's own ``injection_study.ipynb`` uses for the same routine.

``gwpopulation`` supplies the population models and is imported lazily: importing
``sage.search`` must stay free of the heavy stack, and every stage in a campaign imports
the graph.
"""

import numpy as np

_z_max = 1.35

#: Floor for a CDF normalisation. A conditional whose density integrated to zero cannot
#: be sampled from at all, and dividing by it would put nan into every mass ratio.
_TINY = 1.0e-300


def _gwpopulation():
    """
    The ``gwpopulation`` package, or a message saying what is missing and why.

    Lazy rather than top-level so that reading the stage graph does not require the
    population stack, and so a campaign that never draws injections never needs it.
    """
    try:
        import gwpopulation
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "drawing injections from the GWTC-3 Power-Law + Peak population needs the "
            "'gwpopulation' package, which supplies the mass, redshift and spin models "
            "this samples from. It is not a model reimplemented here: the population is "
            "the published one, and a second implementation would be a second thing to "
            "keep in agreement with it"
        ) from error
    return gwpopulation


# --------------------------------------------------------------------------- numpy
# Reference implementation. sgwc-1: gwtc3_analyses/cpu_pp_intrinsic_params.py


def get_p_m1(hyperpost_samp, n=1000):
    """
    Returns gwpopulation.models.mass.SinglePeakSmoothedMassDistribution from array of masses and mass ratios, with hyperposterior sample in
    form of pandas data frame.

    Parameters
    ----------
    hyperpost_samp: dict
        1 hyperposterior sample, defines shape of mass population
        ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m']
    n: int
        number of points at which to evaluate distribution

    Returns
    -------
    p_m1: numpy array
        1D array of probability values given mass given the hyperposterior sample
        This is calculated from a Powerlaw plus Peak model with low mass smoothing.
    """
    gwpopulation = _gwpopulation()

    alpha = hyperpost_samp['alpha']
    beta = hyperpost_samp['beta']
    mmin = hyperpost_samp['mmin']
    mmax = hyperpost_samp['mmax']
    lam = hyperpost_samp['lam']
    mpp = hyperpost_samp['mpp']
    sigpp = hyperpost_samp['sigpp']
    delta_m = hyperpost_samp['delta_m']

    masses = np.linspace(mmin, 100., n)
    qs = np.linspace(0., 1., n)

    param_dict = {'mass_1': masses, 'mass_ratio': qs}

    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    smoothing = mass_model.smoothing(masses, mmin=mmin, mmax=mmax, delta_m=delta_m)

    p_m1 = mass_model.p_m1(param_dict, **{'alpha': alpha, 'mmin': mmin, 'mmax': mmax,
                                          'lam': lam, 'mpp': mpp, 'sigpp': sigpp})

    smooth_p_m1 = smoothing * p_m1
    normed_p_m1 = smooth_p_m1 / np.trapezoid(smooth_p_m1, masses)

    return normed_p_m1, masses


def get_p_q(mass, hyperpost_samp, n=1000):
    """
    Parameters
    ----------
    mass: numpy array
        array of primary masses
    hyperpost_samp: dict
        1 hyperposterior sample, defines shape of mass population
        ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m']
    n: int
        number of points at which to evaluate distribution

    Returns
    -------
    p_q: numpy array
        1D array of probability values given q given the hyperposterior sample
        The mass ratio is given by a powerlaw mass ratio model with slope beta.
    """
    gwpopulation = _gwpopulation()

    qs = np.linspace(0., 1., n)
    param_dict = {'mass_1': mass, 'mass_ratio': qs}

    beta = hyperpost_samp['beta']
    mmin = hyperpost_samp['mmin']
    delta_m = hyperpost_samp['delta_m']

    q_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    p_q = q_model.p_q(param_dict, beta, mmin, delta_m) / np.trapezoid(
        q_model.p_q(param_dict, beta, mmin, delta_m), qs)

    return p_q / np.trapezoid(p_q, qs), qs


def get_p_z(hyperpost_samp, n=1000):
    """
    Parameters
    ----------
    z: numpy array
        array of redshifts
    hyperpost_samp: dict
        1 hyperposterior sample, defines shape of redshift population
        needs dict keys ['lamb']

    Returns
    -------
    p_z: numpy array
        1D array of probability values given z given the hyperposterior sample
        The mass ratio is given by a powerlaw mass ratio model with slope beta.
    """
    gwpopulation = _gwpopulation()

    lamb = hyperpost_samp['lamb']
    z = np.linspace(0., _z_max, n)
    param_dict = {'redshift': z}

    z_model = gwpopulation.models.redshift.PowerLawRedshift()
    p_z = z_model.probability(param_dict, **{'lamb': lamb})

    return p_z / np.trapezoid(p_z, z), z


def get_p_chi(hyperpost_samp, n=1000):
    """
    Parameters
    ----------
    chi: numpy array
        array of component spin magnitudes
    hyperpost_samp: dict
        1 hyperposterior sample, defines shape of mass population
        ['mu_chi', 'sigma_chi', 'xi_spin', 'sigma_spin', 'lamb', 'amax']

    Returns
    -------
    p_chi: numpy array
        1D array of probability values given chi given the hyperposterior sample
        The mass ratio is given by a powerlaw mass ratio model with slope beta.
    """
    gwpopulation = _gwpopulation()

    chi = np.linspace(0., 1., n)

    amax = hyperpost_samp['amax']
    # conversion between hyperposterior sample values and gwpopulation/gwtc-3 pop paper values:
    alpha_chi, beta_chi, amax = gwpopulation.conversions.mu_var_max_to_alpha_beta_max(
        hyperpost_samp['mu_chi'], hyperpost_samp['sigma_chi'], amax)

    param_dict = {'a_1': chi, 'a_2': chi}
    p_chi = gwpopulation.models.spin.iid_spin_magnitude_beta(
        param_dict, amax=amax, alpha_chi=alpha_chi, beta_chi=beta_chi)

    return p_chi / np.trapezoid(p_chi, chi), chi


def get_p_costilt(hyperpost_samp, n=1000):
    """
    Parameters
    ----------
    cos_tilt: numpy array
        array of componant cosine spin tilts
    hyperpost_samp: dict
        1 hyperposterior sample, defines shape of mass population
        ['mu_chi', 'sigma_chi', 'xi_spin', 'sigma_spin', 'lamb', 'amax']

    Returns
    -------
    p_costilt: numpy array
        1D array of probability values given chi given the hyperposterior sample
        The spin tilt is given by a mixture model with a gaussian and a isotropic component.
    """
    gwpopulation = _gwpopulation()

    cos_tilt = np.linspace(-1., 1., n)

    xi_spin = hyperpost_samp['xi_spin']
    sigma_spin = hyperpost_samp['sigma_spin']

    param_dict = {'cos_tilt_1': cos_tilt, 'cos_tilt_2': cos_tilt}
    p_costilt = gwpopulation.models.spin.iid_spin_orientation_gaussian_isotropic(
        param_dict, xi_spin, sigma_spin)

    return p_costilt / np.trapezoid(p_costilt, cos_tilt), cos_tilt


def CDF(distr, theta):
    """
    Calculates a CDF of a PDF (distr) supplied at points theta
    """
    CDF = []
    for i in range(len(theta)):
        CDF.append(np.trapezoid(distr[:i + 1], theta[:i + 1]))
    return CDF, theta


def sample_1D(distr, theta, N):
    """
    returns samples from distribution calculated by interpolation of a given distribution

    Parameters
    ----------
    distr: numpy array
        values of a probability distribution of one of the binary parameters
    theta: numpy
        values of binary parameter at locations of probability distribution
    N: int
        number of samples to return from distr

    Returns
    -------
    samps: numpy array
        samples from distr
    """
    rand = np.random.random(N)
    CDF_theta, theta = CDF(distr, theta)
    samps = np.interp(rand, CDF_theta, theta)
    return samps


def sample_intrinsic(hyperpost_samp, N):
    """
    Draw ``(m1, q, z, chi_1, chi_2, costilt_1, costilt_2)`` for N binaries.

    Reference implementation, and slow: ``get_p_q`` is evaluated once per injection, so
    a set of any size is better drawn with :func:`sample_intrinsic_torch`, which is the
    path a campaign uses.
    """
    sample = np.zeros((N, 7))  # {'m1':0, 'q':0, 'z':0, 'chi_1':0, 'chi_2':0, 'costilt_1':0, 'costilt_2':0}

    # mass model
    p_m1, masses = get_p_m1(hyperpost_samp)
    sample[:, 0] = sample_1D(p_m1, masses, N)

    # mass ratio, one draw per primary from that primary's own conditional
    for i, m1 in enumerate(sample[:, 0]):
        p_q, qs = get_p_q(np.array([m1]), hyperpost_samp)
        sample[i, 1] = sample_1D(p_q, qs, 1)[0]

    # redshift
    p_z, zs = get_p_z(hyperpost_samp)
    sample[:, 2] = sample_1D(p_z, zs, N)

    # spin magnitudes
    # chi_1 and chi_2 sampled the same spin population
    p_chi, chis = get_p_chi(hyperpost_samp)
    sample[:, 3] = sample_1D(p_chi, chis, N)
    sample[:, 4] = sample_1D(p_chi, chis, N)

    # spin tilts
    p_costilt, costilts = get_p_costilt(hyperpost_samp)
    sample[:, 5] = sample_1D(p_costilt, costilts, N)
    sample[:, 6] = sample_1D(p_costilt, costilts, N)

    return sample


def sample_m1(hyperpost_samp, N):
    """Primary masses alone, for a mass-only study."""
    p_m1, masses = get_p_m1(hyperpost_samp)
    samples = sample_1D(p_m1, masses, N)
    return samples


# --------------------------------------------------------------------------- torch
# The path that ran. sgwc-1: gwtc3_analyses/gpu_pp_intrinsic_params.py
#
# Named with a _torch suffix so both variants live in one module: sgwc-1 kept them in two
# files imported under different names, and a single module means the numpy reference and
# the tensor implementation cannot drift apart unnoticed. The bodies are unchanged.
#
# The device is resolved per call rather than fixed at import. sgwc-1 bound
# `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` at module scope,
# which pins the choice to whatever the importing process saw first -- and importing this
# module then initialises CUDA, which a login-node campaign reading the stage graph must
# not do.


def _torch_device(device=None):
    """The device to build grids on, defaulting to sgwc-1's cuda-if-available choice."""
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trapz_torch(y, x, dim=-1):
    """Trapezoidal integral along ``dim``."""
    import torch

    return torch.trapz(y, x, dim=dim)


def interp1d_grid_sample(x, xp, fp):
    """
    Vectorised 1D linear interpolation in PyTorch.

    Parameters
    ----------
    x : tensor of shape (N,)
        Query points.
    xp : tensor of shape (M,)
        Grid points (does not need to be uniform but must be sorted ascending).
    fp : tensor of shape (M,)
        Values at xp.

    Returns
    -------
    tensor of shape (N,)
        Interpolated values at x.
    """
    import torch

    # Clamp x into [xp[0], xp[-1]] to avoid extrapolation
    x_clamped = torch.clamp(x, xp[0], xp[-1])

    # Find the right interval for each x
    # idx[i] is the index of the first xp larger than x[i]
    idx = torch.searchsorted(xp, x_clamped, right=False)
    idx = torch.clamp(idx, 1, len(xp) - 1)  # ensure 1 <= idx <= M-1

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x_clamped - x0)


def interp1d_rows(x, xp_rows, fp):
    """
    Vectorised 1D linear interpolation with one grid **per row**.

    :func:`interp1d_grid_sample` interpolates every query against a single grid. Inverse
    transform sampling from a *conditional* density needs one grid per query, because
    each row has its own CDF -- which is the whole point of ``p(q | m1)``.

    Parameters
    ----------
    x : tensor of shape (N,)
        Query points, one per row.
    xp_rows : tensor of shape (N, M)
        Row ``i``'s grid points, sorted ascending along the last axis.
    fp : tensor of shape (M,)
        Values at the grid points, shared by every row. The mass-ratio grid is common to
        all rows; only the CDF over it differs.

    Returns
    -------
    tensor of shape (N,)
        Row ``i``'s interpolation of ``x[i]`` against ``xp_rows[i]``.
    """
    import torch

    x = x.unsqueeze(-1)
    x = torch.clamp(x, xp_rows[:, :1], xp_rows[:, -1:])

    index = torch.searchsorted(xp_rows.contiguous(), x, right=False)
    index = torch.clamp(index, 1, xp_rows.shape[-1] - 1)

    x0 = xp_rows.gather(-1, index - 1)
    x1 = xp_rows.gather(-1, index)
    y0 = fp[index - 1]
    y1 = fp[index]

    # A conditional CDF is flat wherever the density vanishes, which for p(q | m1) is
    # everything below q_min = mmin/m1 -- a wide interval for a heavy primary. A query
    # landing exactly on a flat step would divide by zero, so those fall back to the
    # left-hand grid value rather than producing a nan that survives into the waveform.
    width = x1 - x0
    slope = torch.where(width > 0, (y1 - y0) / torch.where(width > 0, width, width + 1.0),
                        torch.zeros_like(width))
    return (y0 + slope * (x - x0)).squeeze(-1)


def get_p_m1_torch(hyperpost_samp, n=1000, device=None):
    """Power-Law + Peak primary-mass density on a grid, as tensors."""
    import torch

    gwpopulation = _gwpopulation()
    device = _torch_device(device)

    alpha = hyperpost_samp['alpha']
    mmin = hyperpost_samp['mmin']
    mmax = hyperpost_samp['mmax']
    lam = hyperpost_samp['lam']
    mpp = hyperpost_samp['mpp']
    sigpp = hyperpost_samp['sigpp']
    delta_m = hyperpost_samp['delta_m']

    masses = torch.linspace(mmin, 100., n, device=device)
    qs = torch.linspace(0., 1., n, device=device)

    param_dict = {'mass_1': masses.cpu().numpy(), 'mass_ratio': qs.cpu().numpy()}

    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    smoothing = torch.tensor(mass_model.smoothing(masses.cpu().numpy(),
                                                  mmin=mmin, mmax=mmax, delta_m=delta_m),
                             device=device)

    p_m1 = torch.tensor(mass_model.p_m1(param_dict, alpha=alpha, mmin=mmin,
                                        mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp),
                        device=device)

    smooth_p_m1 = smoothing * p_m1
    normed_p_m1 = smooth_p_m1 / trapz_torch(smooth_p_m1, masses)

    return normed_p_m1, masses


def get_p_q_vec(masses, hyperpost_samp, n=1000, device=None):
    """
    Vectorised p_q evaluation for many m1 values.
    Returns shape (len(masses), n_q)

    Parameters
    ----------
    mass: numpy array
        array of primary masses
    hyperpost_samp: dict
        1 hyperposterior sample, defines shape of mass population
        ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m']
    n: int
        number of points at which to evaluate distribution

    Returns
    -------
    p_q: numpy array
        1D array of probability values given q given the hyperposterior sample
        The mass ratio is given by a powerlaw mass ratio model with slope beta.
    """
    import torch

    gwpopulation = _gwpopulation()
    device = _torch_device(device)

    qs = torch.linspace(0., 1., n, device=device)
    m1_grid = masses[:, None].repeat(1, n).cpu().numpy()
    q_grid = qs[None, :].repeat(len(masses), 1).cpu().numpy()

    beta = hyperpost_samp['beta']
    mmin = hyperpost_samp['mmin']
    delta_m = hyperpost_samp['delta_m']

    q_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    pq_np = q_model.p_q({'mass_1': m1_grid.ravel(),
                         'mass_ratio': q_grid.ravel()},
                        beta, mmin, delta_m)
    pq = torch.tensor(pq_np.reshape(len(masses), n), device=device)

    # Normalise each row over q
    pq = pq / trapz_torch(pq, qs, dim=-1).unsqueeze(-1)

    return pq, qs


def get_p_z_torch(hyperpost_samp, n=1000, device=None):
    """Power-law-in-(1+z) merger-rate redshift density on a grid, as tensors."""
    import torch

    gwpopulation = _gwpopulation()
    device = _torch_device(device)

    lamb = hyperpost_samp['lamb']
    z = torch.linspace(0., _z_max, n, device=device)
    param_dict = {'redshift': z.cpu().numpy()}

    z_model = gwpopulation.models.redshift.PowerLawRedshift()
    pz_np = z_model.probability(param_dict, lamb=lamb)
    p_z = torch.tensor(pz_np, device=device)

    return p_z / trapz_torch(p_z, z), z


def get_p_chi_torch(hyperpost_samp, n=1000, device=None):
    """i.i.d. Beta spin-magnitude density on a grid, as tensors."""
    import torch

    gwpopulation = _gwpopulation()
    device = _torch_device(device)

    chi = torch.linspace(0., 1., n, device=device)

    amax = hyperpost_samp['amax']
    alpha_chi, beta_chi, amax = gwpopulation.conversions.mu_var_max_to_alpha_beta_max(
        hyperpost_samp['mu_chi'], hyperpost_samp['sigma_chi'], amax)

    param_dict = {'a_1': chi.cpu().numpy(), 'a_2': chi.cpu().numpy()}
    pchi_np = gwpopulation.models.spin.iid_spin_magnitude_beta(
        param_dict, amax=amax, alpha_chi=alpha_chi, beta_chi=beta_chi)
    p_chi = torch.tensor(pchi_np, device=device)

    return p_chi / trapz_torch(p_chi, chi), chi


def get_p_costilt_torch(hyperpost_samp, n=1000, device=None):
    """Gaussian-plus-isotropic spin-tilt mixture on a grid, as tensors."""
    import torch

    gwpopulation = _gwpopulation()
    device = _torch_device(device)

    cos_tilt = torch.linspace(-1., 1., n, device=device)

    xi_spin = hyperpost_samp['xi_spin']
    sigma_spin = hyperpost_samp['sigma_spin']

    param_dict = {'cos_tilt_1': cos_tilt.cpu().numpy(),
                  'cos_tilt_2': cos_tilt.cpu().numpy()}
    pcostilt_np = gwpopulation.models.spin.iid_spin_orientation_gaussian_isotropic(
        param_dict, xi_spin, sigma_spin)
    p_costilt = torch.tensor(pcostilt_np, device=device)

    return p_costilt / trapz_torch(p_costilt, cos_tilt), cos_tilt


def CDF_torch(distr, theta, device=None):
    """Cumulative trapezoidal integral of ``distr`` over ``theta``."""
    import torch

    device = _torch_device(device)
    cdf_vals = torch.cumsum(
        torch.cat([torch.tensor([0.], device=device),
                   (distr[1:] + distr[:-1]) / 2 * (theta[1:] - theta[:-1])]),
        dim=0)
    return cdf_vals, theta


def sample_1D_torch(distr, theta, N, device=None):
    """Inverse-CDF samples from a gridded density."""
    import torch

    device = _torch_device(device)
    rand = torch.rand(N, device=device)
    CDF_theta, theta = CDF_torch(distr, theta, device=device)
    samps = interp1d_grid_sample(rand, CDF_theta, theta)
    return samps


def sample_intrinsic_torch(hyperpost_samp, N, device=None):
    """
    Draw ``(m1, q, z, chi_1, chi_2, costilt_1, costilt_2)`` for N binaries.

    The path ``injection_study.ipynb`` calls, and therefore the one that produced the
    injection set behind sgwc-1's ``p(x|signal)`` -- with its mass-ratio defect corrected.
    Every injection's mass ratio is drawn from *its own* ``p(q | m1)``, which is what the
    ``(N, n_q)`` matrix ``get_p_q_vec`` returns was for; sgwc-1 computed it and then read
    one row. See SB-1.
    """
    import torch

    device = _torch_device(device)
    sample = torch.zeros((N, 7), device=device)

    # mass model
    p_m1, masses = get_p_m1_torch(hyperpost_samp, device=device)
    sample[:, 0] = sample_1D_torch(p_m1, masses, N, device=device)

    # vectorised mass ratio, each injection against its own conditional
    pq_all, qs = get_p_q_vec(sample[:, 0], hyperpost_samp, device=device)
    rand = torch.rand(N, device=device)
    cdfs = torch.cumsum(pq_all[:, 1:] * (qs[1:] - qs[:-1]), dim=-1)
    cdfs = torch.cat([torch.zeros(N, 1, device=device), cdfs], dim=-1)
    # Normalised per row: the rows are integrated by rectangles while the densities were
    # normalised by trapezoids, so a row's CDF ends near one rather than at it, and a
    # uniform draw above that endpoint would clamp to q = 1.
    cdfs = cdfs / cdfs[:, -1:].clamp_min(_TINY)
    sample[:, 1] = interp1d_rows(rand, cdfs, qs)

    # redshift
    p_z, zs = get_p_z_torch(hyperpost_samp, device=device)
    sample[:, 2] = sample_1D_torch(p_z, zs, N, device=device)

    # spin magnitudes
    p_chi, chis = get_p_chi_torch(hyperpost_samp, device=device)
    sample[:, 3] = sample_1D_torch(p_chi, chis, N, device=device)
    sample[:, 4] = sample_1D_torch(p_chi, chis, N, device=device)

    # spin tilts
    p_costilt, costilts = get_p_costilt_torch(hyperpost_samp, device=device)
    sample[:, 5] = sample_1D_torch(p_costilt, costilts, N, device=device)
    sample[:, 6] = sample_1D_torch(p_costilt, costilts, N, device=device)

    return sample


#: Column order of :func:`sample_intrinsic` and :func:`sample_intrinsic_torch`, matching
#: the dtype ``injection_study.ipynb`` builds from them.
def plan_marginalisation(n_samples: int, n_available: int):
    """
    How many hyperposterior samples to draw from, and how many injections under each.

    Ported from Thyme's ``_plan_marginalisation``. Distinct posterior points are the
    scarce thing, so it takes as many as it can and only repeats when there are fewer
    posterior samples than injections wanted.

    Returns
    -------
    (n_hyper, n_per_hyper)
        ``n_hyper * n_per_hyper >= n_samples``; the caller truncates.
    """
    import math

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_available <= 0:
        raise ValueError("the hyperposterior holds no samples to marginalise over")
    n_hyper = min(int(n_samples), int(n_available))
    return n_hyper, math.ceil(int(n_samples) / n_hyper)


def sample_intrinsic_marginalised(
    hyperposterior, N, n_hyper=None, device=None, seed=None, progress=False
):
    """
    Draw intrinsic parameters marginalised over the population hyperposterior.

    Conditioning on one hyperposterior sample states a population the data merely
    prefers, and hands the injection set a confidence the inference does not have.
    Drawing each block of injections under a *different* posterior sample propagates that
    uncertainty into the set instead, which is what Thyme's population pipeline defaults
    to.

    It matters here more than it might elsewhere, because these injections define
    ``p(x | signal)`` for p_astro. A signal density built at a single hyperposterior point
    is narrower than the astrophysical uncertainty warrants, and every candidate's
    probability inherits that.

    Parameters
    ----------
    hyperposterior : sequence of mapping
        One mapping of hyperparameters per posterior sample, as
        :func:`sage.search.sources.gwtc3_powerlawpeak.population` returns.
    N : int
        Injections wanted. The result is exactly this many.
    n_hyper : int, optional
        Posterior samples to draw from. Defaults to :func:`plan_marginalisation`, which
        uses as many distinct ones as there are injections.
    seed : int, optional
        Seeds the choice of posterior samples and the shuffle. The per-injection draws
        follow torch's global generator, as the single-sample path does.

    Returns
    -------
    torch.Tensor
        ``(N, 7)`` in :data:`INTRINSIC_COLUMNS` order.
    """
    import numpy as np
    import torch

    device = _torch_device(device)
    samples = list(hyperposterior)
    n_available = len(samples)
    if n_hyper is None:
        n_hyper, n_per = plan_marginalisation(int(N), n_available)
    else:
        n_hyper = int(n_hyper)
        if n_hyper <= 0:
            raise ValueError(f"n_hyper must be positive, got {n_hyper}")
        n_per = int(np.ceil(int(N) / n_hyper))

    rng = np.random.default_rng(seed)
    if n_hyper <= n_available:
        # Without replacement: every block comes from a distinct posterior point, which
        # is the diversity the marginalisation exists to buy.
        chosen = rng.choice(n_available, size=n_hyper, replace=False)
    else:
        chosen = rng.integers(0, n_available, size=n_hyper)

    blocks = []
    iterator = range(n_hyper)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="hyperposterior marginalisation")
    for i in iterator:
        blocks.append(
            sample_intrinsic_torch(samples[int(chosen[i])], n_per, device=device)
        )
    drawn = torch.cat(blocks, dim=0)

    # Shuffled before truncation, so no posterior point is systematically dropped when
    # n_hyper * n_per overshoots N -- otherwise the last blocks are the ones cut.
    order = torch.as_tensor(rng.permutation(drawn.shape[0]), device=drawn.device)
    return drawn[order][: int(N)]


INTRINSIC_COLUMNS = ("m1", "q", "z", "chi_1", "chi_2", "costilt_1", "costilt_2")

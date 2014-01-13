import numpy as np
from .. import SimultaneousSampler, DimAtATimeSampler


def log_single_gaussian(params, tau, mu):
    lnp = 0.5*(np.log(tau))
    lnp += -0.5*tau*(params[0]-mu)**2
    return lnp


def log_five_standard_gaussians(params):
    lnp = -0.5*np.sum(params**2)
    return lnp


def log_5d_standard_normal_dimatatime(params, dimension):
    lnp = -0.5*params[dimension]**2
    return lnp


def test_shrink_1d():
    x0 = 0.1
    w = 2.5
    sampler = SimultaneousSampler(1, log_single_gaussian, w, 
                           method='shrink', 
                           args=[1,0])
    sampler.run(x0, 500)
    sampler.run(sampler.samples[-1,:], 3)
    assert sampler.samples.shape[0] == 503

    
def test_shrink_md():
    x0 = np.random.normal(0, 1, 5)
    w = 2.5
    sampler = SimultaneousSampler(5, log_five_standard_gaussians, 
                           w, method='shrink')
    sampler.run(x0, 500)
    sampler.run(sampler.samples[-1,:], 3)
    assert sampler.samples.shape[0] == 503    

    
def test_crumb_1d():
    x0 = 0.1
    sigma = 2.5
    sampler = SimultaneousSampler(1, log_single_gaussian, sigma, 
                           method='crumb', 
                           args=[1,0])
    sampler.run(x0, 500)
    sampler.run(sampler.samples[-1,:], 3)
    assert sampler.samples.shape[0] == 503


def test_shrink_dt():
    x0 = np.random.normal(0, 1, 5)
    ws = np.zeros(5) + 0.1
    log_conditionals = [log_5d_standard_normal_dimatatime for dim in range(5)]
    args = [[dim] for dim in range(5)]
    sampler = DimAtATimeSampler(5, log_conditionals, 
                           ws, args)
    sampler.run(x0, 500)
    sampler.run(sampler.samples[-1,:], 3)
    assert sampler.samples.shape[0] == 503

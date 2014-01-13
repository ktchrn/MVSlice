import numpy as np
from .. import sampling as s
from numpy.testing import assert_allclose


def log_single_gaussian(params, tau, mu):
    lnp = 0.5*(np.log(tau))
    lnp += -0.5*tau*(params[0]-mu)**2
    return lnp


def log_5d_standard_normal(params):
    lnp = -0.5*np.sum(params**2)
    return lnp


def log_5d_standard_normal_dimatatime(params, dimension):
    lnp = -0.5*params[dimension]**2
    return lnp


def test_draw_slice_level():
    g_x0 = np.random.standard_cauchy()
    level = s.draw_slice_level(g_x0)
    assert g_x0 >= level

    
def test_place_hyperrect():
    x0 = np.random.rand(5)
    w = np.random.uniform(2, 4, 5)
    L, R = s.place_hyperrect(x0, w)
    assert_allclose(R-L, w)
    assert (L <= x0).all()
    assert (R >= x0).all()
    

def test_shrink_hyperrect():
    x0 = np.random.rand(5)
    w0 = np.random.uniform(2, 4, 5)
    L, R = s.place_hyperrect(x0, w0)
    x1 = np.random.uniform(L, R, 5)
    L, R = s.shrink_hyperrect(x0, x1, L, R)
    w1 = R - L
    assert (w1 > 0).all()
    assert (w1 <= w0).all()
    assert (w1 < w0).any()
    assert (x0 <= R).all()
    assert (x0 >= L).all()


def test_simultaneous_sample_2d():
    x0 = np.random.normal(0, 1, 5)
    g_x0 = log_5d_standard_normal(x0)
    w = 2.5
    x1, g_x1, n_attempts = s.shrink_sample(x0, g_x0, log_5d_standard_normal, 
                                           w)
    assert (np.abs(x1 - x0) <= w).all()
    assert n_attempts > 0
    sigma = 2.5
    x1, g_x1, n_attempts = s.crumb_sample(x0, g_x0, log_5d_standard_normal, 
                                           sigma)
    assert n_attempts > 0

    
def test_simultaneous_sample_1d():
    mu = 0.
    tau = 1.
    args = [tau, mu]
    x0 = np.random.normal(0, 1, 1)
    g_x0 = log_single_gaussian(x0, tau, mu)
    w = 2.5
    x1, g_x1, n_attempts = s.shrink_sample(x0, g_x0, log_single_gaussian, w, 
                                           args=args)
    assert (np.abs(x1 - x0) <= w).all()
    assert n_attempts > 0
    

def test_draw_crumb():
    mean = 0
    sigma = 2.5
    x0 = np.random.rand(5)
    s.draw_crumb(x0, mean, sigma, 1)
    s.draw_crumb(x0, mean, sigma, 5)


def test_double_dt():
    log_conditional = log_5d_standard_normal_dimatatime
    dim = 1
    args = [dim]
    w = 0.1
    x0 = np.random.normal(0, 1, 5)
    g_x0 = log_conditional(x0, *args)
    z = s.draw_slice_level(g_x0)
    L, R = s.double_dt(x0, z, w, dim, log_conditional, args)
    xL = x0.copy()
    xR = x0.copy()
    xL[dim] = L
    xR[dim] = R
    g_L = log_conditional(xL, *args)
    g_R = log_conditional(xR, *args)
    assert g_L <= z
    assert g_R <= z
    assert L < x0[dim]
    assert R > x0[dim]


def test_shrink_dt_attempt():
    log_conditional = log_5d_standard_normal_dimatatime
    dim = 1
    args = [dim]
    w = 0.1
    x0 = np.random.normal(0, 1, 5)
    g_x0 = log_conditional(x0, *args)
    z = s.draw_slice_level(g_x0)
    L, R = s.double_dt(x0, z, w, dim, log_conditional, args)
    x1, accept = s.shrink_dt_attempt(x0, z, dim, log_conditional, L, R, args)
    g_x1 = log_conditional(x1, *args)
    should_accept = (g_x1 > z)
    assert should_accept == accept 


def test_shrink_dt_sample():
    log_conditionals = [log_5d_standard_normal_dimatatime for dim in range(5)]
    args = [[dim] for dim in range(5)]
    w = np.zeros(5) + 0.1
    x0 = np.random.normal(0, 1, 5)
    x1, n_attempts = s.shrink_dt_sample(x0, log_conditionals, w, args)
    assert n_attempts > 0
    

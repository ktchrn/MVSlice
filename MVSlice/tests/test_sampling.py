import numpy as np
from .. import sampling as s
from numpy.testing import assert_allclose

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


#1d gaussian
def log_single_gaussian(params, tau, mu):
    lnp = 0.5*(np.log(tau))
    lnp += -0.5*tau*(params[0]-mu)**2
    return lnp

def log_five_standard_gaussians(params):
    lnp = -0.5*np.sum(params**2)
    return lnp


def test_sample_2d():
    x0 = np.random.normal(0, 1, 5)
    g_x0 = log_five_standard_gaussians(x0)
    w = 2.5
    x1, g_x1, n_attempts = s.shrink_sample(x0, g_x0, log_five_standard_gaussians, 
                                           w, count_attempts=True)
    assert (np.abs(x1 - x0) <= w).all()
    assert n_attempts > 0
    sigma = 2.5
    x1, g_x1, n_attempts = s.crumb_sample(x0, g_x0, log_five_standard_gaussians, 
                                           sigma, count_attempts=True)
    assert n_attempts > 0

    
def test_sample_1d():
    mu = 0.
    tau = 1.
    args = [tau, mu]
    x0 = np.random.normal(0, 1, 1)
    g_x0 = log_single_gaussian(x0, tau, mu)
    w = 2.5
    x1, g_x1, n_attempts = s.shrink_sample(x0, g_x0, log_single_gaussian, w, 
                                           args=args, count_attempts=True)
    assert (np.abs(x1 - x0) <= w).all()
    assert n_attempts > 0
    

def test_draw_crumb():
    mean = 0
    sigma = 2.5
    x0 = np.random.rand(5)
    s.draw_crumb(x0, mean, sigma, 1)
    s.draw_crumb(x0, mean, sigma, 5)

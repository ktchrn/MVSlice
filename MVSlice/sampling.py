import numpy as np


def draw_slice_level(g_x0):
    """
    Draw a value for z=log(y) from the exponential distribution.

    z = log(p(x0)) - e
      = g_x0 - e
    e ~ exp(1)
    """
    e = np.random.exponential(1)
    z = g_x0 - e
    return z


def place_hyperrect(x0, w):
    """

    """
    ndim = x0.size
    U = np.random.rand(ndim)
    L = x0 - U*w
    R = L + w
    return L, R
    

def shrink_hyperrect(x0, x1, L, R):
    """
    
    """
    L_or_R = (x1 >= x0) #Modifications to R
    R[L_or_R] = x1[L_or_R]
    np.not_equal(L_or_R, True, L_or_R) #Modifications to L
    L[L_or_R] = x1[L_or_R]
    return L, R


def shrink_attempt(z, log_posterior, L, R, args):
    """
    
    """
    ndim = L.size
    x1 = np.random.uniform(L, R, ndim)
    g_x1 = log_posterior(x1, *args)
    accept = (g_x1 > z)
    return x1, g_x1, accept
    
    
    
def shrink_sample(x0, g_x0, log_posterior, w, 
                  args=[]):
    """
    
    """
    z = draw_slice_level(g_x0)
    _x0 = np.atleast_1d(np.asarray(x0))
    L, R = place_hyperrect(_x0, w)
    accepted = False
    n_attempts = 0
    while not accepted:
        n_attempts += 1
        x1, g_x1, accepted = shrink_attempt(z, log_posterior, L, R, args)
        L, R = shrink_hyperrect(_x0, x1, L, R)
    return x1, g_x1, n_attempts

    
def draw_crumb(x0, mean, sigma, iteration):
    """

    """
    ndim = x0.size
    new_crumb = np.random.normal(x0, sigma, ndim)
    mean = mean + (new_crumb - mean)/iteration
    return mean


def crumb_attempt(z, log_posterior, mean, sigma, args):
    """

    """
    ndim = mean.size
    x1 = np.random.normal(mean, sigma, ndim)
    g_x1 = log_posterior(x1, *args)
    return x1, g_x1, (g_x1 > z)


def crumb_sample(x0, g_x0, log_posterior, sigma, 
                 args=[]):
    """

    """
    z = draw_slice_level(g_x0)
    _x0 = np.atleast_1d(np.asarray(x0))
    accepted = False
    n_attempts = 0
    mean = np.zeros(_x0.size)
    while not accepted:
        n_attempts += 1
        mean = draw_crumb(_x0, mean, sigma, n_attempts)
        x1, g_x1, accepted = crumb_attempt(z, log_posterior, mean, 
                                 sigma/np.sqrt(n_attempts), args)
    return x1, g_x1, n_attempts


def double_dt(x0, z, w, dim, log_conditional, args):
    xL = x0.copy()
    xR = x0.copy()
    uR = np.random.rand()
    uL = uR - 1
    xL[dim] = x0[dim] + uL*w
    xR[dim] = x0[dim] + uR*w
    g_XL = log_conditional(x0, *args)
    g_XR = log_conditional(x0, *args)
    while (g_XL > z) or (g_XR > z):
        uR *= 2
        uL *= 2
        xL[dim] = x0[dim] + uL*w
        xR[dim] = x0[dim] + uR*w
        g_XL = log_conditional(xL, *args)
        g_XR = log_conditional(xR, *args)
    L = x0[dim] + uL*w
    R = x0[dim] + uR*w
    return L, R


def shrink_dt_attempt(x0, z, dim, log_conditional, L, R, args):
    x1 = x0.copy()
    x1[dim] = np.random.uniform(L, R)
    g_x1 = log_conditional(x1, *args)
    accept = (g_x1 > z)
    return x1, accept


def shrink_dt_sample(x0, log_conditionals, w, 
                         args):
    _x0 = np.atleast_1d(np.asarray(x0)).copy()
    ndim = _x0.size
    n_attempts = 0
    for dim in range(ndim):
        log_conditional = log_conditionals[dim]
        _args = args[dim]
        g_x0 = log_conditional(_x0, *_args)
        z = draw_slice_level(g_x0)
        L, R = double_dt(_x0, z, w[dim], dim, log_conditional, _args)
        accepted = False
        while not accepted:
            n_attempts += 1
            x1, accepted = shrink_dt_attempt(_x0, z, dim, log_conditional,
                                             L, R, _args)
            if x1[dim] < x0[dim]:
                L = x1[dim]
            else:
                R = x1[dim]
        _x0 = x1.copy()
    return _x0, n_attempts

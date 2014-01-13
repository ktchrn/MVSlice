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
                  args=[], count_attempts=False):
    """
    
    """
    z = draw_slice_level(g_x0)
    _x0 = np.atleast_1d(np.asarray(x0))
    L, R = place_hyperrect(_x0, w)
    accepted = False
    if count_attempts:
        n_attempts = 0
    while not accepted:
        if count_attempts:
            n_attempts += 1
            
        x1, g_x1, accept = shrink_attempt(z, log_posterior, L, R, args)
        if accept:
            accepted = True
        else:
            L, R = shrink_hyperrect(_x0, x1, L, R)
    if count_attempts:
        return x1, g_x1, n_attempts
    else:
        return x1, g_x1

    
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
                 args=[], count_attempts=False):
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
    if count_attempts:
        return x1, g_x1, n_attempts
    else:
        return x1, g_x1

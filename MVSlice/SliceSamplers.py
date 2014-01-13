from .sampling import shrink_sample, crumb_sample, shrink_dt_sample

import numpy as np

class SliceSampler(object):
    def __init__(self, *args, **kwargs):
        pass

    
    @property
    def chain_length(self):
        _chain_length = self.samples.shape[0]
        if self._started:
            return _chain_length
        else:
            return 0
        

    def _resize_chains(self, niter):
        """

        """
        current_niter = self.chain_length
        new_niter = current_niter + niter
        
        _samples = np.zeros([new_niter, self.ndim])
        _samples[0:current_niter] = self.samples.copy()
        self.samples = _samples

        _n_attempts = np.zeros([new_niter])
        _n_attempts[0:current_niter] = self.n_attempts.copy()
        self.n_attempts = _n_attempts

        try:
            _log_p = np.zeros([new_niter])
            _log_p[0:current_niter] = self.log_posteriors.copy()
            self.log_posteriors = _log_p
        except:
            #dim at a time sampler doesn't track log_p
            pass


class SimultaneousSampler(SliceSampler):
    def __init__(self, ndim, log_posterior,
                 method_pars, method='shrink',
                 args=[]):
        """

        """
        self.ndim = ndim
        self.log_posterior_func = log_posterior

        self.method_pars = method_pars
        if method == 'shrink':
            self.sample = shrink_sample
        elif method == 'crumb':
            self.sample = crumb_sample

        self.samples = np.atleast_2d(np.empty([1, self.ndim]))
        self.log_posteriors = np.empty([1])

        self._started = False
        self.n_attempts = np.empty([1])

        self.args = args
        
    
    def run(self, x0, niter=1, burn=0, thin=1):
        """

        """
        _x0 = np.atleast_1d(np.asarray(x0).copy())
        start_iter = self.chain_length
        end_iter = start_iter + niter
        self._resize_chains(niter)
        self._started = True
        
        g_x0 = self.log_posterior_func(_x0, *self.args)
        for iter_i in range(start_iter, end_iter):
            out = self.sample(_x0, g_x0, 
                              self.log_posterior_func, 
                              self.method_pars,
                              args=self.args)
            self.samples[iter_i] = out[0]
            self.log_posteriors[iter_i] = out[1]
            self.n_attempts[iter_i] = out[2]
            _x0 = out[0].copy()
            g_x0 = out[1]

            
class DimAtATimeSampler(SliceSampler):
    def __init__(self, ndim, log_conditionals,
                 ws, args):
        """

        """
        self.ndim = ndim

        assert len(ws) == ndim, "Wrong number of widths."
        assert len(log_conditionals) == ndim, \
          "Wrong number of conditional probability functions."
        assert len(args) == ndim, "Wrong number of function arguments"

        self.sample = shrink_dt_sample
        
        self.samples = np.atleast_2d(np.empty([1, self.ndim]))
        self._started = False
        self.n_attempts = np.empty([1])

        self.args = args
        self.ws = ws
        self.log_conditional_funcs = log_conditionals
        
    
    def run(self, x0, niter=1, burn=0, thin=1):
        """

        """
        _x0 = np.atleast_1d(np.asarray(x0).copy())
        start_iter = self.chain_length
        end_iter = start_iter + niter
        self._resize_chains(niter)
        self._started = True
        
        for iter_i in range(start_iter, end_iter):
            out = self.sample(_x0,
                              self.log_conditional_funcs, 
                              self.ws,
                              args=self.args)
            self.samples[iter_i] = out[0]
            self.n_attempts[iter_i] = out[1]
            _x0 = out[0].copy()
            g_x0 = out[1]

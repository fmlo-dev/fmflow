# coding: utf-8

# public items
__all__ = [
    'Convergence',
]

# standard library
from decimal import Decimal
from math import log10

# dependent packages
import numpy as np
from numpy.linalg import norm
from scipy.stats import gmean


# classes
class Convergence(object):
    def __init__(self, threshold=0.01, n_maxiters=100, n_miniters=2,
            *, reuse_instance=True, raise_exception=False, display_status=True
        ):
        self.params = {
            'threshold': threshold,
            'n_maxiters': n_maxiters,
            'n_miniters': n_miniters,
            'reuse_instance': reuse_instance,
            'raise_exception': raise_exception,
            'display_status': display_status,
        }

        # threshold list
        if hasattr(threshold, '__len__'):
            self._thresholds = list(threshold)
        else:
            self._thresholds = [threshold]

        self._set_threshold()
        self._reset_status()

    @property
    def status(self):
        try:
            var_data = str(self.vars_data[-1])
        except IndexError:
            var_data = None

        try:
            var_vars = str(self.vars_vars[-1])
        except IndexError:
            var_vars = None

        return {'n_iters': self.n_iters, 'var_data': var_data, 'var_vars': var_vars}

    def _variation_data(self):
        var_data = norm(self.data_new-self.data_old) / norm(self.data_old)
        var_data = round(Decimal(var_data), self._ndigits)
        self.vars_data.append(var_data)
        return var_data

    def _variation_vars(self, n_samples=5):
        if len(self.vars_data) < n_samples+1:
            return np.inf

        var_new  = float(self.vars_data[-1])
        vars_old = np.array(self.vars_data[-n_samples-1:-1], dtype=float)
        var_vars = np.abs(var_new/(gmean(1+vars_old)-1) - 1)
        var_vars = round(Decimal(var_vars), 2)
        self.vars_vars.append(var_vars)
        return var_vars

    def _converged(self, raise_exception=False):
        if self.reuse_instance:
            self._set_threshold()
            self._reset_status()

        if raise_exception:
            raise StopIteration('reached maximum iteration')
        else:
            return True

    def _not_converged(self):
        return False

    def _set_threshold(self):
        try:
            threshold = self._thresholds.pop(0)
            self._ndigits = round(-log10(threshold)) + 1
            self._threshold = round(Decimal(threshold), self._ndigits)
        except IndexError:
            pass

    def _reset_status(self):
        self.n_iters = 0
        self.data_new = None
        self.vars_data = []
        self.vars_vars = []

    def __call__(self, data_new):
        self.n_iters += 1
        self.data_new, self.data_old = data_new, self.data_new

        if self.n_iters <= self.n_miniters and self.n_maxiters:
            return self._not_converged()
        elif self.n_iters > self.n_maxiters:
            return self._converged(self.raise_exception)
        elif not np.any(self.data_new-self.data_old):
            return self._converged()
        elif not np.any(self.data_old):
            return self._not_converged()
        elif self._variation_data() <= self._threshold:
            return self._converged()
        elif self._variation_vars() <= 0.05:
            return self._converged()
        else:
            return self._not_converged()

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        if self.display_status:
            return str(self.status)
        else:
            return str(self.params)

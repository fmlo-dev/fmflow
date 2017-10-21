# coding: utf-8

# public items
__all__ = [
    'Convergence',
]

# standard library
from copy import deepcopy
from decimal import Decimal
from math import log10

# dependent packages
import numpy as np
from numpy.linalg import norm


# classes
class Convergence(object):
    def __init__(self, threshold=1e-3, n_maxiters=300, n_miniters=2,
                 *, centering=False, reuseable=True, raise_exception=False):
        """Determine a convergence of data by monitoring their variation in iterations.

        Args:
            threshold (float or list of float): threshold value(s) of data variation.
            n_maxiters (int): Maximum number of iterations.
            n_miniters (int): Minimum number of iterations.
            centering (bool): If True, variation is calculated after centering data.
            reuseable (bool): If True, status of an instance is reset after converged.
            raise_exception (bool): If True, StopIteration exception is raised
                instead of returning True when data are not converged.

        """
        self.params = {
            'threshold': threshold,
            'n_maxiters': n_maxiters,
            'n_miniters': n_miniters,
            'centering': centering,
            'reuseable': reuseable,
            'raise_exception': raise_exception,
        }

        # threshold list
        if hasattr(threshold, '__len__'):
            self._thresholds = list(threshold)
        else:
            self._thresholds = [threshold]

        self._reset_status()
        self._set_threshold()

    def __call__(self, data_new):
        self.n_iters += 1
        self.data_new, self.data_old = deepcopy(data_new), self.data_new

        if self.n_iters <= min(self.n_miniters, self.n_maxiters):
            return self._not_converged()
        elif self.n_iters > self.n_maxiters:
            return self._converged(self.raise_exception)
        elif not np.any(self.data_new-self.data_old):
            return self._converged()
        elif not np.any(self.data_old):
            return self._not_converged()
        elif self._get_variation() <= self._threshold:
            return self._converged()
        else:
            return self._not_converged()

    @property
    def status(self):
        try:
            var = str(self.variations[-1])
        except IndexError:
            var = None

        return {'n_iters': self.n_iters, 'variation': var}

    def _converged(self, raise_exception=False):
        if self.reuseable:
            self._reset_status()
            self._set_threshold()

        if raise_exception:
            message = 'reached maximum iteration'
            raise StopIteration(message)
        else:
            return True

    def _not_converged(self):
        return False

    def _get_variation(self):
        med = np.median(self.data_old) if self.centering else 0
        var = norm(self.data_new-self.data_old) / norm(self.data_old-med)
        var = round(Decimal(var), self._ndigits)
        self.variations.append(var)
        return var

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
        self.variations = []

    def __getattr__(self, name):
        return self.params[name]

    def __str__(self):
        return str(self.status)

    def __repr__(self):
        cname  = self.__class__.__name__
        params = self.params
        return '{0}({1})'.format(cname, params)

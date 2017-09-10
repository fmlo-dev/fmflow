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


# classes
class Convergence(object):
    def __init__(self, threshold=0.01, n_maxiters=100,
            *, reuse_instance=True, raise_exception=False
        ):
        self.params = {
            'threshold': threshold,
            'n_maxiters': n_maxiters,
            'reuse_instance': reuse_instance,
            'raise_exception': raise_exception,
        }
        self._ndigits = round(-log10(threshold)) + 1
        self._threshold = round(Decimal(threshold), self._ndigits)
        self._reset_status()

    @property
    def status(self):
        return {'n_iters': self.n_iters, 'value': self.value}

    def _judge(self, array_new, array_old):
        diff = norm(array_new-array_old) / norm(array_old)
        value = round(Decimal(diff), self._ndigits)
        self.value = str(value)
        return value

    def _converged(self, raise_exception=False):
        if self.reuse_instance:
            self._reset_status()

        if raise_exception:
            raise StopIteration('reached maximum iteration')
        else:
            return True

    def _not_converged(self):
        return False

    def _reset_status(self):
        self.n_iters = 0
        self.value = None
        self.cache = None

    def __call__(self, array_new):
        self.n_iters += 1
        self.cache, array_old = array_new.copy(), self.cache

        if self.n_iters == 1:
            return self._not_converged()
        elif self.n_iters > self.n_maxiters:
            return self._converged(self.raise_exception)
        elif np.all((array_new-array_old)==0):
            return self._converged()
        elif np.all(array_old==0):
            return self._not_converged()
        elif self._judge(array_new, array_old) <= self._threshold:
            return self._converged()
        else:
            return self._not_converged()

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str(self.status)

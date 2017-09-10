# coding: utf-8

# public items
__all__ = [
    'Convergence',
]

# standard library
from decimal import Decimal

# dependent packages
import numpy as np
from numpy.linalg import norm


# classes
class Convergence(object):
    def __init__(self, convergence=0.01, n_maxiters=100, raise_maxiters=False):
        self.params = {
            'convergence': convergence,
            'n_maxiters': n_maxiters,
            'raise_maxiters': raise_maxiters,
        }

        self._ndigits = round(float(-np.log10(convergence)))
        self._reset_status()

    @property
    def status(self):
        return {'value': self.value, 'n_iters': self.n_iters}

    def _compute(self, array_new, array_old):
        diff = norm(array_new-array_old) / norm(array_old)
        return round(Decimal(diff), self._ndigits)

    def _reset_status(self):
        self.n_iters = 0
        self.value = None
        self._cache = None

    def __call__(self, array_new):
        self.n_iters += 1
        self._cache, array_old = array_new.copy(), self._cache

        if self.n_iters > self.n_maxiters:
            if self.raise_maxiters:
                self._reset_status()
                raise StopIteration('reached maximum iteration')
            else:
                self._reset_status()
                return True

        if self.n_iters <= 2:
            return False
        else:
            self.value = self._compute(array_new, array_old)
            if self.value <= self.convergence:
                self._reset_status()
                return True
            else:
                return False

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'Convergence({0})'.format(self.params)

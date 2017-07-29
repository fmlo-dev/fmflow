# coding: utf-8

# public items
__all__ = [
    'Convergence',
]

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

        self.n_iters = 0
        self.value = None
        self._cache = None

    @property
    def status(self):
        return {'value': self.value, 'n_iters': self.n_iters}

    @staticmethod
    def _compute(array_new, array_old):
        return np.abs(norm(array_new-array_old) / norm(array_old))

    def __call__(self, array_new):
        self.n_iters += 1
        self._cache, array_old = array_new.copy(), self._cache

        if self.n_iters > self.n_maxiters:
            if self.raise_maxiters:
                raise StopIteration('reached maximum iteration')
            else:
                return True

        if self.n_iters <= 2:
            return False

        self.value = self._compute(array_new, array_old)
        return self.value < self.convergence

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'Convergence({0})'.format(self.params)

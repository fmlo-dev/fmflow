# coding: utf-8

# imported items
__all__ = [
    'Convergence',
]

# dependent packages
import numpy as np
from numpy.linalg import norm


# classes
class Convergence(object):
    def __init__(self, convergence=0.01, n_maxiters=100):
        self.params = {
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }

        self.array = None
        self.value = None
        self.n_iters = 0

    @property
    def status(self):
        return {'value': self.value, 'n_iters': self.n_iters}

    @staticmethod
    def _compute(array_new, array_old):
        return np.abs(norm(array_new-array_old) / norm(array_old))

    def __call__(self, array_new):
        self.n_iters += 1
        self.array, array_old = array_new.copy(), self.array

        if self.n_iters > self.n_maxiters:
            raise StopIteration('reached maximum iteration')

        if self.n_iters == 1:
            return False

        self.value = self._compute(array_new, array_old)
        return self.value < self.convergence

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'Convergence(convergence={0}, n_maxiters={1})',
            self.convergence, self.n_maxiters
        )

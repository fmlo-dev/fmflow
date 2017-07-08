# coding: utf-8

# imported items
__all__ = [
    'Convergence',
]

# dependent packages
import numpy as np


# classes
class Convergence(object):
    def __init__(self, convergence=0.01, n_maxiters=100):
        self.params = {
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }

        self.array = None
        self.variation = None
        self.n_iters = 0

    def __call__(self, array_new):
        self.n_iters += 1
        self.array, array_old = array_new.copy(), self.array

        if self.n_iters > self.n_maxiters:
            raise StopIteration('reached maximum iteration')

        if self.n_iters == 1:
            return False

        self.variation = self._compute(array_new, array_old)
        return np.abs(self.variation) < self.convergence

    @staticmethod
    def _compute(array_new, array_old):
        return np.linalg.norm(array_new-array_old) / np.linalg.norm(array_old)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'Convergence(convergence={0}, n_maxiters={1})',
            self.convergence, self.n_maxiters
        )

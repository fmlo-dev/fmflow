# coding: utf-8

# imported items
__all__ = [
    'Convergence',
]

# dependent packages
import numpy as np


# classes
class Convergence(object):
    def __init__(self, threshold=0.01, n_maxiters=100, mode='norm'):
        self.params = {
            'threshold': threshold,
            'n_maxiters': n_maxiters,
            'mode': mode,
        }

        self.n_iters = 0
        self.value = None
        self.array = None

    def __call__(self, array_new):
        self.n_iters += 1
        self.array, array_old = array_new.copy(), self.array

        if self.n_iters > self.n_maxiters:
            raise StopIteration('reached maximum iteration')

        if self.n_iters == 1:
            return False

        self.value = getattr(self, '_'+self.mode)(array_new, array_old)
        return np.abs(self.value) < self.threshold

    @staticmethod
    def _norm(array_new, array_old):
        return np.linalg.norm(array_new-array_old) / np.linalg.norm(array_old)

    @staticmethod
    def _mean(array_new, array_old):
        return np.mean((array_new-array_old) / array_old)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'Convergence(threshold={0}, n_maxiters={1}, mode={2})',
            self.threshold, self.n_maxiters, self.mode,
        )

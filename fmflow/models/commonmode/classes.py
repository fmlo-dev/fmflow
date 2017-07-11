# coding: utf-8

# imported items
__all__ = [
    'EMPCA',
]

# dependent packages
import fmflow as fm
import numpy as np


# classes
class EMPCA(object):
    def __init__(self, n_components=20, convergence=0.01, n_maxiters=100, random_seed=None):
        self.params = {
            'n_components': n_components,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
            'random_seed': random_seed,
        }

    def fit_transform(self, X, W=None):
        # check array and weights
        X = np.asarray(X)

        if W is not None:
            W = np.asarray(W)
        else:
            W = np.ones_like(X)

        if not X.shape == W.shape:
            raise ValueError('X and W must have same shapes')

        # shapes of matrices
        (N, D), K = X.shape, self.n_components
        self.params.update({'N': N, 'D': D, 'K': K})

        # initial random eigen vectors
        np.random.seed(self.random_seed)
        A = np.random.randn(self.K, self.D)
        P = fm.utils.orthonormalize(A)

        # EM algorithm
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters)
        try:
            while not cv(P):
                C = self._update_coefficients(X, W, P)
                P = self._update_eigenvectors(X, W, C)
        except StopIteration:
            fm.logger.warning('reached maximum iteration')

        # finally
        self.components_ = P
        return C

    def _update_coefficients(self, X, W, P):
        C = np.empty([self.N, self.K])
        for n in range(self.N):
            Pn = P @ (P * W[n]).T
            xn = P @ (X[n] * W[n])
            C[n] = np.linalg.solve(Pn, xn)

        return C

    def _update_eigenvectors(self, X, W, C):
        X = X.copy()
        P = np.empty([self.K, self.D])
        for k in range(self.K):
            P[k] = (C[:,k] @ (X*W)) / (C[:,k]**2 @ W)
            X -= np.outer(C[:,k], P[k])

        return fm.utils.orthonormalize(P)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'EMPCA(n_components={0}, n_maxiters={1}, random_seed={2})',
            self.n_components, self.n_maxiters, self.random_seed,
        )

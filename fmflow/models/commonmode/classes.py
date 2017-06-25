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
    def __init__(self, n_components=10, n_maxiters=25, random_seed=None):
        self.info = {
            'n_components': n_components,
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
            raise fm.utils.FMFlowError('error!')

        # shapes of matrices
        (N, D), K = X.shape, self.n_components
        self.info.update({'N': N, 'D': D, 'K': K})

        # initial random eigen vectors
        np.random.seed(self.random_seed)
        A = np.random.randn(self.K, self.D)
        P = self._orthonormalize(A)

        # EM algorithm
        for i in range(self.n_maxiters):
            print(i+1, end=' ')
            C = self._update_coefficients(X, W, P)
            P = self._update_eigenvectors(X, W, C)

        # finally
        self.components_ = P
        return C

    def _update_coefficients(self, X, W, P):
        C = np.zeros([self.N, self.K])
        for n in range(self.N):
            Pn = P @ (P * W[n]).T
            xn = P @ (X[n] * W[n])
            C[n] = np.linalg.solve(Pn, xn)

        return C

    def _update_eigenvectors(self, X, W, C):
        X = X.copy()
        P = np.zeros([self.K, self.D])
        for k in range(self.K):
            ck = C[:,k]
            P[k] = (ck @ (X*W)) / (ck**2 @ W)
            X -= np.outer(ck, P[k])

        return self._orthonormalize(P)

    def __getattr__(self, name):
        return self.info[name]

    def __repr__(self):
        string = str.format(
            'EMPCA(n_components={0}, n_maxiters={1}, random_seed={2})',
            self.n_components, self.n_maxiters, self.random_seed
        )

        return string

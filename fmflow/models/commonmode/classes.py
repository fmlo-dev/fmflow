# coding: utf-8

# imported items
__all__ = [
    'EMPCA',
]

# dependent packages
import fmflow as fm
import numpy as np
from numba import jit


# classes
class EMPCA(object):
    def __init__(
            self, n_components=20, convergence=0.01, n_maxiters=100,
            random_seed=None, *, logger=None
        ):
        self.params = {
            'n_components': n_components,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
            'random_seed': random_seed,
        }

        self.logger = logger or fm.logger

    def fit_transform(self, X, W=None):
        # check array and weights
        X = np.asarray(X)

        if W is not None:
            W = np.asarray(W)
        else:
            W = np.ones_like(X)

        if not X.shape == W.shape:
            raise ValueError('X and W must have same shapes')

        XW = X * W

        # shapes of matrices
        (N, D), K = X.shape, self.n_components

        # initial coefficients and eigenvectors
        C = np.empty([N, K])
        P = self._random_orthogonal([K, D])

        # EM algorithm
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters, True)
        try:
            while not cv(C @ P):
                self.logger.debug(cv.status)
                C = self._update_coefficients(XW, W, P, C)
                P = self._update_eigenvectors(XW.copy(), W, P, C)
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        # finally
        self.components_ = P
        return C

    def _random_orthogonal(self, shape):
        np.random.seed(self.random_seed)
        A = np.random.randn(*shape)
        for i in range(A.shape[0]):
            for j in range(i):
                A[i] -= (A[i] @ A[j]) * A[j]

            A[i] /= np.linalg.norm(A[i])

        return A

    @staticmethod
    @jit(nopython=True, cache=True, nogil=True)
    def _update_coefficients(XW, W, P, C):
        N, D = XW.shape
        for n in range(N):
            Pn = P @ (P * W[n]).T
            xn = P @ XW[n]
            C[n] = np.linalg.solve(Pn, xn)

        return C

    @staticmethod
    @jit(nopython=True, cache=True, nogil=True)
    def _update_eigenvectors(XW, W, P, C):
        (N, D), K = XW.shape, C.shape[1]
        for k in range(K):
            ck = C[:, k]
            pk = (ck @ XW) / (ck**2 @ W)

            for n in range(N):
                for d in range(D):
                    XW[n, d] -= W[n, d] * ck[n] * pk[d]

            P[k] = pk

        for k in range(K):
            for l in range(k):
                P[k] -= (P[k] @ P[l]) * P[l]

            P[k] /= np.linalg.norm(P[k])

        return P

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'EMPCA(n_components={0}, n_maxiters={1}, random_seed={2})',
            self.n_components, self.n_maxiters, self.random_seed
        )

# coding: utf-8

# public items
__all__ = [
    'EMPCA',
]

# dependent packages
import fmflow as fm
import numpy as np
from numba import jit
from sklearn import decomposition
from scipy.signal import savgol_filter


# classes
class EMPCA(object):
    def __init__(
            self, n_components=20, initialize='random', random_seed=None,
            ch_smooth=None, convergence=1e-3, n_maxiters=100, *, logger=None):
        self.params = {
            'n_components': n_components,
            'initialize': initialize,
            'random_seed': random_seed,
            'ch_smooth': ch_smooth,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }

        self.logger = logger or fm.logger

    def fit_transform(self, X, W=None):
        # check array and weights
        if W is None:
            W = np.ones_like(X)

        if not X.shape == W.shape:
            raise ValueError('X and W must have same shapes')

        # shapes of matrices (for convergence)
        N, D, K = *X.shape, self.n_components

        # initial arrays
        C = np.zeros([N, K])
        if self.initialize == 'random':
            P = self._orthogonal_from_random(K, D)
        elif self.initialize == 'svd':
            P = self._orthogonal_from_svd(X, K)
        else:
            raise ValueError(self.initialize)

        # convergence
        cv = fm.utils.Convergence(
            self.convergence, self.n_maxiters, raise_exception=True
        )

        # EM algorithm
        try:
            _WX = W * X
            while not cv(C @ P):
                WX = _WX.copy()
                self.logger.debug(cv.status)
                C = self._update_coefficients(C, P, WX, W)
                P = self._update_eigenvectors(C, P, WX, W)
                if (self.ch_smooth is not None) and self.ch_smooth:
                    P = self._smooth_eigenvectors(P, self.ch_smooth)
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        # finally
        self.components_ = P
        return C

    def _orthogonal_from_random(self, *shape):
        np.random.seed(self.random_seed)
        A = np.random.randn(*shape)
        for i in range(A.shape[0]):
            for j in range(i):
                A[i] -= (A[i] @ A[j]) * A[j]

            A[i] /= np.linalg.norm(A[i])

        return A

    def _orthogonal_from_svd(self, X, K):
        svd = decomposition.TruncatedSVD(K)
        svd.fit(X)
        return svd.components_

    def _smooth_eigenvectors(self, P, ch_smooth):
        return savgol_filter(P, ch_smooth, polyorder=3, axis=1)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _update_coefficients(C, P, WX, W):
        N, D = WX.shape

        for n in range(N):
            # equiv to the equation 16
            Pn = P @ (P * W[n]).T
            xn = P @ WX[n]
            C[n] = np.linalg.solve(Pn, xn)

        return C

    @staticmethod
    @jit(nopython=True, cache=True)
    def _update_eigenvectors(C, P, WX, W):
        N, D = WX.shape
        K = P.shape[0]

        for k in range(K):
            # equiv to the equation 21
            Ck = C[:, k]
            P[k] = (Ck @ WX) / (Ck**2 @ W)

            # equiv to the equation 22
            # but subtracting from WX
            for n in range(N):
                for d in range(D):
                    WX[n, d] -= W[n, d] * P[k, d] * Ck[n]

            # renormalization
            for m in range(k):
                P[k] -= (P[k] @ P[m]) * P[m]

            P[k] /= np.linalg.norm(P[k])

        return P

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'EMPCA({0})'.format(self.params)

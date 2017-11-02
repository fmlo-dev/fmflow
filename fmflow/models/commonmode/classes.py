# coding: utf-8

# public items
__all__ = [
    'PCA',
    'EMPCA',
]

# standard library
from copy import deepcopy

# dependent packages
import fmflow as fm
import numpy as np
from .. import BaseModel
from numba import jit
from sklearn import decomposition
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


# classes
class PCA(BaseModel):
    def __init__(self, n_components=50, optimize_n=True, *, logger=None):
        params = {
            'n_components': n_components,
            'optimize_n': optimize_n,
        }
        super().__init__(params, logger)

    def fit_transform(self, X):
        K = deepcopy(self.n_components)
        model = decomposition.TruncatedSVD(K)

        X = np.asarray(X)
        C = model.fit_transform(X)
        P = model.components_

        if self.optimize_n:
            K_opt = self._optimize_K(C, 7)
            if K_opt < self.n_components:
                C, P = C[:,:K_opt], P[:K_opt]
                self.logger.info('optimized n_components: {}'.format(K_opt))
            else:
                self.logger.warning('optimized n_components exceeds the original')
                self.logger.warning('the original is used for reconstruction')

        self.components_ = P
        return C

    @staticmethod
    def _optimize_K(C, level=5):
        npc = np.arange(C.shape[1])
        lmd = np.log10(C.var(0)) # log eigen values

        def func(x, a, b, c):
            return a * 2**(-b*x) + c

        popt, pcov = curve_fit(func, npc, lmd)
        return int(level/popt[1]) + 1


class EMPCA(BaseModel):
    def __init__(self, n_components=50, ch_smooth=None, optimize_n=True,
                 initialize='random', random_seed=None, *, convergence=1e-3,
                 n_maxiters=300, logger=None):
        params = {
            'n_components': n_components,
            'ch_smooth': ch_smooth,
            'optimize_n': optimize_n,
            'initialize': initialize,
            'random_seed': random_seed,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }
        super().__init__(params, logger)

    def fit_transform(self, X, W=None):
        X = np.asarray(X)

        # check array and weights
        if W is None:
            W = np.ones_like(X)
        else:
            W = np.asarray(W)

        if not X.shape == W.shape:
            raise ValueError('X and W must have same shapes')

        # shapes of matrices (for convergence)
        N, D, K = *X.shape, deepcopy(self.n_components)

        if self.optimize_n:
            model = decomposition.TruncatedSVD(K)
            C = model.fit_transform(X)
            K_opt = 2 * self._optimize_K(C, 7)
            K = K_opt if K_opt < K else K

        # initial arrays
        np.random.seed(self.random_seed)

        C = np.zeros([N, K])
        if self.initialize == 'random':
            P = self._orthogonal_from_random(K, D)
        elif self.initialize == 'svd':
            P = self._orthogonal_from_svd(X, K)
        else:
            raise ValueError(self.initialize)

        # convergence
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters,
                                  centering=True, raise_exception=True)

        # EM algorithm
        try:
            _WX = W * X
            while not cv(C @ P):
                WX = _WX.copy()
                self.logger.debug(cv)
                C = self._update_coefficients(C, P, WX, W)
                P = self._update_eigenvectors(C, P, WX, W)
                if (self.ch_smooth is not None) and (self.ch_smooth > 0):
                    P = self._smooth_eigenvectors(P, self.ch_smooth)
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        # finally
        if self.optimize_n:
            K_opt = self._optimize_K(C, 7)
            if K_opt < self.n_components:
                self.logger.info('optimized n_components: {}'.format(K_opt))
                C, P = C[:,:K_opt], P[:K_opt]
            else:
                self.logger.warning('optimized n_components exceeds the original')
                self.logger.warning('the original is used for reconstruction')

        self.components_ = P
        return C

    @staticmethod
    def _orthogonal_from_random(*shape):
        A = np.random.randn(*shape)
        for i in range(A.shape[0]):
            for j in range(i):
                A[i] -= (A[i] @ A[j]) * A[j]

            A[i] /= np.linalg.norm(A[i])

        return A

    @staticmethod
    def _orthogonal_from_svd(X, K):
        svd = decomposition.TruncatedSVD(K)
        svd.fit(X)
        return svd.components_

    @staticmethod
    def _smooth_eigenvectors(P, ch_smooth):
        return savgol_filter(P, ch_smooth, polyorder=3, axis=1)

    @staticmethod
    def _optimize_K(C, level=5):
        npc = np.arange(C.shape[1])
        lmd = np.log10(C.var(0)) # log eigen values

        def func(x, a, b, c):
            return a * 2**(-b*x) + c

        popt, pcov = curve_fit(func, npc, lmd)
        return int(level/popt[1]) + 1

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

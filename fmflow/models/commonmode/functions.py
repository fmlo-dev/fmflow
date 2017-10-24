# coding: utf-8

# public items
__all__ = [
    'pca',
    'empca',
    'decomposition',
]

# standard library
from collections import defaultdict
from copy import deepcopy
from logging import getLogger

# dependent packages
import numpy as np
import fmflow as fm
from sklearn import decomposition as _decomposition

# module constants
SKPARAMS = defaultdict(dict)
SKPARAMS['KernelPCA'] = {'fit_inverse_transform': True}


# functions
@fm.xarrayfunc
@fm.chunk('array')
def pca(array, n_components=50, optimize_n=True, centering=True):
    """Reconstruct an array from decomposed one with PCA.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        n_components (int): A number of components to keep.
        optimize_n (bool): If True, `n_components` used for reconstruction is
            optimized by an exponential fitting of eigen values of PCA.
        centering (bool): If True, mean vector along time axis is subtracted from
            `array` before computing PCA and then added to the reconstructed one.
        kwargs (dict): Parameters for the timechunk calculation such as
            `timechunk`, `n_processes`. See `fmflow.timechunk` for more detail.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    """
    logger = getLogger('fmflow.models.pca')
    model = fm.models.PCA(n_components, optimize_n, logger=logger)

    mean = np.mean(array, 0) if centering else 0
    transformed = model.fit_transform(array-mean)
    return transformed @ model.components_ + mean


@fm.xarrayfunc
@fm.chunk('array', 'weights')
def empca(array, weights=None, n_components=50, ch_smooth=None, optimize_n=True,
          initialize='random', random_seed=None, centering=True,
          convergence=1e-3, n_maxiters=300, **kwargs):
    """Reconstruct an array from decomposed one with EMPCA.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        weights (xarray.DataArray): A weight array. The shape must be same as `array`.
        n_components (int): A number of components to keep.
        ch_smooth (int): A length of the filter window for smoothing eigenvectors.
            It must be a positive odd integer.
        optimize_n (bool): If True, `n_components` used for reconstruction is
            optimized by an exponential fitting of eigen values of EMPCA.
        initialize (string): A method of initializing eigenvectors.
            Options are `random` (random orthogonal matrix) and `svd`
            (orthogonal matrix from singular value decomposition).
        random_seed (int): random seed values used for the initial state.
        centering (bool): If True, mean vector along time axis is subtracted from
            `array` before computing EMPCA and then added to the reconstructed one.
        convergence (float): A convergence threshold.
            See `fmflow.utils.Convergence` for more detail.
        n_maxiters (int): A number of maximum iterations of the EM step.
        kwargs (dict): Parameters for the timechunk calculation such as
            `timechunk`, `n_processes`. See `fmflow.timechunk` for more detail.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    """
    logger = getLogger('fmflow.models.empca')
    model = fm.models.EMPCA(n_components, ch_smooth, optimize_n, initialize, random_seed,
                            convergence=convergence, n_maxiters=n_maxiters, logger=logger)

    mean = np.mean(array, 0) if centering else 0
    transformed = model.fit_transform(array-mean, weights)
    return transformed @ model.components_ + mean


@fm.xarrayfunc
@fm.chunk('array')
def decomposition(array, n_components=None, decomposer='TruncatedSVD',
                  centering=True, **kwargs):
    """Reconstruct an array from decomposed one with a scikit-learn decomposer.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        n_components (int): A number of components to keep.
        decomposer (str): A name of algorithm provided by sklearn.decomposition.
        centering (bool): If True, mean vector along time axis is subtracted from
            `array` before decomposition and then added to the reconstructed one.
        kwargs (dict): Parameters for the spacified algorithm such as
            `n_components` and for the timechunk calculation such as
            `timechunk`, `n_processes`. See `fmflow.timechunk` for more detail.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    Example:
        To reconstruct an array from top two principal components:

        >>> result = fm.model.reducedim(array, 'PCA', n_components=2)

    """
    logger = getLogger('fmflow.models.decomposition')

    AlgorithmClass = getattr(_decomposition, decomposer)
    params = deepcopy(SKPARAMS[decomposer])
    params.update(kwargs)
    model = AlgorithmClass(n_components, **params)

    mean = np.mean(array, 0) if centering else 0
    transformed = model.fit_transform(array-mean)

    if hasattr(model, 'components_'):
        return transformed @ model.components_ + mean
    elif hasattr(model, 'inverse_transform'):
        return model.inverse_transform(transformed) + mean
    else:
        raise ValueError('cannot reconstruct with the spacified algorithm')

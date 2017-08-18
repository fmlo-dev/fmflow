# coding: utf-8

# public items
__all__ = [
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
@fm.numpyfunc
@fm.timechunk
def empca(
        array, weights, n_components=20, initialize='random', random_seed=None,
        smooth=None, centering=True, convergence=1e-3, n_maxiters=100,  **kwargs
    ):
    """Reconstruct an array from decomposed one with EMPCA.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        weights (xarray.DataArray): A weight array. It must have the same shape
            as `array`. Just spacify `None` in the case of no weights.
        n_components (int): A number of components to keep.
        initialize (string): A method of initializing eigenvectors.
            Options are `random` (random orthogonal matrix) and `svd`
            (orthogonal matrix from singular value decomposition).
        random_seed (int): random seed values used for the initial state.
        smooth (int): A length of the filter window for smoothing eigenvectors.
            It must be a positive odd integer.
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
    logger.debug('n_components: {0}'.format(n_components))
    logger.debug('initialize: {0}'.format(initialize))
    logger.debug('random_seed: {0}'.format(random_seed))
    logger.debug('smooth: {0}'.format(smooth))
    logger.debug('centering: {0}'.format(centering))
    logger.debug('convergence: {0}'.format(convergence))
    logger.debug('n_maxiters: {0}'.format(n_maxiters))

    model = fm.models.EMPCA(
        n_components, initialize, random_seed, smooth,
        convergence, n_maxiters, logger=logger
    )

    mean = np.mean(array, 0) if centering else 0
    transformed = model.fit_transform(array-mean, weights)
    return transformed @ model.components_ + mean


@fm.numpyfunc
@fm.timechunk
def decomposition(array, decomposer='TruncatedSVD', n_components=None, centering=True, **kwargs):
    """Reconstruct an array from decomposed one with a scikit-learn decomposer.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        decomposer (str): A name of algorithm provided by sklearn.decomposition.
        n_components (int): A number of components to keep.
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
    logger.debug('decomposer: {0}'.format(decomposer))
    logger.debug('n_components: {0}'.format(n_components))
    logger.debug('centering: {0}'.format(centering))

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

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
        array, weights, n_components=20, centering=True,
        convergence=0.001, n_maxiters=100, random_seed=None, **kwargs
    ):
    """Reconstruct an array from decomposed one with EMPCA.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        weights (xarray.DataArray): A weight array. It must have the same shape
            as `array`. Just spacify `None` in the case of no weights.
        n_components (int): A number of components to keep.
        centering (bool): If True, mean vector along time axis is subtracted from
            `array` before computing EMPCA and then added to the reconstructed one.
        convergence (float): A convergence threshold.
            See `fmflow.utils.Convergence` for more detail.
        n_maxiters (int): A number of maximum iterations of the EM step.
        random_seed (int): random seed values used for the initial state.
        kwargs (dict): Parameters for the timechunk calculation such as
            `timechunk`, `n_processes`. See `fmflow.timechunk` for more detail.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    """
    logger = getLogger('fmflow.models.empca')
    logger.info('n_components: {0}'.format(n_components))
    logger.info('centering: {0}'.format(centering))
    logger.debug('convergence: {0}'.format(convergence))
    logger.debug('n_maxiters: {0}'.format(n_maxiters))
    logger.debug('random_seed: {0}'.format(random_seed))

    model = fm.models.EMPCA(
        n_components, convergence, n_maxiters, random_seed, logger=logger
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
    logger.info('decomposer: {0}'.format(decomposer))
    logger.info('n_components: {0}'.format(n_components))
    logger.info('centering: {0}'.format(centering))

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

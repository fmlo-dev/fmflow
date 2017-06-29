# coding: utf-8

# imported items
__all__ = [
    'empca',
    'skdecomposition',
]

# standard library
from collections import defaultdict
from copy import deepcopy

# dependent packages
import numpy as np
import fmflow as fm
from sklearn import decomposition

# constants
SKPARAMS = defaultdict(dict)
SKPARAMS['KernelPCA'] = {'fit_inverse_transform': True}


# functions
@fm.timechunk
def empca(array, weights, n_components=20, n_maxiters=10, random_seed=None, **kwargs):
    """Reconstruct an array from decomposed one with EMPCA.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        weights (xarray.DataArray): A weight array. It must have the same shape
            as `array`. Just spacify `None` in the case of no weights.
        n_components (int): A number of components to keep.
        n_maxiters (int): A number of maximum iterations of the EM step.
        random_seed (int): random seed values used for the initial state.
        kwargs (dict): Parameters for the timechunk calculation such as
            `timechunk`, `n_processes`. See `fmflow.timechunk` for more detail.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    """
    if weights is None:
        weights = np.ones_like(array)

    model = fm.models.EMPCA(n_components, n_maxiters, random_seed)
    transformed = model.fit_transform(array-array.mean(0), weights)
    return transformed @ model.components_ + array.mean(0)


@fm.timechunk
def skdecomposition(array, decomposer='TruncatedSVD', n_components=None, **kwargs):
    """Reconstruct an array from decomposed one with a scikit-learn decomposer.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        decomposer (str): A name of algorithm provided by sklearn.decomposition.
        n_components (int): A number of components to keep.
        kwargs (dict): Parameters for the spacified algorithm such as
            `n_components` and for the timechunk calculation such as
            `timechunk`, `n_processes`. See `fmflow.timechunk` for more detail.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    Example:
        To reconstruct an array from top two principal components:

        >>> result = fm.model.reducedim(array, 'PCA', n_components=2)

    """
    AlgorithmClass = getattr(decomposition, decomposer)
    params = deepcopy(SKPARAMS[decomposer])
    params.update(kwargs)

    model = AlgorithmClass(n_components, **params)
    transformed = model.fit_transform(array-array.mean(0))

    if hasattr(model, 'components_'):
        return transformed @ model.components_ + array.mean(0)
    elif hasattr(model, 'inverse_transform'):
        return model.inverse_transform(transformed) + array.mean(0)
    else:
        raise fm.utils.FMFlowError('cannot reconstruct with the spacified algorithm')

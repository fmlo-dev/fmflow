# coding: utf-8

# imported items
__all__ = [
    'decompose',
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
def decompose(array, decomposer='EMPCA', **kwargs):
    """Decompose an array with given algorithm.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        decomposer (str): A name of decomposition class provided by
            fmflow.models or sklearn.decomposition. Default is 'EMPCA'.
        kwargs (dict): Parameters for the spacified algorithm such as `n_components`.

    Returns:
        bases (numpy.ndarray): Basis vectors.
        coords (numpy.ndarray): Coordinate vectors.

    """
    try:
        AlgorithmClass = getattr(fm.models, decomposer)
    except AttributeError:
        AlgorithmClass = getattr(decomposition, decomposer)

    params = deepcopy(PARAMS[decomposer])
    params.update(kwargs)

    model = AlgorithmClass(**params)
    fit = model.fit_transform(array)

    if hasattr(model, 'components_'):
        return fit, model.components_
    else:
        raise fm.utils.FMFlowError('cannot decompose with the spacified algorithm')


@fm.timechunk
def skdecomposition(array, decomposer='TruncatedSVD', **kwargs):
    """Reconstruct an array from decomposed one with a scikit-learn decomposer.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        decomposer (str): A name of algorithm provided by sklearn.decomposition.
        kwargs (dict): Parameters for the spacified algorithm such as `n_components`.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    Example:
        To reconstruct an array from top two principal components:

        >>> result = fm.model.reducedim(array, 'PCA', n_components=2)

    """
    AlgorithmClass = getattr(decomposition, decomposer)
    params = deepcopy(SKPARAMS[decomposer])
    params.update(kwargs)

    model = AlgorithmClass(**params)
    transformed = model.fit_transform(array)

    if hasattr(model, 'components_'):
        return transformed @ model.components_
    elif hasattr(model, 'inverse_transform'):
        return model.inverse_transform(fit)
    else:
        raise fm.utils.FMFlowError('cannot reconstruct with the spacified algorithm')

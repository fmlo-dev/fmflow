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
def empca(array, weights, **kwargs):
    """Reconstruct an array from decomposed one with EMPCA.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        weights (xarray.DataArray): A weight array. It must have the same shape
            as `array`. Just spacify `None` if executing this function without weights.
        kwargs (dict): Parameters for the spacified algorithm such as `n_components`.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    """
    model = fm.models.EMPCA(**kwargs)
    transformed = model.fit_transform(array, weights)
    return transformed @ model.components_


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

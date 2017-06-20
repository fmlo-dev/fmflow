# coding: utf-8

# imported items
__all__ = [
    'reconstruct',
]

# standard library
from collections import defaultdict
from copy import deepcopy

# dependent packages
import numpy as np
import fmflow as fm
from sklearn import decomposition

# constants
PARAMS = defaultdict(dict)
PARAMS['KernelPCA'] = {'fit_inverse_transform': True}


# functions
@fm.timechunk
def reconstruct(array, decomposer='TruncatedSVD', **kwargs):
    """Reconstruct an array from decomposed one with given algorithm.

    Args:
        array (xarray.DataArray): An input array to be decomposed.
        decomposer (str): A name of decomposition class
            which sklearn.decomposition provides.
        kwargs (dict): Parameters for the spacified algorithm such as `n_components`.

    Returns:
        array (xarray.DataArray): An output reconstructed array.

    Example:
        To reconstruct an array from top two principal components:

        >>> result = fm.model.reducedim(array, 'PCA', n_components=2)

    """
    AlgorithmClass = getattr(decomposition, decomposer)
    params = deepcopy(PARAMS[decomposer])
    params.update(kwargs)

    model = AlgorithmClass(**params)
    fit = model.fit_transform(array)

    if hasattr(model, 'components_'):
        return fit @ model.components_
    elif hasattr(model, 'inverse_transform'):
        return model.inverse_transform(fit)
    else:
        raise fm.utils.FMFlowError('cannot reconstruct with the spacified algorithm')

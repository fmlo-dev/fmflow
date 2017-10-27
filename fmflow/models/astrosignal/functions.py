# coding: utf-8

# public items
__all__ = [
    'astrolines',
]

# standard library
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np


# functions
@fm.chunk('array', 'weights')
def astrolines(array, weights=None, reverse=False, freqlim=None,
               snr_threshold=10, cutoff_width=3, despiking=True, fit_function=None,
               subtraction_gain=0.5, convergence=1e-3, n_maxiters=10000):
    """Model astrolines by cutoff-and-fitting method.

    Args:
        array (xarray.DataArray):
        weights (xarray.DataArray, optional):
        reverse (bool, optional):
        freqlim (list of float, optional):
        snr_threshold (float, optional):
        cutoff_width (float, optional):
        despiking (bool, optional):
        fit_function (str, optional):
        subtraction_gain (float, optional):
        convergence (float, optional):
        n_maxiters (int, optional):

    Returns:
        model (xarray.DataArray):

    """
    logger = getLogger('fmflow.models.atmoslines')
    model = fm.models.AstroLines(snr_threshold, cutoff_width, despiking, fit_function,
                                 subtraction_gain, convergence=convergence,
                                 n_maxiters=n_maxiters, logger=logger)

    spec = fm.tospectrum(array, weights, reverse)
    spec[:] = model.fit(1e-9*spec.freq, spec, spec.noise, freqlim)
    return fm.fromspectrum(spec, array)

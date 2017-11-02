# coding: utf-8

# public items
__all__ = [
    'astrolines',
]

# standard library
from itertools import product
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np


# functions
@fm.chunk('array', 'weights')
def astrolines(array, weights=None, reverse=False, mode='spectrum', freqlim=None,
               snr_threshold=10, cutoff_width=3, despiking=True, fit_function=None,
               subtraction_gain=0.5, convergence=1e-3, n_maxiters=10000, **kwargs):
    """Model astrolines by cutoff-and-fitting method.

    Args:
        array (xarray.DataArray):
        weights (xarray.DataArray, optional):
        reverse (bool, optional):
        mode (str, optional):
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

    if mode.lower() == 'spectrum':
        spec = fm.tospectrum(array, weights, reverse, **kwargs)
        freq = 1e-9 * spec.freq
        spec[:] = model.fit(1e-9*spec.freq, spec, spec.noise, freqlim)
        return fm.fromspectrum(spec, array)
    elif mode.lower() == 'cube':
        cube  = fm.tocube(array, weights, reverse, **kwargs)
        freq  = 1e-9 * cube.freq
        noise = cube.noise
        for x, y in product(range(len(cube.x)), range(len(cube.y))):
            cube[x, y, :] = model.fit(freq, cube[x, y], noise[x, y], freqlim)

        return fm.fromcube(cube, array)
    else:
        raise ValueError(mode)

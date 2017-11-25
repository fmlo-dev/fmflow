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
def astrolines(array, weights=None, reverse=False, mode='spectrum',
               freqlim=None, function='cutoff', despiking=True, snr_threshold=10,
               deconvolution_width=3, subtraction_gain=0.5, convergence=1e-3,
               n_maxiters=10000, **kwargs):
    """Model astrolines by cutoff-and-fitting method.

    Args:
        array (xarray.DataArray):
        weights (xarray.DataArray, optional):
        reverse (bool, optional):
        mode (str, optional):
        freqlim (list of float, optional):
        function (str, optional):
        despiking (bool, optional):
        snr_threshold (float, optional):
        deconvolution_width (float, optional):
        subtraction_gain (float, optional):
        convergence (float, optional):
        n_maxiters (int, optional):

    Returns:
        model (xarray.DataArray):

    """
    logger = getLogger('fmflow.models.atmoslines')
    model = fm.models.AstroLines(function, despiking, snr_threshold,
                                 deconvolution_width, subtraction_gain,
                                 convergence=convergence, n_maxiters=n_maxiters,
                                 logger=logger)

    if mode.lower() == 'spectrum':
        spec = fm.tospectrum(array, weights, reverse, **kwargs)
        freq = 1e-9 * spec.freq
        spec[:] = model.fit(1e-9*spec.freq, spec, spec.noise, freqlim)
        return fm.fromspectrum(spec, array)
    elif mode.lower() == 'cube':
        cube  = fm.tocube(array, weights, reverse, **kwargs)
        cube.values[np.isnan(cube.values)] = 0
        freq  = 1e-9 * cube.freq
        noise = cube.noise
        for y, x in product(range(len(cube.y)), range(len(cube.x))):
            cube[:, y, x] = model.fit(freq, cube[:, y, x], noise[:, y, x], freqlim)

        return fm.fromcube(cube, array)
    else:
        raise ValueError(mode)

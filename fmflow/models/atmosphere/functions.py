# coding: utf-8

# public items
__all__ = [
    'atmoslines',
    'computeam',
]

# standard library
from itertools import product
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np


# functions
@fm.chunk('array', 'weights')
def atmoslines(array, weights=None, reverse=False, mode='spectrum',
               freqlim=None, snr_threshold=10, ch_tolerance=5, **kwargs):
    logger = getLogger('fmflow.models.atmoslines')
    model = fm.models.AtmosLines(snr_threshold, ch_tolerance, logger=logger)

    if mode.lower() == 'spectrum':
        spec  = fm.tospectrum(array, weights, reverse, **kwargs)
        freq  = 1e-9 * spec.freq
        noise = spec.noise
        vrad  = array.vrad.values.mean()
        spec[:] = model.fit(freq, spec, noise, vrad, freqlim)
        return fm.fromspectrum(spec, array)
    elif mode.lower() == 'cube':
        cube  = fm.tocube(array, weights, reverse, **kwargs)
        spec  = cube.mean(('y', 'x'))
        freq  = 1e-9 * cube.freq

        with fm.utils.ignore_numpy_errors():
            noise  = (cube.noise**2).sum(('y', 'x'))**0.5
            noise /= (~np.isnan(cube)).sum(('y', 'x'))

        vrad  = array.vrad.values.mean()
        cube[:] = model.fit(freq, spec, noise, vrad, freqlim)[:, None, None]
        return fm.fromcube(cube, array)
    else:
        raise ValueError(mode)


def computeam(array, reverse=False):
    logger = getLogger('fmflow.models.computeam')
    model = fm.models.AtmosLines(logger=logger)

    # signal sideband
    spec_s = fm.tospectrum(array, reverse=False)
    spec_s[:] = model.generate(1e-9 * spec_s.freq)

    # image sideband
    spec_i = fm.tospectrum(array, reverse=True)
    spec_i[:] = model.generate(1e-9 * spec_i.freq)

    if reverse:
        return fm.fromspectrum(spec_i, array)
    else:
        return fm.fromspectrum(spec_s, array)

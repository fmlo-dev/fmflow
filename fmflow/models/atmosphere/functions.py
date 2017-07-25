# coding: utf-8

# imported items
__all__ = [
    'atmoslines',
]

# standard library
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np


# functions
def atmoslines(array, weights=None, mode='fit', ch_tolerance=5):
    logger = getLogger('fmflow.models.atmoslines')
    model = fm.models.AtmosLines(ch_tolerance, logger=logger)

    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    vrad = array.vrad.values.mean()

    if mode == 'fit':
        logger.info('mode: fit')
        tb = model.fit(freq, spec, vrad)
    elif mode == 'generate':
        logger.info('mode: generate')
        tb = model.generate(freq, vrad)
    else:
        logger.error('invalid mode')
        raise ValueError(mode)

    return fm.full_like(array, tb)

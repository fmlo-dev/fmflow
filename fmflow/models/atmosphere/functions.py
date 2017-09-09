# coding: utf-8

# public items
__all__ = [
    'atmoslines',
    'computeam',
]

# standard library
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np


# functions
@fm.chunk('array', 'weights')
def atmoslines(array, reverse=False, weights=None, snr_threshold=5, ch_tolerance=5):
    params = locals()
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug(params)

    freq = fm.getfreq(array, reverse, unit='GHz').values
    spec = fm.getspec(array, reverse, weights=weights).values
    noise = fm.getnoise(array, reverse, weights=weights).values
    vrad = array.vrad.values.mean()

    model = fm.models.AtmosLines(snr_threshold, ch_tolerance, logger=logger)
    tb = model.fit(freq, spec, noise, vrad)
    return fm.full_like(array, tb)


def computeam(array, reverse=False):
    params = locals()
    logger = getLogger('fmflow.models.computeam')
    logger.debug(params)

    freq = fm.getfreq(array, reverse, unit='GHz').values
    model = fm.models.AtmosLines(logger=logger)
    tb = model.generate(freq)
    return fm.full_like(array, tb)

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
def astrolines(
        array, reverse=False, weights=None, function='gaussian',
        snr_threshold=5, subtraction_gain=0.5
    ):
    params = locals()
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug(params)

    model = fm.models.AstroLines(
        function, snr_threshold, subtraction_gain, logger=logger
    )

    freq  = fm.getfreq(array, reverse, unit='GHz').values
    spec  = fm.getspec(array, reverse, weights=weights).values
    noise = fm.getnoise(array, reverse, weights=weights).values
    tb = model.fit(freq, spec, noise)
    return fm.full_like(array, tb)

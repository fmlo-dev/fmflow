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
@fm.chunk('array')
def astrolines(
        array, weights=None, fit_function='gaussian',
        snr_threshold=5, subtraction_gain=0.5
    ):
    params = locals()
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug(params)

    model = fm.models.AstroLines(
        fit_function, snr_threshold, subtraction_gain, logger=logger
    )

    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    nrms = fm.demodulate(fm.ones_like(array)).sum('t').values**-0.5
    nrms /= np.min(nrms)
    tb = model.fit(freq, spec, nrms)
    return fm.full_like(array, tb)

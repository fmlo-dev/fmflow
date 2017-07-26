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
@fm.timechunk
def astrolines(
        array, weights=None, fit_function='gaussian',
        snr_threshold=5, subtraction_gain=0.5
    ):
    logger = getLogger('fmflow.models.atmoslines')
    logger.info('fit function: {0}'.format(fit_function))
    logger.info('S/N threshold: {0}'.format(snr_threshold))
    logger.info('subtracttion gain: {0}'.format(subtraction_gain))

    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values

    # normalized rms
    nrms = fm.demodulate(fm.ones_like(array)).sum('t').values**-0.5
    nrms /= np.min(rms)

    model = fm.models.AstroLines(
        fit_function, snr_threshold, subtraction_gain, logger=logger
    )
    tb = model.fit(freq, spec, nrms)
    return fm.full_like(array, tb)

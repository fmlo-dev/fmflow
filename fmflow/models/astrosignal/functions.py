# coding: utf-8

# imported items
__all__ = [
    'astrolines',
]

# standard library
import logging

# dependent packages
import fmflow as fm
import numpy as np


# functions
def astrolines(
        array, weights=None, mode='fit', fit_function='gaussian',
        snr_threshold=5, subtraction_gain=0.5
    ):
    logger = logging.getLogger('fmflow.models.atmoslines')

    model = fm.models.AstroLines(
        fit_function, snr_threshold, subtraction_gain, logger=logger
    )

    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    weight = fm.demodulate(fm.ones_like(array)).sum('t')
    weight = np.sqrt(weight / weight.max()).values

    if mode == 'fit':
        logger.info('mode: fit')
        logger.info('fit function: {0}'.format(fit_function))
        logger.info('S/N threshold: {0}'.format(snr_threshold))
        logger.info('subtracttion gain: {0}'.format(subtraction_gain))
        tb = model.fit(freq, spec, weight)
    elif mode == 'raw':
        logger.info('mode: raw')
        tb = spec
    else:
        logger.error('invalid mode')
        raise ValueError(mode)

    return fm.full_like(array, tb)

# coding: utf-8

# public items
__all__ = [
    'atmoslines',
]

# standard library
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np


# functions
@fm.timechunk
def atmoslines(array, weights=None, output='tb', ch_tolerance=5):
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug('ch_tolerance: {0}'.format(ch_tolerance))

    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    vrad = array.vrad.values.mean()

    model = fm.models.AtmosLines(ch_tolerance, logger=logger)
    tau, tb = model.fit(freq, spec, vrad)

    if output == 'tau':
        logger.info('output: tau')
        return fm.full_like(array, tau)
    elif output == 'tb':
        logger.info('output: tb')
        return fm.full_like(array, tb)
    else:
        logger.error('invalid output')
        raise ValueError(output)

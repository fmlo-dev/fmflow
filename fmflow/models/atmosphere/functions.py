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
@fm.timechunk
def atmoslines(array, weights=None, output='tb', ch_tolerance=5):
    model = fm.models.AtmosLines(ch_tolerance, logger=logger)
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug(model.params)

    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    vrad = array.vrad.values.mean()
    tau, tb = model.fit(freq, spec, vrad)

    if output == 'tau':
        logger.debug('output: tau')
        return fm.full_like(array, tau)
    elif output == 'tb':
        logger.debug('output: tb')
        return fm.full_like(array, tb)
    else:
        logger.error('invalid output')
        raise ValueError(output)


def computeam(array, output='tb'):
    model = fm.models.AtmosLines(logger=logger)
    logger = getLogger('fmflow.models.computeam')
    logger.debug(model.params)

    freq = fm.getfreq(array, unit='GHz').values
    tau, tb = model.generate(freq)

    if output == 'tau':
        return fm.full_like(array, tau)
    elif output == 'tb':
        return fm.full_like(array, tb)
    else:
        logger.error('invalid output')
        raise ValueError(output)

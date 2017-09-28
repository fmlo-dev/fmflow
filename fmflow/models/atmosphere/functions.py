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
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug({k:v for k,v in locals().items() if k!='logger'})

    model = fm.models.AtmosLines(snr_threshold, ch_tolerance, logger=logger)

    spec = fm.tospectrum(array, weights, reverse)
    vrad = array.vrad.values.mean()
    spec[:] = model.fit(1e-9*spec.freq, spec, spec.noise, vrad)
    return fm.fromspectrum(spec, array)


def computeam(array, reverse=False):
    logger = getLogger('fmflow.models.computeam')
    logger.debug({k:v for k,v in locals().items() if k!='logger'})

    model = fm.models.AtmosLines(logger=logger)

    spec = fm.tospectrum(array, None, reverse)
    spec[:] = model.generate(1e-9*spec.freq)
    return fm.fromspectrum(spec, array)

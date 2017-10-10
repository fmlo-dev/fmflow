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
        array, weights=None, reverse=False, freqlim=None,
        function='gaussian', despiking=False, snr_threshold=10,
        subtraction_gain=1.0
    ):
    logger = getLogger('fmflow.models.atmoslines')
    model  = fm.models.AstroLines(
        function, despiking, snr_threshold, subtraction_gain, logger=logger
    )
    logger.debug(model)

    spec = fm.tospectrum(array, weights, reverse)
    spec[:] = model.fit(1e-9*spec.freq, spec, spec.noise, freqlim)
    return fm.fromspectrum(spec, array)

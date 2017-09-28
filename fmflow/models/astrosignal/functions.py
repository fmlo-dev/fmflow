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
        array, weights=None, reverse=False, function='gaussian',
        despiking=False, snr_threshold=5, subtraction_gain=0.5
    ):
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug({k:v for k,v in locals().items() if k!='logger'})

    model = fm.models.AstroLines(
        function, despiking, snr_threshold, subtraction_gain, logger=logger
    )

    spec = fm.tospectrum(array, weights, reverse)
    spec[:] = model.fit(1e-9*spec.freq, spec, spec.noise)
    return fm.fromspectrum(spec, array)

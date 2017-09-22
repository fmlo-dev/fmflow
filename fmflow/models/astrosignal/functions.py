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
        array, reverse=False, weights=None, function='cutoff',
        despiking=True, snr_threshold=5, subtraction_gain=0.5
    ):
    params = locals()
    logger = getLogger('fmflow.models.atmoslines')
    logger.debug(params)

    model = fm.models.AstroLines(
        function, despiking, snr_threshold, subtraction_gain, logger=logger
    )

    spec = fm.tospectrum(array, weights, reverse)
    spec[:] = model.fit(1e-9*spec.freq, spec, spec.noise)
    return fm.fromspectrum(spec, array)

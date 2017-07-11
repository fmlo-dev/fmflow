# coding: utf-8

# imported items
__all__ = [
    'atmoslines',
]

# dependent packages
import fmflow as fm
import numpy as np


# functions
def atmoslines(array, weights=None, mode='fit', ch_tolerance=5):
    model = fm.models.AtmosLines(ch_tolerance)
    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    vrad = array.vrad.values.mean()

    if mode == 'fit':
        fm.logger.info('mode: fit')
        tb_ = model.fit(freq, spec, vrad)
    elif mode == 'generate':
        fm.logger.info('mode: generate')
        tb_ = model.generate(freq, vrad)

    array_ = fm.demodulate(array)
    return fm.modulate(fm.zeros_like(array_) + tb_)

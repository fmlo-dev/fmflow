# coding: utf-8

# imported items
__all__ = [
    'ozonelines',
]

# dependent packages
import fmflow as fm
import numpy as np


# functions
def ozonelines(array, weights=None, mode='fit'):
    model = fm.models.OzoneLines()
    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    vrad = array.vrad.values.mean()

    if mode == 'fit':
        tb_ = model.fit(freq, spec, vrad)
    elif mode == 'generate':
        tb_ = model.generate(freq, vrad)

    array_ = fm.demodulate(array)
    return fm.modulate(fm.zeros_like(array_)+tb_)

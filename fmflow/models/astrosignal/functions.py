# coding: utf-8

# imported items
__all__ = [
    'astrolines',
]

# dependent packages
import fmflow as fm
import numpy as np


# functions
def astrolines(array, weights=None, mode='fit', snr_threshold=10, ch_fwhm=5, ch_smooth=50):
    model = fm.models.AstroLines(snr_threshold, ch_fwhm, ch_smooth)
    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values

    if mode == 'fit':
        tb_ = model.fit(freq, spec)
    elif mode == 'smooth':
        tb_ = model.smooth(freq, spec)
    elif mode == 'raw':
        tb_ = spec

    array_ = fm.demodulate(array)
    return fm.modulate(fm.zeros_like(array_) + tb_)

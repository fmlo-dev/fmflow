# coding: utf-8

# imported items
__all__ = [
    'astrolines',
]

# dependent packages
import fmflow as fm
import numpy as np


# functions
def astrolines(array, weights=None, mode='fit', fit_function='gaussian', snr_threshold=5, ch_smooth=50):
    freq = fm.getfreq(array, unit='GHz').values
    spec = fm.getspec(array, weights=weights).values
    weight = fm.demodulate(fm.ones_like(array)).sum('t')
    weight = np.sqrt(weight / weight.max()).values

    model = fm.models.AstroLines(fit_function, snr_threshold, ch_smooth)

    if mode == 'fit':
        fm.logger.info('mode: fit')
        fm.logger.info('fit function: {0}'.format(fit_function))
        fm.logger.info('S/N threshold: {0}'.format(snr_threshold))
        tb_ = model.fit(freq, spec, weight)
    elif mode == 'smooth':
        fm.logger.info('mode: smooth')
        fm.logger.info('ch smooth: {0}'.format(ch_smooth))
        tb_ = model.smooth(spec)
    elif mode == 'raw':
        fm.logger.info('mode: raw')
        tb_ = spec
    else:
        fm.logger.error('invalid mode')
        raise ValueError(mode)

    array_ = fm.demodulate(array)
    return fm.modulate(fm.zeros_like(array_) + tb_)

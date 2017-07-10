# coding: utf-8

# imported items
__all__ = [
    'AstroLines',
]

# standard library

# dependent packages
import fmflow as fm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


# classes
class AstroLines(object):
    def __init__(self, snr_threshold=10, ch_fwhm=5, ch_smooth=50):
        self.params = {
            'snr_threshold': snr_threshold,
            'ch_fwhm': ch_fwhm,
            'ch_smooth': ch_smooth,
        }

    def fit(self, freq, spec):
        model = np.zeros_like(spec)
        resid = spec.copy()

        def snr(spec):
            return spec / fm.utils.mad(spec)

        while np.max(snr(resid)) > self.snr_threshold:
            cent0 = freq[np.argmax(resid)]
            ampl0 = np.max(resid)
            fwhm0 = self.ch_fwhm

            p0 = [cent0, fwhm0, ampl0]
            bs = ([np.min(freq), 0.0, 0.0], [np.max(freq), 10.0*fwhm0, 10.0*ampl0])
            popt, pcov = curve_fit(fm.utils.gaussian, freq, resid, p0, bounds=bs)

            model += 0.5 * fm.utils.gaussian(freq, *popt)
            resid -= 0.5 * fm.utils.gaussian(freq, *popt)

        return model

    def _smooth(self, freq, spec):
        return gaussian_filter(spec, self.ch_smooth)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'AstroLines'

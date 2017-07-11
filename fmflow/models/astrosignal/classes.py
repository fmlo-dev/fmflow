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
    def __init__(self, fit_function='gaussian', snr_threshold=5, ch_smooth=50):
        self.params = {
            'fit_function': fit_function,
            'snr_threshold': snr_threshold,
            'ch_smooth': ch_smooth,
        }

    def fit(self, freq, spec, weight):
        model = np.zeros_like(spec)
        resid = spec.copy()

        def snr(spec, weight):
            return spec*weight / fm.utils.mad(spec)

        while np.max(snr(resid, weight)) > self.snr_threshold:
            cent0 = freq[np.argmax(resid*weight)]
            ampl0 = resid[np.argmax(resid*weight)]
            fwhm0 = np.diff(freq).mean()

            p0 = [cent0, fwhm0, ampl0]
            popt, pcov = curve_fit(fm.utils.gaussian, freq, resid, p0)

            model += 0.5 * getattr(fm.utils, self.fit_function)(freq, *popt)
            resid -= 0.5 * getattr(fm.utils, self.fit_function)(freq, *popt)

        return model

    def smooth(self, spec):
        return gaussian_filter(spec, self.ch_smooth)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'AstroLines(fit_function={0}, snr_threshold={1}, ch_smooth={2})',
            self.fit_function, self.snr_threshold, self.ch_smooth
        )

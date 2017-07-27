# coding: utf-8

# public items
__all__ = [
    'AstroLines',
]

# dependent packages
import fmflow as fm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


# classes
class AstroLines(object):
    def __init__(
            self, fit_function='gaussian', snr_threshold=5,
            subtraction_gain=0.5, *, logger=None
        ):
        self.params = {
            'fit_function': fit_function,
            'snr_threshold': snr_threshold,
            'subtraction_gain': subtraction_gain,
        }

        self.logger = logger or fm.logger

    def fit(self, freq, spec, nrms=None):
        func = getattr(fm.utils, self.fit_function)

        if nrms is None:
            nrmsr = np.ones_like(spec)

        model = np.zeros_like(spec)
        resid = spec.copy()

        def snr(spec):
            return spec / (fm.utils.mad(spec) * nrms)

        while np.max(snr(resid)) > self.snr_threshold:
            cent0 = freq[np.argmax(resid/nrms)]
            ampl0 = resid[np.argmax(resid/nrms)]
            fwhm0 = np.diff(freq).mean()

            p0 = [cent0, fwhm0, ampl0]
            popt, pcov = curve_fit(fm.utils.gaussian, freq, resid, p0)

            model += self.subtraction_gain * func(freq, *popt)
            resid -= self.subtraction_gain * func(freq, *popt)

        return model

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'AstroLines(fit_function={0}, snr_threshold={1})',
            self.fit_function, self.snr_threshold
        )

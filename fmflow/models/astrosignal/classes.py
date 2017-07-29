# coding: utf-8

# public items
__all__ = [
    'AstroLines',
]

# standard library
import warnings

# dependent packages
import fmflow as fm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)


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
            nrms = np.ones_like(spec)

        model = np.zeros_like(spec)
        resid = spec.copy()

        def snr(spec):
            return spec / (fm.utils.mad(spec) * nrms)

        fwhm0 = np.mean(np.diff(freq))
        maxsnr = np.max(snr(resid))
        while maxsnr > self.snr_threshold:
            index = np.argmax(resid/nrms)
            cent0, ampl0 = freq[index], resid[index]
            p0 = [cent0, fwhm0, ampl0]

            try:
                popt, pcov = curve_fit(fm.utils.gaussian, freq, resid, p0)
                model += self.subtraction_gain * func(freq, *popt)
                resid -= self.subtraction_gain * func(freq, *popt)
                maxsnr = np.max(snr(resid))
                self.logger.debug('max residual S/N: {0}'.format(maxsnr))
            except RuntimeError:
                self.logger.warning('breaks with runtime error')
                break

        return model

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'AstroLines({0})'.format(self.params)

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
from .. import BaseModel
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)


# classes
class AstroLines(BaseModel):
    def __init__(
            self, function='gaussian', despiking=False,
            snr_threshold=10, subtraction_gain=1.0, *, logger=None
        ):
        super().__init__(logger)
        self.params = {
            'function': function,
            'despiking': despiking,
            'snr_threshold': snr_threshold,
            'subtraction_gain': subtraction_gain,
        }

    def fit(self, freq, spec, noise, freqlim=None):
        freq  = np.asarray(freq)
        spec  = np.asarray(spec)
        noise =  np.asarray(noise)

        # model making
        if self.function == 'none':
            model = spec.copy()
        elif self.function == 'cutoff':
            model = spec.copy()
            model[spec/noise<self.snr_threshold] = 0.0
        else:
            func = getattr(fm.utils, self.function)
            model = self._fit(func, freq, spec, noise, freqlim)

        if self.despiking:
            model = self._despike(model, noise)

        return model

    def _fit(self, func, freq, spec, noise, freqlim=None):
        fwhm0 = np.abs(np.diff(freq).mean())

        # parameter limits
        if freqlim is None:
            freqlim = [freq.min(), freq.max()]

        fwhmlim = [0.5*fwhm0, np.inf]
        ampllim = [noise.min(), np.inf]
        bs = list(zip(freqlim, fwhmlim, ampllim))

        # S/N function
        def _snr(spec):
            snr = spec / noise
            snr[freq<freqlim[0]] = 0
            snr[freq>freqlim[1]] = 0
            return snr

        model, resid = np.zeros_like(spec), spec.copy()
        snr = _snr(resid)
        while np.max(snr) > self.snr_threshold:
            index = np.argmax(snr)
            cent0, ampl0 = freq[index], resid[index]
            p0 = [cent0, fwhm0, ampl0]

            try:
                popt, pcov = curve_fit(func, freq, resid, p0, noise, bounds=bs)
                model += self.subtraction_gain * func(freq, *popt)
                resid -= self.subtraction_gain * func(freq, *popt)
                snr = _snr(resid)
                self.logger.debug('max residual S/N: {0}'.format(np.max(snr)))
            except RuntimeError:
                self.logger.warning('breaks with runtime error')
                break

        return model

    def _despike(self, model, noise):
        spec = np.hstack([model[1:],0]) + np.hstack([0,model[:-1]])
        model[spec/noise<0.5*self.snr_threshold] = 0.0
        return model

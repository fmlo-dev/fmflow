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
            self, function='cutoff', despiking=True,
            snr_threshold=5, subtraction_gain=0.5, *, logger=None
        ):
        super().__init__(logger)
        self.params = {
            'function': function,
            'despiking': despiking,
            'snr_threshold': snr_threshold,
            'subtraction_gain': subtraction_gain,
        }

    def fit(self, freq, spec, noise):
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
            model = self._fit(func, freq, spec, noise)

        if self.despiking:
            model = self._despike(model, noise)

        return model

    def _fit(self, func, freq, spec, noise):
        model = np.zeros_like(spec)
        resid = spec.copy()

        def snr(spec):
            return spec / noise

        fwhm0 = np.mean(np.abs(np.diff(freq)))
        maxsnr = np.max(snr(resid))
        while maxsnr > self.snr_threshold:
            index = np.argmax(snr(resid))
            cent0, ampl0 = freq[index], resid[index]
            p0 = [cent0, fwhm0, ampl0]

            try:
                popt, pcov = curve_fit(func, freq, resid, p0)
                model += self.subtraction_gain * func(freq, *popt)
                resid -= self.subtraction_gain * func(freq, *popt)
                maxsnr = np.max(snr(resid))
                self.logger.debug('max residual S/N: {0}'.format(maxsnr))
            except RuntimeError:
                self.logger.warning('breaks with runtime error')
                break

        return model

    def _despike(self, model, noise):
        spec = np.hstack([model[1:],0]) + np.hstack([0,model[:-1]])
        model[spec/noise<0.5*self.snr_threshold] = 0.0
        return model

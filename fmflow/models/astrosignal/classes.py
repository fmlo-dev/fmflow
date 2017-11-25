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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import convolve, gaussian, savgol_filter
warnings.simplefilter('ignore', OptimizeWarning)


# classes
class AstroLines(BaseModel):
    def __init__(self, function='cutoff', despiking=True, snr_threshold=10,
                 deconvolution_width=3, subtraction_gain=0.5,
                 *, convergence=1e-3, n_maxiters=10000, logger=None):
        params = {
            'function': function,
            'despiking': despiking,
            'snr_threshold': snr_threshold,
            'deconvolution_width': deconvolution_width,
            'subtraction_gain': subtraction_gain,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }
        super().__init__(params, logger)

    def fit(self, freq, spec, noise, freqlim=None):
        freq  = np.asarray(freq).copy()
        spec  = np.asarray(spec).copy()
        noise = np.asarray(noise).copy()

        # freq limits
        if freqlim is not None:
            spec[freq<freqlim[0]] = 0
            spec[freq>freqlim[1]] = 0

        # fitting
        if self.function is None or not np.any(spec):
            model = spec.copy()
        elif self.function == 'cutoff':
            with fm.utils.ignore_numpy_errors():
                model = spec.copy()
                model[spec/noise < self.snr_threshold] = 0
        elif self.function == 'deconvolution':
            model = self._deconvolve(freq, spec, noise)
        else:
            func  = getattr(fm.utils, self.function)
            model = self._funcfit(func, freq, spec, freqlim)

        # despiking (if any)
        if self.despiking:
            self._despike(model, noise)

        return model

    def _deconvolve(self, freq, spec, noise):
        model = np.zeros_like(spec)
        resid = spec.copy()
        snr = resid / noise

        while np.max(snr) > self.snr_threshold:
            xpeak, ypeak = np.argmax(snr), resid[np.argmax(snr)]
            left, right = resid[:xpeak+1][::-1], resid[xpeak:]

            xhalf_l = xpeak - np.where(left<=0.5*ypeak)[0][0]
            xhalf_r = xpeak + np.where(right<=0.5*ypeak)[0][0]
            xpeak = int(xhalf_r/2 + xhalf_l/2) # update
            left, right = resid[:xpeak+1][::-1], resid[xpeak:] # update

            xlimit = int(self.deconvolution_width * (xhalf_r-xhalf_l)/2)
            xmin_l = xpeak - np.argmin(left[:xlimit])
            xmin_r = xpeak + np.argmin(right[:xlimit])
            ymin_l = resid[xmin_l]
            ymin_r = resid[xmin_r]

            signal = resid[xmin_l:xmin_r+1] - (ymin_l+ymin_r)/2
            if not np.any(signal):
                break

            model[xmin_l:xmin_r+1] += signal
            resid[xmin_l:xmin_r+1] -= signal
            snr = resid / noise

        return model

    def _funcfit(self, func, freq, spec, freqlim):
        # parameter limits
        if freqlim is None:
            freqlim = freq.min(), freq.max()

        chwidth = np.abs(np.diff(freq).mean())
        fwhmlim = 0.5*chwidth, np.inf

        # convergence
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters,
                                  centering=False, raise_exception=True)

        try:
            model = np.zeros_like(spec)
            resid = spec.copy()
            while not cv(model):
                self.logger.debug(cv)
                peak = np.argmax(resid)
                ampllim = 0.9*resid[peak], 1.1*resid[peak]
                p0 = freq[peak], chwidth, resid[peak]
                bs = list(zip(freqlim, fwhmlim, ampllim))
                popt, pcov = curve_fit(func, freq, resid, p0, bounds=bs)

                model += self.subtraction_gain * func(freq, *popt)
                resid -= self.subtraction_gain * func(freq, *popt)
        except RuntimeError:
            self.logger.warning('breaks with runtime error')
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        return model

    def _despike(self, model, noise):
        with fm.utils.ignore_numpy_errors():
            neighbors = np.hstack([model[1:],0]) + np.hstack([0,model[:-1]])
            model[neighbors/noise < 0.5*self.snr_threshold] = 0

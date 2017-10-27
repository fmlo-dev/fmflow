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
    def __init__(self, snr_threshold=10, cutoff_width=3, despiking=True,
                 fit_function=None, subtraction_gain=0.5, *, convergence=1e-3,
                 n_maxiters=10000, logger=None):
        super().__init__(logger)
        self.params = {
            'snr_threshold': snr_threshold,
            'cutoff_width': cutoff_width,
            'despiking': despiking,
            'fit_function': fit_function,
            'subtraction_gain': subtraction_gain,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }

    def fit(self, freq, spec, noise, freqlim=None):
        freq  = np.asarray(freq).copy()
        spec  = np.asarray(spec).copy()
        noise = np.asarray(noise).copy()

        # cutoff
        spec[freq<freqlim[0]] = 0
        spec[freq>freqlim[1]] = 0
        spec = self._cutoff(freq, spec, noise)

        # despiking
        if self.despiking:
            spec = self._despike(spec, noise)

        # fitting
        if not np.any(spec):
            return spec
        elif self.fit_function is None:
            return spec
        else:
            func = getattr(fm.utils, self.fit_function)
            return self._fit(func, freq, spec, freqlim)

    def _cutoff(self, freq, spec, noise):
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

            xlimit = int(self.cutoff_width * (xhalf_r-xhalf_l)/2)
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

    def _despike(self, spec, noise):
        neighbors = np.hstack([spec[1:],0]) + np.hstack([0,spec[:-1]])
        spec[neighbors/noise < 0.5*self.snr_threshold] = 0
        return spec

    def _fit(self, func, freq, spec, freqlim):
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

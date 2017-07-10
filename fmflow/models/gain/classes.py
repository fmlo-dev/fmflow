# coding: utf-8

# imported items
__all__ = [
    'RFGain',
]

# dependent packages
import fmflow as fm
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


# classes
class RFGain(object):
    def __init__(self, ch_smooth=50, convergence=0.01, n_maxiters=100, *, include_logain=False):
        self.params = {
            'ch_smooth': ch_smooth,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
            'include_logain': include_logain,
        }

    def fit(self, ON):
        logON = np.log10(ON)

        # interpolated fmch
        fmch = logON.fmch.values
        ifmch = np.arange(fmch.min(), fmch.max()+3, 3)
        ifmch[ifmch>fmch.max()] = fmch.max()

        # interpolated logON
        interp = interp1d(np.sort(fmch), logON[np.argsort(fmch)], axis=0)
        ilogON = fm.array(interp(ifmch), {'fmch': ifmch})

        # interpolated logGrf and logGcom
        ilogGrf = fm.zeros_like(ilogON)
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters)
        try:
            while not cv(ilogGrf):
                ilogGcom = fm.models.empca(ilogON-ilogGrf, None, 1, centering=False)
                ilogGrf  = self._estimate_logGrf(ilogON-ilogGcom)
        except StopIteration:
            fm.logger.warning('reached maximum iteration')

        # revert logGrf to original plane
        if self.include_logain:
            ilogGlo = ilogGcom.mean(1, keepdims=True)
            ilogGrf = ilogGrf + ilogGlo - ilogGlo.mean()

        interp = interp1d(ifmch, ilogGrf, axis=0)
        return fm.zeros_like(logON) + 10**interp(fmch)

    def _estimate_logGrf(self, logX):
        logGrf_ = gaussian_filter(fm.getspec(logX), self.ch_smooth)
        logGrf_ = fm.zeros_like(fm.demodulate(logX)) + logGrf_
        return fm.modulate(logGrf_)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'RFGain(ch_step={0}, n_maxiters={1})',
            self.ch_step, self.n_maxiters,
        )

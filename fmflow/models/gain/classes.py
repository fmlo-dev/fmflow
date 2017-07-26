# coding: utf-8

# public items
__all__ = [
    'ONGain',
]

# dependent packages
import fmflow as fm
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


# classes
class ONGain(object):
    def __init__(
            self, ch_smooth=50, convergence=0.01, n_maxiters=100,
            include_logain=False, *, logger=None
        ):
        self.params = {
            'ch_smooth': ch_smooth,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
            'include_logain': include_logain,
        }

        self.logger = logger or fm.logger

    def fit(self, ON):
        logON = np.log10(ON)

        # interpolated fmch
        fmch = logON.fmch.values
        ifmch = np.arange(fmch.min(), fmch.max()+1, 1)
        ifmch[ifmch>fmch.max()] = fmch.max()

        # interpolated logON
        interp = interp1d(np.sort(fmch), logON[np.argsort(fmch)], axis=0)
        ilogON = fm.array(interp(ifmch), {'fmch': ifmch})

        # interpolated logGif, logGlo, logGrf
        ilogGrf = fm.zeros_like(ilogON)
        ilogGlo = ilogGrf.mean('ch')
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters)
        try:
            while not cv(ilogGrf):
                self.logger.debug(cv.status)
                ilogGif = self._estimate_logGif(ilogON-ilogGrf-ilogGlo)
                ilogGlo = self._estimate_logGlo(ilogON-ilogGrf-ilogGif)
                ilogGrf = self._estimate_logGrf(ilogON-ilogGif-ilogGlo)
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        if self.include_logain:
            interp = interp1d(ifmch, ilogGrf+ilogGlo, axis=0)
        else:
            interp = interp1d(ifmch, ilogGrf, axis=0)

        return fm.zeros_like(logON) + 10**interp(fmch)

    def _estimate_logGif(self, logX):
        return logX.mean('t')

    def _estimate_logGlo(self, logX):
        logglo = logX.mean('ch')
        return logglo - logglo.mean()

    def _estimate_logGrf(self, logX):
        loggrf = gaussian_filter(fm.getspec(logX), self.ch_smooth)
        return fm.full_like(logX, loggrf-loggrf.mean())

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'RFGain(ch_step={0}, n_maxiters={1})',
            self.ch_step, self.n_maxiters,
        )

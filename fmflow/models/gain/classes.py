# coding: utf-8

# public items
__all__ = [
    'Gain',
]

# dependent packages
import fmflow as fm
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


# classes
class Gain(object):
    def __init__(
            self, include=['RF', 'LO'], ch_smooth=1,
            convergence=0.01, n_maxiters=100, *, logger=None
        ):
        if not set(include) <= {'RF', 'LO', 'IF'}:
            raise ValueError(include)

        self.params = {
            'include': include,
            'ch_smooth': ch_smooth,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }

        self.logger = logger or fm.logger

    def fit(self, X):
        logX = np.log10(X)
        ilogX = self.to_ilogX(logX)

        # initial arrays
        ilogGif = fm.zeros_like(ilogX[0])
        ilogGlo = fm.zeros_like(ilogX[:,0])
        ilogGrf = fm.zeros_like(ilogX)

        # convergence
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters, True)

        # algorithm
        try:
            while not cv(ilogX-ilogGif-ilogGlo-ilogGrf):
                self.logger.debug(cv.status)
                ilogGif = self._estimate_ilogGif(ilogX-ilogGrf-ilogGlo)
                ilogGlo = self._estimate_ilogGlo(ilogX-ilogGrf-ilogGif)
                ilogGrf = self._estimate_ilogGrf(ilogX-ilogGif-ilogGlo)
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        # return result
        ilogG = fm.zeros_like(ilogX)

        if 'RF' in self.include:
            ilogG += ilogGrf

        if 'LO' in self.include:
            ilogG += ilogGlo

        if 'IF' in self.include:
            ilogG += ilogGif

        logG = self.to_logX(ilogG)
        return fm.full_like(logX, 10**(logG.values))

    @staticmethod
    def to_ilogX(logX):
        bfmch = logX.fmch.values.tobytes()
        ifmch = np.arange(logX.fmch.min(), logX.fmch.max()+1)

        glogX = logX.groupby('fmch').mean('t')
        interp = interp1d(glogX.fmch, glogX, axis=0)
        return fm.array(interp(ifmch), {'fmch': ifmch}, {}, {'bfmch': bfmch})

    @staticmethod
    def to_logX(ilogX):
        ifmch = ilogX.fmch.values
        fmch = np.fromstring(ilogX.bfmch.values, int)

        interp = interp1d(ifmch, ilogX, axis=0)
        return fm.array(interp(fmch), {'fmch': fmch})

    def _estimate_ilogGif(self, ilogX):
        return ilogX.mean('t')

    def _estimate_ilogGlo(self, ilogX):
        ilogGlo = ilogX.mean('ch')
        return ilogGlo-ilogGlo.mean()

    def _estimate_ilogGrf(self, ilogX):
        ilogGrf = fm.getspec(ilogX)
        if (self.ch_smooth is not None) and self.ch_smooth:
            ilogGrf = savgol_filter(ilogGrf, self.ch_smooth, polyorder=3)

        return fm.full_like(ilogX, ilogGrf-ilogGrf.mean())

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'Gain({0})'.format(self.params)

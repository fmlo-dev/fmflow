# coding: utf-8

# public items
__all__ = [
    'ONGain',
]

# dependent packages
import fmflow as fm
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


# classes
class ONGain(object):
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

    def fit(self, ON):
        logON = np.log10(ON)
        ilogON = self.to_ilogON(logON)

        # initial arrays
        ilogGif = fm.zeros_like(ilogON[0])
        ilogGlo = fm.zeros_like(ilogON[:,0])
        ilogGrf = fm.zeros_like(ilogON)

        # convergence
        cv = fm.utils.Convergence(self.convergence, self.n_maxiters, True)

        # algorithm
        try:
            while not cv(ilogON-ilogGif-ilogGlo-ilogGrf):
                self.logger.debug(cv.status)
                ilogGif = self._estimate_ilogGif(ilogON-ilogGrf-ilogGlo)
                ilogGlo = self._estimate_ilogGlo(ilogON-ilogGrf-ilogGif)
                ilogGrf = self._estimate_ilogGrf(ilogON-ilogGif-ilogGlo)
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        # return result
        ilogGon = fm.zeros_like(ilogON)

        if 'RF' in self.include:
            ilogGon += ilogGrf

        if 'LO' in self.include:
            ilogGon += ilogGlo

        if 'IF' in self.include:
            ilogGon += ilogGif

        logGon = self.to_logON(ilogGon)
        return fm.full_like(logON, 10**(logGon.values))

    @staticmethod
    def to_ilogON(logON):
        bfmch = logON.fmch.values.tobytes()
        ifmch = np.arange(logON.fmch.min(), logON.fmch.max()+1)

        glogON = logON.groupby('fmch').mean('t')
        interp = interp1d(glogON.fmch, glogON, axis=0)
        return fm.array(interp(ifmch), {'fmch': ifmch}, {}, {'bfmch': bfmch})

    @staticmethod
    def to_logON(ilogON):
        ifmch = ilogON.fmch.values
        fmch = np.fromstring(ilogON.bfmch.values, int)

        interp = interp1d(ifmch, ilogON, axis=0)
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
        return 'ONGain({0})'.format(self.params)

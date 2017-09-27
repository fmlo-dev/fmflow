# coding: utf-8

# public items
__all__ = [
    'Gain',
]

# dependent packages
import fmflow as fm
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
savgol_filter = fm.numpyfunc(savgol_filter)


# classes
class Gain(object):
    def __init__(self, polyorders=[1,2,3], convergence=0.001, n_maxiters=100, logger=None):
        self.params = {
            'polyorders': polyorders,
            'convergence': convergence,
            'n_maxiters': n_maxiters,
        }
        self.outputs = {}
        self.logger = logger or fm.logger

    def fit(self, Pon):
        logG = np.log10(Pon)
        ilogG = self.to_ilogG(logG)

        # initial arrays
        ilogGif = ilogG[ilogG.fmch==0].mean('t')
        ilogGlo_offset = fm.zeros_like(ilogG)
        ilogGlo_slopes = fm.zeros_like(ilogG)
        ilogGlo = ilogGlo_offset + ilogGlo_slopes

        # convergence
        cv = fm.utils.Convergence(
            self.convergence, self.n_maxiters, raise_exception=True
        )

        # algorithm
        try:
            while not cv(ilogGlo):
                self.logger.debug(cv)
                ilogGlo_offset = self.estimate_offset(ilogG-ilogGif-ilogGlo_slopes)
                ilogGlo_slopes = self.estimate_slopes(ilogG-ilogGif-ilogGlo_offset)
                ilogGlo = ilogGlo_offset + ilogGlo_slopes
        except StopIteration:
            self.logger.warning('reached maximum iteration')

        # save outputs
        self.outputs['ilogG'] = ilogG
        self.outputs['ilogGif'] = ilogGif
        self.outputs['ilogGlo_offset'] = ilogGlo_offset
        self.outputs['ilogGlo_slopes'] = ilogGlo_slopes
        self.outputs['ilogG_residual'] = ilogG - ilogGif - ilogGlo

        return fm.full_like(logG, 10**self.to_logG(ilogGlo))

    def estimate_offset(self, logG):
        fmch = logG.fmch.values
        offset = logG.mean('ch')
        offset -= offset[fmch==0].values
        return offset

    def estimate_slopes(self, logG):
        fmch = logG.fmch.values
        popt = polyfit(fmch, logG, self.polyorders)
        slopes = fm.full_like(logG, polyval(fmch, popt).T)
        return slopes

    @staticmethod
    def to_ilogG(logG):
        bfmch = logG.fmch.values.tobytes()
        ifmch = np.arange(logG.fmch.min(), logG.fmch.max()+1)

        glogG = logG.groupby('fmch').mean('t')
        interp = interp1d(glogG.fmch, glogG, axis=0)
        return fm.array(interp(ifmch), {'fmch': ifmch}, {}, {'bfmch': bfmch})

    @staticmethod
    def to_logG(ilogG):
        ifmch = ilogG.fmch.values
        fmch = np.fromstring(ilogG.bfmch.values, int)

        interp = interp1d(ifmch, ilogG, axis=0)
        return fm.array(interp(fmch), {'fmch': fmch})

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'Gain({0})'.format(self.params)

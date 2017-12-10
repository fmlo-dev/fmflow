# coding: utf-8

# public items
__all__ = [
    'ONGain',
    'FMGain',
]

# standard library
from copy import deepcopy

# dependent packages
import fmflow as fm
import numpy as np
from .. import BaseModel
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
savgol_filter = fm.xarrayfunc(savgol_filter)

# module constants
i8 = 'i8'


# classes
class ONGain(BaseModel):
    def __init__(self, window_length=51, polyorder=3, *, logger=None):
        params = {
            'window_length': window_length,
            'polyorder': polyorder,
        }
        super().__init__(params, logger)

    def fit(self, Pon):
        logG = np.log10(Pon)
        ilogG = self.to_ilogG(logG)
        ilogG = savgol_filter(ilogG, self.window_length, self.polyorder, axis=0)
        ilogGif = ilogG[ilogG.fmch==0].mean('t')
        ilogGlo = ilogG - ilogGif

        return fm.full_like(logG, 10**self.to_logG(ilogGlo))

    @staticmethod
    def to_ilogG(logG):
        bfmch = logG.fmch.values.astype(i8).tobytes()
        ifmch = np.arange(logG.fmch.min(), logG.fmch.max()+1)
        glogG = logG.groupby('fmch').mean('t')
        ilogG = interp1d(glogG.fmch, glogG, axis=0)(ifmch)

        # coords
        tcoords = {'fmch': ifmch}
        chcoords = deepcopy(logG.fma.chcoords)
        scalarcoords = {'bfmch': bfmch}

        return fm.array(ilogG, tcoords, chcoords, scalarcoords)

    @staticmethod
    def to_logG(ilogG):
        ifmch = ilogG.fmch.values
        fmch  = np.fromstring(ilogG.bfmch.values, i8)
        logG  = interp1d(ifmch, ilogG, axis=0)(fmch)

        # coords
        tcoords = {'fmch': fmch}
        chcoords = deepcopy(ilogG.fma.chcoords)

        return fm.array(logG, tcoords, chcoords)


# temporary
FMGain = ONGain

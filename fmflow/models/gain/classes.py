# coding: utf-8

# public items
__all__ = [
    'ONGain',
]

# dependent packages
import fmflow as fm
import numpy as np
from .. import BaseModel
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
savgol_filter = fm.numpyfunc(savgol_filter)

# module constants
i8 = 'i8'


# classes
class ONGain(BaseModel):
    def __init__(self, window_length=51, polyorder=3, logger=None):
        super().__init__(logger)
        self.params = {
            'window_length': window_length,
            'polyorder': polyorder,
        }

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
        interp = interp1d(glogG.fmch, glogG, axis=0)
        return fm.array(interp(ifmch), {'fmch': ifmch}, {}, {'bfmch': bfmch})

    @staticmethod
    def to_logG(ilogG):
        ifmch = ilogG.fmch.values
        fmch = np.fromstring(ilogG.bfmch.values, i8)

        interp = interp1d(ifmch, ilogG, axis=0)
        return fm.array(interp(fmch), {'fmch': fmch})

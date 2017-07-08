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
    def __init__(self, ch_step=2, ch_smooth=50, n_maxiters=5):
        self.params = {
            'ch_step': ch_step,
            'ch_smooth': ch_smooth,
            'n_maxiters': n_maxiters,
        }

    def fit(self, ON):
        logON = np.log10(ON)
        fmch = logON.fmch.values
        sfmch = np.arange(fmch.min(), fmch.max()+self.ch_step, self.ch_step)
        sfmch[sfmch>fmch.max()] = fmch.max()
        spl = interp1d(np.sort(fmch), logON[np.argsort(fmch)], axis=0)
        slogON = fm.array(spl(sfmch), {'fmch': sfmch})

        slogGrf = fm.zeros_like(slogON)
        for n in range(self.n_maxiters):
            slogGcom = fm.models.empca(slogON-slogGrf, None, 1, centering=False)
            slogGrf  = self._estimate_slogGrf(slogON-slogGcom)

        Grf = fm.zeros_like(logON)
        spl = interp1d(sfmch, slogGrf, axis=0)
        return Grf + 10**spl(fmch)

    def _estimate_slogGrf(self, slogX):
        sloggrf_ = gaussian_filter(fm.getspec(slogX), self.ch_smooth)
        slogGrf_ = fm.zeros_like(fm.demodulate(slogX)) + sloggrf_
        return fm.modulate(slogGrf_)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'RFGain(ch_step={0}, n_maxiters={1})',
            self.ch_step, self.n_maxiters,
        )

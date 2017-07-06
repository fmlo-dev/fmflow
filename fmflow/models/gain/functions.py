# coding: utf-8

# imported items
__all__ = [
    'rfgain',
]

# dependent packages
import fmflow as fm
import numpy as np

# functions
def rfgain(ON, ch_step=2, ch_smooth=50, n_maxiters=5):
    model = fm.models.RFGain(ch_step, ch_smooth, n_maxiters)
    return model.fit(ON)

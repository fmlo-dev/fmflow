# coding: utf-8

# imported items
__all__ = [
    'rfgain',
]

# dependent packages
import fmflow as fm
import numpy as np

# functions
def rfgain(ON, ch_smooth=50, convergence=0.01, n_maxiters=100, *, include_logain=False):
    model = fm.models.RFGain(ch_smooth, convergence, n_maxiters)
    return model.fit(ON)

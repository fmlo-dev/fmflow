# coding: utf-8

# public items
__all__ = [
    'rfgain',
]

# standard library
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np

# functions
def rfgain(ON, ch_smooth=50, convergence=0.01, n_maxiters=100, *, include_logain=False):
    logger = getLogger('fmflow.models.rfgain')

    model = fm.models.RFGain(
        ch_smooth, convergence, n_maxiters, include_logain, logger=logger
    )
    return model.fit(ON)

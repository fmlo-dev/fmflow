# coding: utf-8

# public items
__all__ = [
    'ongain',
    'fmgain',
]

# standard library
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np


# functions
@fm.chunk('Pon')
def ongain(Pon, window_length=51, polyorder=3):
    logger = getLogger('fmflow.models.ongain')
    model  = fm.models.ONGain(window_length, polyorder, logger=logger)
    return model.fit(Pon)


# temporary
fmgain = ongain

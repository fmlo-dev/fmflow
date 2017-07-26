# coding: utf-8

# public items
__all__ = [
    'ongain',
]

# standard library
from logging import getLogger

# dependent packages
import fmflow as fm
import numpy as np

# functions
@fm.timechunk
def ongain(ON, include=['RF', 'LO'], convergence=0.01, n_maxiters=100):
    logger = getLogger('fmflow.models.ongain')
    model = fm.models.ONGain(include, convergence, n_maxiters, logger=logger)
    return model.fit(ON)

# coding: utf-8

# public items
__all__ = [
    'ongain',
    'rgain',
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


def rgain(Gon):
    logger = getLogger('fmflow.models.rgain')
    iGon = fm.models.ONGain.to_ilogON(Gon)
    gr = iGon[iGon.fmch==0][0]
    return fm.zeros_like(Gon[0].drop(Gon.fm.tcoords.keys())) + gr

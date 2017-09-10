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
@fm.chunk('Pon')
def ongain(Pon, include=['RF', 'LO'], ch_smooth=None, convergence=0.01, n_maxiters=100):
    params = locals()
    logger = getLogger('fmflow.models.ongain')
    logger.debug(params)

    model = fm.models.Gain(include, ch_smooth, convergence, n_maxiters, logger=logger)
    return model.fit(Pon)


def rgain(Gon):
    params = locals()
    logger = getLogger('fmflow.models.rgain')
    logger.debug(params)

    iGon = fm.models.Gain.to_ilogX(Gon)
    gr = iGon[iGon.fmch==0][0].values
    return fm.full_like(Gon[0].drop(Gon.fm.tcoords.keys()), gr)

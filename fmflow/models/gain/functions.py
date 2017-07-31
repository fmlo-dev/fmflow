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
def ongain(ON, include=['RF', 'LO'], ch_smooth=1, convergence=0.01, n_maxiters=100):
    logger = getLogger('fmflow.models.ongain')
    logger.debug('include: {0}'.format(include))
    logger.debug('ch_smooth: {0}'.format(ch_smooth))
    logger.debug('convergence: {0}'.format(convergence))
    logger.debug('n_maxiters: {0}'.format(n_maxiters))

    model = fm.models.ONGain(
        include, ch_smooth, convergence, n_maxiters, logger=logger
    )
    return model.fit(ON)


def rgain(Gon):
    logger = getLogger('fmflow.models.rgain')
    iGon = fm.models.ONGain.to_ilogON(Gon)
    gr = iGon[iGon.fmch==0][0].values
    return fm.full_like(Gon[0].drop(Gon.fm.tcoords.keys()), gr)

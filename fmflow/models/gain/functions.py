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

@fm.chunk('Pon')
def ongain(Pon, polyorders=[1,2,3], convergence=0.001, n_maxiters=100):
    logger = getLogger('fmflow.models.ongain')
    logger.debug({k:v for k,v in locals().items() if k!='logger'})

    model = fm.models.ONGain(polyorders, convergence, n_maxiters, logger=logger)
    return model.fit(Pon)

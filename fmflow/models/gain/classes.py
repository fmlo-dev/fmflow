# coding: utf-8

# imported items
__all__ = [
    'RFGain',
]

# dependent packages
import fmflow as fm
import numpy as np
from scipy.interpolate import UnivariateSpline


# classes
class RFGain(object):
    def __init__(self, R, include_isb=False):
        self.R = R
        self.info = {
            'include_isb': include_isb,
        }

    def fit(self, ON):
        pass

    def _fit(self):
        pass

    def __getattr__(self, name):
        return self.info[name]

    def __repr__(self):
        string = str.format(
            'RFGain(include_isb={0})',
            self.include_isb,
        )

        return string

# coding: utf-8

# imported items
__all__ = [
    'RFGain',
]

# dependent packages
import fmflow as fm
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


# classes
class RFGain(object):
    def __init__(self, include_logain=False, include_ifgain=False):
        self.info = {
            'include_logain': include_logain,
            'include_ifgain': include_ifgain,
        }

    def fit(self, X):
        pass

    def _fit(self):
        pass

    def __getattr__(self, name):
        return self.info[name]

    def __repr__(self):
        string = str.format(
            'RFGain(include_logain={0}, include_ifgain={1})',
            self.include_logain, self.include_ifgain,
        )

        return string

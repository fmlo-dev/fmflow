# coding: utf-8

# imported items
__all__ = [
    'ozonelines',
]

# dependent packages
import fmflow as fm
import numpy as np


# functions
def ozonelines(array, weights=None, fitmode='normal', smooth=50):
    """

    Args:
        array (xarray.DataArray):
        weights (xarray.DataArray):
        fitmode (str):
        smooth (int):

    Returns:
        tb (xarray.DataArray):

    """
    model = fm.models.OzoneLines(fitmode, smooth)
    freq = fm.getfreq(array, unit='GHz')
    spec = fm.getspec(array, weights=weights)
    vrad = array.vrad.values.mean()

    tb_ = model.fit(freq, spec, vrad)
    array_ = fm.demodulate(array)
    return fm.modulate(fm.zeros_like(array_ + tb_))

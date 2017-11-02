# coding: utf-8

# public items
__all__ = [
    'spectrum',
    'tospectrum',
    'fromspectrum',
]

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr


# functions
def spectrum(data, chcoords=None, scalarcoords=None, attrs=None, name=None):
    """Create a spectrum as an instance of xarray.DataArray with FM spectrum accessor.

    Args:
        data (numpy.ndarray): A 1D (channel) array.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        scalarcoords (dict, optional): A dictionary of values that don't label any axes.
        attrs (dict, optional): A dictionary of attributes to add to the instance.
        name (str, optional): A string that names the instance.

    Returns:
        spectrum (xarray.DataArray): A spectrum.

    """
    # initialize coords with default values
    spectrum = xr.DataArray(data, dims='ch', attrs=attrs, name=name)
    spectrum.fms._initcoords()

    # update coords with input values (if any)
    if chcoords is not None:
        spectrum.fms.updatecoords(chcoords, 'ch')

    if scalarcoords is not None:
        spectrum.fms.updatecoords(scalarcoords)

    return spectrum


def tospectrum(array, weights=None, reverse=False):
    """Create a spectrum from an array.

    Args:
        array (xarray.DataArray): An array.
        weights (xarray.DataArray, optional): A weight array.
        reverse (bool, optional): If True, the array is reverse-demodulated
            (i.e. -1 * fmch is used for demodulation). Default is False.

    Returns:
        spectrum (xarray.DataArray): A spectrum.

    """
    return xr.DataArray.fms.fromarray(array, weights, reverse)


def fromspectrum(spectrum, array):
    """Create an array filled with the spectrum.

    Args:
        spectrum (xarray.DataArray): A spectrum to be cast.
        array (xarray.DataArray): An array whose shape the spectrum is cast on.

    Returns:
        array (xarray.DataArray): An array filled with the spectrum.

    """
    return spectrum.fms.toarray(array)

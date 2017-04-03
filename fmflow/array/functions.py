# coding: utf-8

# imported items
__all__ = [
    'array', 'demodulate', 'modulate', 'getfreq', 'getspec',
    'ones', 'zeros', 'full', 'empty',
    'ones_like', 'zeros_like', 'full_like', 'empty_like',
    'align', 'concat', 'merge',
]

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr
from astropy import units as u
from xarray import align, concat, merge


# functions
def array(data, tcoords=None, chcoords=None, ptcoords=None, attrs=None, name=None):
    """Create a modulated array as an instance of xarray.DataArray with FM accessor.

    Args:
        data (array): A 2D (time x channel) array.
        tcoords (dict, optional): A dictionary of arrays that label time axis.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        ptcoords (dict, optional): A dictionary of values that don't label any axes (point-like).
        attrs (dict, optional): A dictionary of attributes to add to the instance.
        name (str, optional): A string that names the instance.

    Returns:
        array (xarray.DataArray): A modulated array.

    """
    # initialize coords with default values
    array = xr.DataArray(data, dims=('t', 'ch'), attrs=attrs, name=name)
    array.fm._initcoords()

    # update coords with input values (if any)
    if tcoords is not None:
        array.coords.update({key: ('t', tcoords[key]) for key in tcoords})

    if chcoords is not None:
        array.coords.update({key: ('ch', chcoords[key]) for key in chcoords})

    if ptcoords is not None:
        array.coords.update(ptcoords)

    return array


def zeros(shape, dtype=None, **kwargs):
    """Create a modulated array of given shape and type, filled with zeros.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): The desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (FMArray): A modulated array filled with zeros.

    """
    data = np.zeros(shape, dtype)
    return fm.array(data, **kwargs)


def ones(shape, dtype=None, **kwargs):
    """Create a modulated array of given shape and type, filled with ones.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): The desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (FMArray): A modulated array filled with ones.

    """
    data = np.ones(shape, dtype)
    return fm.array(data, **kwargs)


def full(shape, fill_value, dtype=None, **kwargs):
    """Create a modulated array of given shape and type, filled with `fill_value`.

    Args:
        shape (sequence of ints): 2D shape of the array.
        fill_value (scalar): Fill value.
        dtype (data-type, optional): The desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (FMArray): A modulated array filled with `fill_value`.

    """
    data = np.full(shape, dtype)
    return fm.array(data, **kwargs)


def empty(shape, dtype=None, **kwargs):
    """Create a modulated array of given shape and type, without initializing entries.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): The desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (FMArray): A modulated array without initializing entries.

    """
    data = np.empty(shape, dtype)
    return fm.array(data, **kwargs)


def zeros_like(array, dtype=None, keepmeta=True):
    """Create an array of zeros with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (xarray.DataArray): An array filled with zeros.

    """
    if keepmeta:
        return xr.zeros_like(array, dtype)
    else:
        return fm.zeros(array.shape, dtype)


def ones_like(array, dtype=None, keepmeta=True):
    """Create an array of ones with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (xarray.DataArray): An array filled with ones.

    """
    if keepmeta:
        return xr.ones_like(array, dtype)
    else:
        return fm.ones(array.shape, dtype)


def full_like(array, fill_value, dtype=None, keepmeta=True):
    """Create an array of `fill_value` with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        fill_value (scalar): Fill value.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (xarray.DataArray): An array filled with `fill_value`.

    """
    if keepmeta:
        return xr.full_like(array, dtype)
    else:
        return fm.full(array.shape, dtype)


def empty_like(array, dtype=None, keepmeta=True):
    """Create an array of empty with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (xarray.DataArray): An array without initializing entries.

    """
    if keepmeta:
        return fm.empty(array.shape,
            array.fm.tcoords, array.fm.chcoords, array.fm.ptcoords,
            array.attrs, array.name
        )
    else:
        return fm.empty(array.shape, dtype)


def demodulate(array, reverse=False):
    """Create a demodulated array from the modulated one.

    This function is only available when the array is modulated.

    Args:
        array (xarray.DataArray): A modulated array.
        reverse (bool, optional): If True, the array is reverse-demodulated
            (i.e. -1 * fmch is used for demodulation). Default is False.

    Returns:
        array (xarray.DataArray): A demodulated array.

    """
    return array.fm.demodulate(reverse)


def modulate(array):
    """Create a modulated array from the demodulated one.

    This function is only available when the array is demodulated.

    Args:
        array (xarray.DataArray): A demodulated array.

    Returns:
        array (xarray.DataArray): A modulated array.

    """
    return array.fm.modulate()


def getfreq(array, reverse=False, unit='Hz'):
    """Compute the observed frequency of given unit.

    If the array is reverse-demodulated, or modulated and `reverse=True`,
    this function returns `fimg` (the observed frequency of image sideband).
    Otherwise, this function returns `fsig` (that of signal sideband).

    Args:
        array (xarray.DataArray): An array. If it is modulated, then this function
            demodulates it with `reverse` option before computing the observed frequency.
        reverse (bool, optional): If True, and if the array is modulated, then
            the array is reverse-demodulated (i.e. -1 * fmch is used for demodulation).
            Default is False.
        unit (str, optional): An unit of the observed frequency. Default is Hz.

    Returns:
        freq (numpy.ndarray): An array of the observed frequency in given unit.

    """
    if array.fm.ismodulated:
        array = fm.demodulate(array, reverse)

    if array.fm.isdemodulated_r:
        freq_Hz = array.fimg.values
    else:
        freq_Hz = array.fsig.values

    return (freq_Hz*u.Hz).to(getattr(u, unit)).value


def getspec(array, reverse=False, weights=None):
    """Compute the time-averaged spectrum.

    If the array is reverse-demodulated, or modulated and `reverse=True`,
    this function computes the spectrum of image sideband.
    Otherwise, this function computes that of signal sideband.

    Args:
        array (xarray.DataArray): An array. If it is modulated, then this function
            demodulates it with `reverse` option before computing the spectrum.
        reverse (bool, optional): If True, and if the array is modulated, then
            the array is reverse-demodulated (i.e. -1 * fmch is used for demodulation).
            Default is False.
        weights (xarray.DataArray, optional): An array of weights associated with the array.
            The shape of it must be same as the input array.

    Returns:
        spec (numpy.ndarray): An array of the time-averaged spectrum.

    """
    if weights is not None:
        if array.fm.ismodulated:
            weights = fm.demodulate(weights)

    if array.fm.ismodulated:
        array = fm.demodulate(array, reverse)

    masked_array = np.ma.array(array, mask=np.isnan(array))
    return np.ma.average(masked_array, axis=0, weights=weights).data

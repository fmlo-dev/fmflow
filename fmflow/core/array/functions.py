# coding: utf-8

# public items
__all__ = [
    'array',
    'demodulate',
    'modulate',
    'mad',
    'ones',
    'zeros',
    'full',
    'empty',
    'ones_like',
    'zeros_like',
    'full_like',
    'empty_like',
    'save',
    'load',
    'chbinning',
]

# standard library
from uuid import uuid4

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr
from astropy import units as u
from scipy.special import erfinv

# module constants
MAD_TO_STD = (np.sqrt(2) * erfinv(0.5))**-1


# functions
def array(data, tcoords=None, chcoords=None, scalarcoords=None, attrs=None, name=None):
    """Create a modulated array as an instance of xarray.DataArray with FM array accessor.

    Args:
        data (numpy.ndarray): A 2D (time x channel) array.
        tcoords (dict, optional): A dictionary of arrays that label time axis.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        scalarcoords (dict, optional): A dictionary of values that don't label any axes.
        attrs (dict, optional): A dictionary of attributes to add to the instance.
        name (str, optional): A string that names the instance.

    Returns:
        array (xarray.DataArray): A modulated array.

    """
    # initialize coords with default values
    array = xr.DataArray(data, dims=('t', 'ch'), attrs=attrs, name=name)
    array.fma._initcoords()

    # update coords with input values (if any)
    if tcoords is not None:
        array.fma.updatecoords(tcoords, 't')

    if chcoords is not None:
        array.fma.updatecoords(chcoords, 'ch')

    if scalarcoords is not None:
        array.fma.updatecoords(scalarcoords)

    return array


def zeros(shape, dtype=None, **kwargs):
    """Create a modulated array of given shape and type, filled with zeros.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): The desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (xarray.DataArray): A modulated array filled with zeros.

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
        array (xarray.DataArray): A modulated array filled with ones.

    """
    data = np.ones(shape, dtype)
    return fm.array(data, **kwargs)


def full(shape, fill_value, dtype=None, **kwargs):
    """Create a modulated array of given shape and type, filled with `fill_value`.

    Args:
        shape (sequence of ints): 2D shape of the array.
        fill_value (scalar or numpy.ndarray): Fill value or array.
        dtype (data-type, optional): The desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (xarray.DataArray): A modulated array filled with `fill_value`.

    """
    return (fm.zeros(shape, **kwargs) + fill_value).astype(dtype)


def empty(shape, dtype=None, **kwargs):
    """Create a modulated array of given shape and type, without initializing entries.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): The desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (xarray.DataArray): A modulated array without initializing entries.

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
        fill_value (scalar or numpy.ndarray or xarray.DataArray): Fill value or array.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (xarray.DataArray): An array filled with `fill_value`.

    """
    if keepmeta:
        return (fm.zeros_like(array) + fill_value).astype(dtype)
    else:
        return fm.full(array.shape, fill_value, dtype)


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
        return fm.empty(array.shape, dtype,
            tcoords=array.fma.tcoords, chcoords=array.fma.chcoords,
            scalarcoords=array.fma.scalarcoords, attrs=array.attrs, name=array.name
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
    return array.fma.demodulate(reverse)


def modulate(array):
    """Create a modulated array from the demodulated one.

    This function is only available when the array is demodulated.

    Args:
        array (xarray.DataArray): A demodulated array.

    Returns:
        array (xarray.DataArray): A modulated array.

    """
    return array.fma.modulate()


def mad(array, dim=None, axis=None):
    """Compute the median absolute deviation (MAD) along the given dim or axis.

    Only one of the `dim` and `axis` arguments can be supplied.
    If neither are supplied, then mad will be calculated over axes.

    Args:
        array (xarray.DataArray): An input array.
        dim (str, optional): Dim along which the MADs are computed.
        axis (int, optional): Axis along which the MADs are computed.

    Returns:
        mad (xarray.DataArray): An array of the MAD.

    """
    return np.abs(array - array.median(dim, axis)).median(dim, axis)


def save(dataarray, filename=None):
    """Save a dataarray to a NetCDF file.

    Args:
        dataarray (xarray.DataArray): A dataarray to be saved.
        filename (str): A filename (used as <filename>.nc).
            If not spacified, random 8-character name will be used.

    """
    if filename is None:
        if dataarray.name is not None:
            filename = dataarray.name
        else:
            filename = uuid4().hex[:8]

    if not filename.endswith('.nc'):
        filename += '.nc'

    dataarray.to_netcdf(filename)


def load(filename, copy=True):
    """Load a dataarray from a NetCDF file.

    Args:
        filename (str): A file name (*.nc).
        copy (bool): If True, dataarray is copied in memory. Default is True.

    Returns:
        dataarray (xarray.DataArray): A loaded dataarray.

    """
    if copy:
        dataarray = xr.open_dataarray(filename).copy()
    else:
        dataarray = xr.open_dataarray(filename)

    if dataarray.name is None:
        dataarray.name = filename.rstrip('.nc')

    for key, val in dataarray.coords.items():
        if val.dtype.kind == 'S':
            dataarray[key] = val.astype('U')
        elif val.dtype == np.int32:
            dataarray[key] = val.astype('i8')

    return dataarray


def chbinning(array, size=2):
    """Binning an array along ch axis with given size.

    Args:
        array (xarray.DataArray): An input array.
        size (int): Ch length of a bin.

    Returns:
        binarray (xarray.DataArray): An output binned array.

    """
    if set(array.dims) == {'t', 'ch'}:
        shape = array.shape
    elif set(array.dims) == {'ch'}:
        shape = (1, *array.shape)

    if shape[1] % size:
        raise ValueError('ch shape cannot be divided by size')

    binshape = shape[0], int(shape[1]/size)
    binarray = fm.zeros(binshape, tcoords=array.fma.tcoords, scalarcoords=array.fma.scalarcoords)

    # binning of data
    binarray.values = array.values.reshape([*binshape, size]).mean(2)

    # binning of fsig, fimg
    binarray['fsig'].values = array['fsig'].values.reshape([binshape[1], size]).mean(1)
    binarray['fimg'].values = array['fimg'].values.reshape([binshape[1], size]).mean(1)

    # convert fmch (if any)
    if 'fmch' in array:
        binarray['fmch'].values = (array['fmch'].values / size).astype(int)

    if set(array.dims) == {'t', 'ch'}:
        return binarray.squeeze()
    elif set(array.dims) == {'ch'}:
        return binarray.squeeze().drop(binarray.fma.tcoords.keys())

# coding: utf-8

# imported items
__all__ = ['array', 'ones', 'zeros', 'full', 'empty']

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr


# functions
def array(data, tcoords=None, chcoords=None, ptcoords=None, attrs=None, name=None):
    """Create a modulated array as an instance of xarray.DataArray with FM accessor.

    Args:
        data (array): A 2D (time x channel) array.
        tcoords (dict, optional): A dictionary of arrays that label time axis.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        ptcoords (dict, optional): A dictionary of values that don't label any axes (point like).
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



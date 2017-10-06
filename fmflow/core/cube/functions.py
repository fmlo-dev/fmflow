# coding: utf-8

# public items
__all__ = []

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr


# functions
def cube(data, chcoords=None, ptcoords=None, attrs=None, name=None):
    """Create a cube as an instance of xarray.DataArray with FM accessor.

    Args:
        data (numpy.ndarray): A 3D (channel) array.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        ptcoords (dict, optional): A dictionary of values that don't label any axes (point-like).
        attrs (dict, optional): A dictionary of attributes to add to the instance.
        name (str, optional): A string that names the instance.

    Returns:
        cube (xarray.DataArray): A cube.

    """
    # initialize coords with default values
    cube = xr.DataArray(data, dims='ch', attrs=attrs, name=name)
    cube.fmc._initcoords()

    # update coords with input values (if any)
    if chcoords is not None:
        cube.coords.update({key: ('ch', val) for key, val in chcoords.items()})

    if ptcoords is not None:
        cube.coords.update(ptcoords)

    return cube


def tocube(array, weights=None, reverse=False):
    """Create a cube from an array.

    Args:
        array (xarray.DataArray): An array.
        weights (xarray.DataArray, optional): A weight array.
        reverse (bool, optional): If True, the array is reverse-demodulated
            (i.e. -1 * fmch is used for demodulation). Default is False.

    Returns:
        cube (xarray.DataArray): A cube.

    """
    return array.fmc.fromarray(weights, reverse)


def fromcube(cube, array):
    """Create an array filled with the cube.

    Args:
        cube (xarray.DataArray): A cube to be cast.
        array (xarray.DataArray): An array whose shape the cube is cast on.

    Returns:
        array (xarray.DataArray): An array filled with the cube.

    """
    return cube.fmc.toarray(array)

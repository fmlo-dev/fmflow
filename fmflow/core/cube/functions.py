# coding: utf-8

# public items
__all__ = [
    'cube',
    'tocube',
    'fromcube',
]

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr


# functions
def cube(data, xcoords=None, ycoords=None, chcoords=None,
         datacoords=None, scalarcoords=None, attrs=None, name=None):
    """Create a cube as an instance of xarray.DataArray with FM accessor.

    Args:
        data (numpy.ndarray): A 3D (channel) array.
        xcoords (dict, optional): A dictionary of arrays that label x axis.
        ycoords (dict, optional): A dictionary of arrays that label y axis.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        datacoords (dict, optional): A dictionary of values that label all axes.
        scalarcoords (dict, optional): A dictionary of values that don't label any axes.
        attrs (dict, optional): A dictionary of attributes to add to the instance.
        name (str, optional): A string that names the instance.

    Returns:
        cube (xarray.DataArray): A cube.

    """
    # initialize coords with default values
    cube = xr.DataArray(data, dims=('x', 'y', 'ch'), attrs=attrs, name=name)
    cube.fmc._initcoords()

    # update coords with input values (if any)
    if xcoords is not None:
        cube.fmc.updatecoords(xcoords, 'x')

    if ycoords is not None:
        cube.fmc.updatecoords(ycoords, 'y')

    if chcoords is not None:
        cube.fmc.updatecoords(chcoords, 'ch')

    if datacoords is not None:
        cube.fmc.updatecoords(datacoords, ('x', 'y', 'ch'))

    if scalarcoords is not None:
        cube.fmc.updatecoords(scalarcoords)

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

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
def cube(data, chcoords=None, ycoords=None, xcoords=None,
         datacoords=None, scalarcoords=None, attrs=None, name=None):
    """Create a cube as an instance of xarray.DataArray with FM accessor.

    Args:
        data (numpy.ndarray): A 3D (channel) array.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        ycoords (dict, optional): A dictionary of arrays that label y axis.
        xcoords (dict, optional): A dictionary of arrays that label x axis.
        datacoords (dict, optional): A dictionary of values that label all axes.
        scalarcoords (dict, optional): A dictionary of values that don't label any axes.
        attrs (dict, optional): A dictionary of attributes to add to the instance.
        name (str, optional): A string that names the instance.

    Returns:
        cube (xarray.DataArray): A cube.

    """
    # initialize coords with default values
    cube = xr.DataArray(data, dims=('ch', 'y', 'x'), attrs=attrs, name=name)
    cube.fmc._initcoords()

    # update coords with input values (if any)
    if chcoords is not None:
        cube.fmc.updatecoords(chcoords, 'ch')

    if ycoords is not None:
        cube.fmc.updatecoords(ycoords, 'y')

    if xcoords is not None:
        cube.fmc.updatecoords(xcoords, 'x')

    if datacoords is not None:
        cube.fmc.updatecoords(datacoords, ('ch', 'y', 'x'))

    if scalarcoords is not None:
        cube.fmc.updatecoords(scalarcoords)

    return cube


def tocube(array, weights=None, reverse=False,
           gridsize=10, gridunit='arcsec', gcf='besselgauss'):
    """Create a cube from an array.

    Args:
        array (xarray.DataArray): An array.
        weights (xarray.DataArray, optional): A weight array.
        reverse (bool, optional): If True, the array is reverse-demodulated
            (i.e. -1 * fmch is used for demodulation). Default is False.
        gridsize (float, optional): Grid size in units of `gridunit`.
        gridunit (str, optional): Grid unit of `gridsize`.
        gcf (str, optional): Grid convolution function.

    Returns:
        cube (xarray.DataArray): A cube.

    """
    return xr.DataArray.fmc.fromarray(array, weights, reverse,
                                      gridsize, gridunit, gcf)


def fromcube(cube, array):
    """Create an array filled with the cube.

    Args:
        cube (xarray.DataArray): A cube to be cast.
        array (xarray.DataArray): An array whose shape the cube is cast on.

    Returns:
        array (xarray.DataArray): An array filled with the cube.

    """
    return cube.fmc.toarray(array)

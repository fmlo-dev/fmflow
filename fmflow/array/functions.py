# coding: utf-8

# imported items
__all__ = ['array']

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



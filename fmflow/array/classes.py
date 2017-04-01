# coding: utf-8

# imported items
__all__ = []

# standard library
import re
from collections import OrderedDict
from datetime import datetime

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr

# constants
TCOORDS = lambda size=0: OrderedDict([
    ('fmch', ('t', np.zeros(size, dtype=int))),
    ('vrad', ('t', np.zeros(size, dtype=float))),
    ('xrel', ('t', np.zeros(size, dtype=float))),
    ('yrel', ('t', np.zeros(size, dtype=float))),
    ('time', ('t', np.tile(datetime(2000,1,1), size))),
])

CHCOORDS = lambda size=0: OrderedDict([
    ('indx', ('ch', np.arange(size, dtype=int))),
    ('fsig', ('ch', np.arange(size, dtype=float))),
    ('fimg', ('ch', np.arange(size, dtype=float)[::-1])),
])

PTCOORDS = lambda size=0: OrderedDict([
    ('status', 'MODULATED'),
    ('coordsys', 'RADEC'),
    ('nch', int(size)),
    ('xref', 0.0),
    ('yref', 0.0),
])


# classes
@xr.register_dataarray_accessor('fm')
class FMAccessor(object):
    def __init__(self, array):
        """Initialize the FM accessor of an array.

        Normally, this method is for the internal use.
        Users should create an array with the fm.array function.

        Args:
            array (xarray.DataArray): An array to which FM accessor is added.

        """
        self._array = array


    @property
    def isdemodulated(self):
        """A boolean that indicates whether the array is demodulated."""
        return bool(re.search('^DEMODULATED', self.status.values.item()))

    @property
    def ismodulated(self):
        """A boolean that indicates whether the array is modulated."""
        return bool(re.search('^MODULATED', self.status.values.item()))

    @property
    def tcoords(self):
        """A dictionary of arrays that label time axis."""
        return {key: getattr(self, key).values for key in TCOORDS()}

    @property
    def chcoords(self):
        """A dictionary of arrays that label channel axis."""
        return {key: getattr(self, key).values for key in CHCOORDS()}

    @property
    def ptcoords(self):
        """A dictionary of values that don't label any axes (point like)."""
        return {key: getattr(self, key).values.item() for key in PTCOORDS()}

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after an array is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(TCOORDS(self.shape[0]))
        self.coords.update(CHCOORDS(self.shape[1]))
        self.coords.update(PTCOORDS(self.shape[1]))

    def __getattr__(self, name):
        """array.fm.name <=> array.name"""
        return getattr(self._array, name)

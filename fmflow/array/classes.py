# coding: utf-8

# imported items
__all__ = []

# standard library
import re
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# constants
TCOORDS = lambda size=0: OrderedDict([
    ('fmch', ('t', np.zeros(size, dtype=int))),
    ('vrad', ('t', np.zeros(size, dtype=float))),
    ('xrel', ('t', np.zeros(size, dtype=float))),
    ('yrel', ('t', np.zeros(size, dtype=float))),
    ('time', ('t', np.tile(datetime(2000,1,1), size))),
])

CHCOORDS = lambda size=0: OrderedDict([
    ('fsig', ('ch', np.arange(size, dtype=float))),
    ('fimg', ('ch', np.arange(size, dtype=float)[::-1])),
])

PTCOORDS = OrderedDict([
    ('status', 'MODULATED'),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
])


# classes
@xr.register_dataarray_accessor('fm')
class FMAccessor(object):
    def __init__(self, array):
        """Initialize the FM accessor of an array.

        Note:
            This method is only for the internal use.
            Users can create an array with FM accessor using fm.array.

        Args:
            array (xarray.DataArray): An array to which FM accessor is added.

        """
        self._array = array

    def demodulate(self, reverse=False):
        """Create a demodulated array from the modulated one.

        This method is only available when the original array is modulated.
        It is equivalent to the fm.demodulate function (recommended to use).
        i.e. array.fm.demodulate(reverse) <=> fm.demodulate(array, reverse)

        Args:
            reverse (bool): If True, the original array is reverse-demodulated
                (i.e. -1 * fmch is used for demodulation). Default is False.

        Returns:
            array (xarray.DataArray): A demodulated array.

        """
        if self.isdemodulated:
            raise fm.utils.FMFlowError('already demodulated')

        fmch = [1, -1][reverse] * self.fmch.values
        shape = (self.shape[0], self.shape[1]+np.ptp(fmch))

        # demodulate data
        if np.ptp(fmch) != 0:
            data = np.full(shape, np.nan)
            data[:,:-np.ptp(fmch)] = self.values
            data = fm.utils.rollrows(data, fmch-np.min(fmch))
        else:
            data = self.values.copy()

        # update coords
        indx = np.arange(np.min(fmch), np.min(fmch)+shape[1])
        fsig = interp1d(self.indx, self.fsig, fill_value='extrapolate')(indx)
        fimg = interp1d(self.indx, self.fimg, fill_value='extrapolate')(indx)
        status = 'DEMODULATED' + ['+', '-'][reverse]

        tcoords  = deepcopy(self.tcoords)
        chcoords = deepcopy(self.chcoords)
        ptcoords = deepcopy(self.ptcoords)
        tcoords.update({'fmch': fmch})
        chcoords.update({'indx': indx, 'fsig': fsig, 'fimg': fimg})
        ptcoords.update({'status': status})

        return fm.array(data, tcoords, chcoords, ptcoords)


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
        return {key: getattr(self, key).item() for key in PTCOORDS}

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after an array is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(TCOORDS(self.shape[0]))
        self.coords.update(CHCOORDS(self.shape[1]))
        self.coords.update(PTCOORDS)

    def __getattr__(self, name):
        """array.fm.name <=> array.name"""
        return getattr(self._array, name)

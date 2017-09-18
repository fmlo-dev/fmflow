# coding: utf-8

# public items
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

# module constants
TCOORDS = lambda size=0: OrderedDict([
    ('fmch', ('t', np.zeros(size, dtype=int))),
    ('vrad', ('t', np.zeros(size, dtype=float))),
    ('time', ('t', np.tile(datetime(2000,1,1), size))),
    ('x',    ('t', np.zeros(size, dtype=float))),
    ('y',    ('t', np.zeros(size, dtype=float))),
])

CHCOORDS = lambda size=0: OrderedDict([
    ('fsig', ('ch', np.zeros(size, dtype=float))),
    ('fimg', ('ch', np.zeros(size, dtype=float))),
])

PTCOORDS = OrderedDict([
    ('status', 'MODULATED'),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
])


# classes
class BaseAccessor(object):
    def __init__(self, dataarray):
        """Initialize the base accessor."""
        self._dataarray = dataarray

    def __getattr__(self, name):
        """self._dataarray.name <=> self.name."""
        return getattr(self._dataarray, name)

    @property
    def isdemodulated(self):
        """Whether the array is demodulated (regardless of reverse)."""
        return bool(re.search('^DEMODULATED', str(self.status.values)))

    @property
    def isdemodulated_r(self):
        """Whether the array is reverse-demodulated."""
        return bool(re.search('^DEMODULATED-', str(self.status.values)))

    @property
    def ismodulated(self):
        """Whether the array is modulated."""
        return bool(re.search('^MODULATED', str(self.status.values)))

    @property
    def tcoords(self):
        """A dictionary of arrays that label time axis."""
        return {key: val.values for key, val in self.coords.items() if val.dims==('t',)}

    @property
    def chcoords(self):
        """A dictionary of arrays that label channel axis."""
        return {key: val.values for key, val in self.coords.items() if val.dims==('ch',)}

    @property
    def ptcoords(self):
        """A dictionary of values that don't label any axes (point-like)."""
        return {key: val.values for key, val in self.coords.items() if val.dims==()}


@xr.register_dataarray_accessor('fma')
class FMArrayAccessor(BaseAccessor):
    def __init__(self, array):
        """Initialize the FM array accessor.

        Note:
            This method is only for the internal use.
            User can create an array with this accessor using `fm.array`.

        Args:
            array (xarray.DataArray): An array to which this accessor is added.

        """
        super().__init__(array)

    def demodulate(self, reverse=False):
        """Create a demodulated array from the modulated one.

        This method is only available when the array is modulated.
        It is equivalent to the fm.demodulate function (recommended to use).
        i.e. array.fma.demodulate(reverse) <=> fm.demodulate(array, reverse)

        Args:
            reverse (bool, optional): If True, the array is reverse-demodulated
                (i.e. -1 * fmch is used for demodulation). Default is False.

        Returns:
            array (xarray.DataArray): A demodulated array.

        """
        if self.isdemodulated:
            raise FMArrayError('already demodulated')

        fmch = [1, -1][reverse] * self.fmch.values
        newshape = (self.shape[0], self.shape[1]+np.ptp(fmch))

        # demodulate data
        if np.ptp(fmch) != 0:
            data = np.full(newshape, np.nan)
            data[:,:-np.ptp(fmch)] = self.values
            data = fm.utils.rollrows(data, fmch-np.min(fmch))
        else:
            data = self.values.copy()

        # update coords
        chno = self.shape[1]
        chid = np.arange(np.min(fmch), np.min(fmch)+newshape[1])
        fsig = interp1d(np.arange(chno), self.fsig, fill_value='extrapolate')(chid)
        fimg = interp1d(np.arange(chno), self.fimg, fill_value='extrapolate')(chid)
        status = 'DEMODULATED' + ['+', '-'][reverse]

        tcoords  = deepcopy(self.tcoords)
        chcoords = deepcopy(self.chcoords)
        ptcoords = deepcopy(self.ptcoords)
        tcoords.update({'fmch': fmch})
        chcoords.update({'fsig': fsig, 'fimg': fimg, 'chid': chid})
        ptcoords.update({'status': status, 'chno': chno})

        return fm.array(data, tcoords, chcoords, ptcoords)

    def modulate(self):
        """Create a modulated array from the demodulated one.

        This method is only available when the array is demodulated.
        It is equivalent to the fm.modulate function (recommended to use).
        i.e. array.fma.modulate() <=> fm.modulate(array)

        Returns:
            array (xrray.DataArray): A modulated array.

        """
        if self.ismodulated:
            raise FMArrayError('already modulated')

        fmch = self.fmch.values.copy()
        lextch = np.max([0, np.min(self.chid.values)])
        rextch = np.max([0, np.min((self.chno-self.chid).values)])
        extshape = (self.shape[0], self.shape[1]+lextch+rextch)
        extchid = np.arange(np.min(self.chid)-lextch, np.max(self.chid)+rextch+1)
        newchid = np.arange(self.chno)

        # modulate data
        if np.ptp(fmch) != 0:
            data = np.full(extshape, np.nan)
            data[:,lextch:extshape[1]-rextch] = self.values
            data = fm.utils.rollrows(data, -fmch)
            data = data[:,np.in1d(extchid, newchid)]
        else:
            data = self.values.copy()

        # update coords
        fmch *= eval('{}1'.format(self.status.item()[-1]))
        fsig = interp1d(self.chid, self.fsig, fill_value='extrapolate')(newchid)
        fimg = interp1d(self.chid, self.fimg, fill_value='extrapolate')(newchid)
        status = 'MODULATED'

        tcoords  = deepcopy(self.tcoords)
        chcoords = deepcopy(self.chcoords)
        ptcoords = deepcopy(self.ptcoords)
        tcoords.update({'fmch': fmch})
        chcoords.update({'fsig': fsig, 'fimg': fimg})
        ptcoords.update({'status': status})
        chcoords.pop('chid')
        ptcoords.pop('chno')

        return fm.array(data, tcoords, chcoords, ptcoords)

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after an array is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(TCOORDS(self.shape[0]))
        self.coords.update(CHCOORDS(self.shape[1]))
        self.coords.update(PTCOORDS)


class FMArrayError(Exception):
    """Error class of FM array."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
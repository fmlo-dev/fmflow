# coding: utf-8

# public items
__all__ = []

# standard library
import re
from collections import OrderedDict
from copy import deepcopy

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr
from ..classes import BaseAccessor

# module constants
CHCOORDS = lambda size=0: OrderedDict([
    ('chid', ('ch', np.arange(size, dtype=int))),
    ('freq', ('ch', np.zeros(size, dtype=float))),
    ('noise', ('ch', np.zeros(size, dtype=float))),
])

PTCOORDS = lambda size=0: OrderedDict([
    ('status', 'DEMODULATED+'),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('chno', size),
])


# classes
@xr.register_dataarray_accessor('fms')
class FMSpectrumAccessor(BaseAccessor):
    def __init__(self, spectrum):
        """Initialize the FM spectrum accessor.

        Note:
            This method is only for the internal use.
            User can create an array with this accessor using `fm.spectrum`.

        Args:
            spectrum (xarray.DataArray): An array to which this accessor is added.

        """
        super().__init__(spectrum)

    def fromarray(self, array, weights=None, reverse=False):
        """Create a spectrum from an array.

        Args:
            array (xarray.DataArray): An array.
            weights (xarray.DataArray, optional): A weight array.
            reverse (bool, optional): If True, the array is reverse-demodulated
                (i.e. -1 * fmch is used for demodulation). Default is False.

        Returns:
            spectrum (xarray.DataArray): A spectrum.

        """
        if weights is None:
            weights = fm.ones_like(array)

        if array.fma.ismodulated:
            array = fm.demodulate(array, reverse)

        if weights.fma.ismodulated:
            weights = fm.demodulate(weights, reverse)

        # data (spectrum by weighted mean)
        data = (weights*array).sum('t') / weights.sum('t')

        # noise (by weighted rms)
        noise = np.sqrt((weights*array**2).sum('t') / weights.sum('t'))
        noise /= np.sqrt((~np.isnan(array)).sum('t'))

        # freq
        if reverse:
            freq = array.fimg
        else:
            freq = array.fsig

        # coords
        chcoords = deepcopy(array.fma.chcoords)
        ptcoords = deepcopy(array.fma.ptcoords)
        chcoords.update({'freq': freq.values, 'noise': noise.values})

        return fm.spectrum(data.values, chcoords, ptcoords)

    def toarray(self, array):
        """Create an array filled with the spectrum.

        Args:
            spectrum (xarray.DataArray): A spectrum to be cast.
            array (xarray.DataArray): An array whose shape the spectrum is cast on.

        Returns:
            array (xarray.DataArray): An array filled with the spectrum.

        """
        array = fm.zeros_like(array)

        if array.fma.ismodulated:
            if self.isdemodulated_r:
                array = fm.demodulate(array, True)
            else:
                array = fm.demodulate(array, False)

        # check compatibility
        if not np.all(self.chid == array.chid):
            raise FMSpectrumError('cannot cast the spectrum on the array')

        if not self.chno == array.chno:
            raise FMSpectrumError('cannot cast the spectrum to the array')

        return fm.modulate(array + self.values)

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after a spectrum is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(CHCOORDS(self.shape[0]))
        self.coords.update(PTCOORDS(self.shape[0]))


class FMSpectrumError(Exception):
    """Error class of FM spectrum."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

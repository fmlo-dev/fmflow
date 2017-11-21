# coding: utf-8

# public items
__all__ = []

# standard library
from collections import OrderedDict
from copy import deepcopy

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr
from .. import BaseAccessor

# module constants
CHCOORDS = lambda array: OrderedDict([
    ('chid',  ('ch', np.zeros(array.shape[0], dtype=int))),
    ('freq',  ('ch', np.zeros(array.shape[0], dtype=float))),
    ('noise', ('ch', np.zeros(array.shape[0], dtype=float))),
])

SCALARCOORDS = OrderedDict([
    ('type', BaseAccessor.FMSPECTRUM),
    ('status', BaseAccessor.DEMODULATED),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('chno', 0),
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

    @classmethod
    def fromarray(cls, array, weights=None, reverse=False):
        """Create a spectrum from the array.

        Args:
            array (xarray.DataArray): An array.
            weights (xarray.DataArray, optional): A weight array.
            reverse (bool, optional): If True, the array is reverse-demodulated
                (i.e. -1 * fmch is used for demodulation). Default is False.

        Returns:
            spectrum (xarray.DataArray): A spectrum.

        """
        array = array.copy()

        if weights is None:
            weights = fm.ones_like(array)

        if array.fma.ismodulated:
            array   = fm.demodulate(array, reverse)
            weights = fm.demodulate(weights, reverse)
        else:
            weights.values[np.isnan(array.values)] = np.nan

        with fm.utils.ignore_numpy_errors():
            # weighted mean and square mean
            mean1 = (weights*array).sum('t') / weights.sum('t')
            mean2 = (weights*array**2).sum('t') / weights.sum('t')
            sum_n = (~np.isnan(weights)).sum('t')

            # noise (weighted std)
            noise = ((mean2-mean1**2) / sum_n)**0.5
            noise.values[sum_n.values<=2] = np.inf # edge treatment

        # coords
        freq = array.fimg if reverse else array.fsig

        chcoords = deepcopy(array.fma.chcoords)
        scalarcoords = deepcopy(array.fma.scalarcoords)
        chcoords.update({'freq': freq.values, 'noise': noise.values})

        return fm.spectrum(mean1.values, chcoords, scalarcoords)

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
            ismodulated = True
            if self.isdemodulated_r:
                array = fm.demodulate(array, True)
            else:
                array = fm.demodulate(array, False)
        else:
            ismodulated = False

        # check compatibility
        if not np.all(self.chid == array.chid):
            raise FMSpectrumError('cannot cast the spectrum on the array')

        if not self.chno == array.chno:
            raise FMSpectrumError('cannot cast the spectrum to the array')

        if ismodulated:
            return fm.modulate(array + self.values)
        else:
            return array

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after a spectrum is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(CHCOORDS(self))
        self.coords.update(SCALARCOORDS)


class FMSpectrumError(Exception):
    """Error class of FM spectrum."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

# coding: utf-8

# standard library
import re as _re


# base accessor
class BaseAccessor(object):
    FMARRAY    = 'FMARRAY'
    FMCUBE     = 'FMCUBE'
    FMSPECTRUM = 'FMSPECTRUM'
    MODULATED     = 'MODULATED'
    DEMODULATED   = 'DEMODULATED'
    DEMODULATED_R = 'DEMODULATED_R'

    def __init__(self, dataarray):
        """Initialize the base accessor."""
        self._dataarray = dataarray

    def dimcoords(self, dim):
        """A dictionary of values that label `dim` axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==(dim,)}

    def updatecoords(self, coords, dim=None):
        if dim is None:
            self.coords.update(coords)
        else:
            self.coords.update({key: (dim, val) for key, val in coords.items()})

    @property
    def scalarcoords(self):
        """A dictionary of values that don't label any axes."""
        return {k: v.values for k, v in self.coords.items() if v.dims==()}

    @property
    def datacoords(self):
        """A dictionary of arrays that label full axes of the data."""
        return {k: v.values for k, v in self.coords.items() if v.dims==self.dims}

    @property
    def isdemodulated(self):
        """Whether the array is demodulated (regardless of reverse)."""
        return bool(_re.search('^'+self.DEMODULATED, str(self.status.values)))

    @property
    def isdemodulated_r(self):
        """Whether the array is reverse-demodulated."""
        return bool(_re.search('^'+self.DEMODULATED_R, str(self.status.values)))

    @property
    def ismodulated(self):
        """Whether the array is modulated."""
        return bool(_re.search('^'+self.MODULATED, str(self.status.values)))

    def __getattr__(self, name):
        """Return self.`dim`coords or convert self.name to self._dataarray.name."""
        if _re.search('.+coords$', name):
            return self.dimcoords(name.rstrip('coords'))
        else:
            return getattr(self._dataarray, name)

    def __setstate__(self, state):
        """A method used for pickling."""
        self.__dict__ = state

    def __getstate__(self):
        """A method used for unpickling."""
        return self.__dict__


# submodules
from .array.classes import *
from .array.decorators import *
from .array.functions import *
from .cube.classes import *
from .cube.functions import *
from .spectrum.classes import *
from .spectrum.functions import *

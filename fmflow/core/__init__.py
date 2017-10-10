# coding: utf-8

# base accessor
class BaseAccessor(object):
    def __init__(self, dataarray):
        """Initialize the base accessor."""
        self._dataarray = dataarray

    def dimcoords(self, dim):
        """A dictionary of values that label `dim` axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==(dim,)}

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
        return bool(re.search('^DEMODULATED', str(self.status.values)))

    @property
    def isdemodulated_r(self):
        """Whether the array is reverse-demodulated."""
        return bool(re.search('^DEMODULATED-', str(self.status.values)))

    @property
    def ismodulated(self):
        """Whether the array is modulated."""
        return bool(re.search('^MODULATED', str(self.status.values)))

    def __getattr__(self, name):
        """Return self.`dim`coords or convert self.name to self._dataarray.name."""
        import re
        if re.search('.+coords$', name):
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
from .classes import *
from .decorators import *
from .functions import *
from .cube.classes import *
from .cube.functions import *
from .spectrum.classes import *
from .spectrum.functions import *
del classes
del decorators
del functions

# coding: utf-8

# base accessor
class BaseAccessor(object):
    def __init__(self, dataarray):
        """Initialize the base accessor."""
        self._dataarray = dataarray

    def __getattr__(self, name):
        """self._dataarray.name <=> self.name."""
        return getattr(self._dataarray, name)

    def __setstate__(self, state):
        """A method used for pickling."""
        self.__dict__ = state

    def __getstate__(self):
        """A method used for unpickling."""
        return self.__dict__

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
        return {k: v.values for k, v in self.coords.items() if v.dims==('t',)}

    @property
    def chcoords(self):
        """A dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('ch',)}

    @property
    def xcoords(self):
        """A dictionary of arrays that label x axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('x',)}

    @property
    def ycoords(self):
        """A dictionary of arrays that label y axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('y',)}

    @property
    def scalarcoords(self):
        """A dictionary of values that don't label any axes."""
        return {k: v.values for k, v in self.coords.items() if v.dims==()}

    @property
    def datacoords(self):
        """A dictionary of arrays that label full axes of the data."""
        return {k: v.values for k, v in self.coords.items() if v.dims==self.dims}


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

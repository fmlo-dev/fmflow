# coding: utf-8

# public items
__all__ = []

# standard library
import os
from collections import OrderedDict
from copy import deepcopy
from itertools import product

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr
from .. import BaseAccessor
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import map_coordinates
from scipy.special import j1

# module constants
XCOORDS = lambda array: OrderedDict([
    ('x', ('x', np.zeros(array.shape[0], dtype=float))),
])

YCOORDS = lambda array: OrderedDict([
    ('y', ('y', np.zeros(array.shape[1], dtype=float))),
])

CHCOORDS = lambda array: OrderedDict([
    ('chid', ('ch', np.zeros(array.shape[2], dtype=int))),
    ('freq', ('ch', np.zeros(array.shape[2], dtype=float))),
])

CUBECOORDS = lambda array: OrderedDict([
    ('noise', (('x', 'y', 'ch'), np.zeros(array.shape, dtype=float))),
])

PTCOORDS = OrderedDict([
    ('status', 'DEMODULATED+'),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('chno', 0),
])

DIST_MAX = 3
N_CHUNKS = os.cpu_count() - 1


# classes
@xr.register_dataarray_accessor('fmc')
class FMCubeAccessor(BaseAccessor):
    def __init__(self, cube):
        """Initialize the FM cube accessor.

        Note:
            This method is only for the internal use.
            User can create an array with this accessor using `fm.cube`.

        Args:
            cube (xarray.DataArray): An array to which this accessor is added.

        """
        super().__init__(cube)

    def fromarray(self, weights=None, reverse=False, gridsize=10.0/3600):
        """Create a cube from an array.

        Args:
            weights (xarray.DataArray, optional): A weight array.
            reverse (bool, optional): If True, the array is reverse-demodulated
                (i.e. -1 * fmch is used for demodulation). Default is False.

        Returns:
            cube (xarray.DataArray): A cube.

        """
        array = self._dataarray.copy()

        if weights is None:
            weights = fm.ones_like(array)

        if array.fma.ismodulated:
            array = fm.demodulate(array, reverse)

        if weights.fma.ismodulated:
            weights = fm.demodulate(weights, reverse)

        array1 = weights * array
        array2 = weights * array**2

        # set gridparams
        self.set_gridparams(array, gridsize)

        # calc gridding to make cube, noise
        @fm.chunk('i', concatfunc=self.concatfunc)
        def gridding(i):
            sgcvs  = np.array_split(cls.gcvs, N_CHUNKS)[int(i)]
            smasks = (sgcvs > 0)

            shape = (*sgcvs.shape[:2], array.shape[1])
            cube  = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))
            noise = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))

            with fm.utils.ignore_numpy_errors():
                for i, j in product(*map(range, shape[:2])):
                    gcv, mask = sgcvs[i, j], smasks[i, j]
                    mgcv, mweights = gcv[mask], weights[mask]
                    marray1, marray2 = array1[mask], array2[mask]

                    mean1 = (mgcv*marray1).sum('t') / mweights.sum('t')
                    mean2 = (mgcv*marray2).sum('t') / mweights.sum('t')
                    num   = (~np.isnan(marray1)).sum('t')

                    cube[i, j]  = mean1
                    noise[i, j] = ((mean2-mean1**2)/num)**0.5

            return cube, noise

        cube, noise = mapping(np.arange(N_CHUNKS), numchunk=N_CHUNKS)

        # freq
        if array.fma.isdemodulated_r:
            freq = array.fimg
        else:
            freq = array.fsig

    def toarray(self, array):
        """Create an array filled with the cube.

        Args:
            cube (xarray.DataArray): A cube to be cast.
            array (xarray.DataArray): An array whose shape the cube is cast on.

        Returns:
            array (xarray.DataArray): An array filled with the cube.

        """
        array = fm.zeros_like(array)

    @classmethod
    def set_gridparams(cls, array, gridsize=10/3600):
        # x, y (and their references) of the array
        cls.x  = xr.DataArray(array.x.values, dims='t')
        cls.y  = xr.DataArray(array.y.values, dims='t')
        cls.x0 = array.xref.values
        cls.y0 = array.yref.values

        # x, y of the grid
        gxrel_min = gridsize * np.floor((cls.x-cls.x0).min() / gridsize)
        gyrel_min = gridsize * np.floor((cls.y-cls.y0).min() / gridsize)
        gxrel_max = gridsize * np.ceil((cls.x-cls.x0).max() / gridsize)
        gyrel_max = gridsize * np.ceil((cls.y-cls.y0).max() / gridsize)
        cls.gx = x0 + np.arange(gxrel_min, gxrel_max+gridsize, gridsize)
        cls.gy = y0 + np.arange(gyrel_min, gyrel_max+gridsize, gridsize)

        # grid distances between grid (x,y)s and each (x,y) of array
        # as an xarray.DataArray whose dims = ('x', 'y', 't')
        mgx = xr.DataArray(np.meshgrid(cls.gx, cls.gy)[0].T, dims=('x', 'y'))
        mgy = xr.DataArray(np.meshgrid(cls.gx, cls.gy)[1].T, dims=('x', 'y'))
        cls.dist = np.sqrt((mgx-x)**2+(mgy-y)**2) / gridsize

        # grid convolution values
        cls.gcvs = self.gcf_besselgauss(cls.dist)
        cls.gcvs.values[cls.dist>DIST_MAX] = 0.0

    @staticmethod
    def gcf_besselgauss(r, a=1.55, b=2.52):
        with fm.utils.ignore_numpy_errors():
            gcf = 2*j1(np.pi*r/a)/(np.pi*r/a) * np.exp(-(r/b)**2)

        return np.select([r!=0], [gcf], 1)

    @staticmethod
    def gcf_sincgauss(r, a=1.55, b=2.52):
        return np.sinc(r/a) * np.exp(-(r/b)**2)

    @staticmethod
    def gcf_gauss(r, a=1.00):
        return np.exp(-(r/a)**2)

    @staticmethod
    def concatfunc(gridding_results):
        cube  = xr.concat([r[0] for r in gridding_results], dim='x')
        noise = xr.concat([r[1] for r in gridding_results], dim='x')
        return cube, noise

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after a spectrum is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(XCOORDS(self))
        self.coords.update(YCOORDS(self))
        self.coords.update(CHCOORDS(self))
        self.coords.update(CUBECOORDS(self))
        self.coords.update(PTSCOORDS)


class FMCubeError(Exception):
    """Error class of FM cube."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

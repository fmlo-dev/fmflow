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

DATACOORDS = lambda array: OrderedDict([
    ('noise', (('x', 'y', 'ch'), np.zeros(array.shape, dtype=float))),
])

SCALARCOORDS = OrderedDict([
    ('status', 'DEMODULATED+'),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('chno', 0),
])

GRIDDIST_MAX = 3
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

    @classmethod
    def fromarray(cls, array, weights=None, reverse=False, gridsize=10.0/3600):
        """Create a cube from an array.

        Args:
            array (xarray.DataArray): An array.
            weights (xarray.DataArray, optional): A weight array.
            reverse (bool, optional): If True, the array is reverse-demodulated
                (i.e. -1 * fmch is used for demodulation). Default is False.
            gridsize (float, optional): Grid size in units of degree.

        Returns:
            cube (xarray.DataArray): A cube.

        """
        array = array.copy()

        if weights is None:
            weights = fm.ones_like(array)

        if array.fma.ismodulated:
            array = fm.demodulate(array, reverse)

        if weights.fma.ismodulated:
            weights = fm.demodulate(weights, reverse)

        array1 = weights * array
        array2 = weights * array**2

        # set gridparams
        cls.set_gridparams(array, gridsize)

        # make cube, noise
        gcv = cls.gcv
        shape = (*gcv.shape[:2], array.shape[1])

        sum_a1 = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))
        sum_a2 = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))
        sum_w  = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))
        sum_n  = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))

        with fm.utils.ignore_numpy_errors():
            for i, j in product(*map(range, shape[:2])):
                gcv_ij = gcv[i, j]
                mask   = (gcv_ij > 0)
                mgcv, mweights = gcv_ij[mask], weights[mask]
                marray1, marray2 = array1[mask], array2[mask]

                sum_a1[i, j] = (mgcv*marray1).sum('t')
                sum_a2[i, j] = (mgcv*marray2).sum('t')
                sum_w[i, j]  = mweights.sum('t')
                sum_n[i, j]  = (~np.isnan(mweights)).sum('t')

            # weighted mean and square mean
            mean1 = sum_a1 / sum_w
            mean2 = sum_a2 / sum_w

            # noise (weighted std)
            noise = ((mean2-mean1**2) / sum_n)**0.5
            noise[sum_n<=2] = np.inf # edge treatment

        # coords
        freq = array.fimg if reverse else array.fsig

        xcoords = {'x': cls.gx}
        ycoords = {'y': cls.gy}
        chcoords = deepcopy(array.fma.chcoords)
        chcoords.update({'freq': freq.values})
        datacoords = {'noise': noise.values}
        scalarcoords = deepcopy(array.fma.scalarcoords)

        return fm.cube(mean1.values, xcoords, ycoords,
                       chcoords, datacoords, scalarcoords)

    def toarray(self, array):
        """Create an array filled with the cube.

        Args:
            cube (xarray.DataArray): A cube to be cast.
            array (xarray.DataArray): An array whose shape the cube is cast on.

        Returns:
            array (xarray.DataArray): An array filled with the cube.

        """
        pass

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
        cls.gx = cls.x0 + np.arange(gxrel_min, gxrel_max+gridsize, gridsize)
        cls.gy = cls.y0 + np.arange(gyrel_min, gyrel_max+gridsize, gridsize)

        # grid distances between grid (x,y)s and each (x,y) of array
        # as an xarray.DataArray whose dims = ('x', 'y', 't')
        mgx = xr.DataArray(np.meshgrid(cls.gx, cls.gy)[0].T, dims=('x', 'y'))
        mgy = xr.DataArray(np.meshgrid(cls.gx, cls.gy)[1].T, dims=('x', 'y'))
        cls.gdist = np.sqrt((mgx-cls.x)**2+(mgy-cls.y)**2) / gridsize

        # grid convolution values
        cls.gcv = fm.xarrayfunc(cls.gcf_besselgauss)(cls.gdist)
        cls.gcv.values[cls.gdist>GRIDDIST_MAX] = 0.0

    @staticmethod
    def gcf_besselgauss(r, a=1.55, b=2.52):
        """Grid convolution function of Bessel-Gaussian."""
        with fm.utils.ignore_numpy_errors():
            gcf = 2*j1(np.pi*r/a)/(np.pi*r/a) * np.exp(-(r/b)**2)

        return np.select([r!=0], [gcf], 1)

    @staticmethod
    def gcf_sincgauss(r, a=1.55, b=2.52):
        """Grid convolution function of Sinc-Gaussian."""
        return np.sinc(r/a) * np.exp(-(r/b)**2)

    @staticmethod
    def gcf_gauss(r, a=1.00):
        """Grid convolution function of Gaussian."""
        return np.exp(-(r/a)**2)

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after a spectrum is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(XCOORDS(self))
        self.coords.update(YCOORDS(self))
        self.coords.update(CHCOORDS(self))
        self.coords.update(DATACOORDS(self))
        self.coords.update(SCALARCOORDS)


class FMCubeError(Exception):
    """Error class of FM cube."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

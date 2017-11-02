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
from numba import jit
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
    ('type', BaseAccessor.FMCUBE),
    ('status', BaseAccessor.DEMODULATED),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('chno', 0),
])

GDIST_MAX = 3


# classes
@xr.register_dataarray_accessor('fmc')
class FMCubeAccessor(BaseAccessor):
    kernel = None

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
    def fromarray(cls, array, weights=None, reverse=False, gridsize=10.0/3600,
                  gcf='besselgauss', reuse_kernel=True):
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

        warray1 = weights * array
        warray2 = weights * array**2

        # make cube, noise
        cls.setgrid(array, gridsize, gcf, reuse_kernel)
        shape = (*cls.kernel.shape[:2], array.shape[1])

        sum_a1 = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))
        sum_a2 = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))
        sum_w  = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))
        sum_n  = xr.DataArray(np.empty(shape), dims=('x', 'y', 'ch'))

        with fm.utils.ignore_numpy_errors():
            for x, y in product(*map(range, shape[:2])):
                k_ij = cls.kernel[x, y]
                mask = (k_ij > 0)
                mkernel   = k_ij[mask]
                mwarray1  = warray1[mask]
                mwarray2  = warray2[mask]
                mweights  = weights[mask]

                sum_a1[x, y] = (mkernel * mwarray1).sum('t')
                sum_a2[x, y] = (mkernel * mwarray2).sum('t')
                sum_w[x, y]  = (mkernel * mweights).sum('t')
                sum_n[x, y]  = (~np.isnan(mweights)).sum('t')

            # weighted mean and square mean
            mean1 = sum_a1 / sum_w
            mean2 = sum_a2 / sum_w

            # noise (weighted std)
            noise = ((mean2-mean1**2) / sum_n)**0.5
            noise.values[sum_n.values<=2] = np.inf # edge treatment

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
            raise FMCubeError('cannot cast the cube on the array')

        if not self.chno == array.chno:
            raise FMCubeError('cannot cast the cube to the array')

        x, y = array.x, array.y
        gx, gy = self.x, self.y
        ix = interp1d(gx, np.arange(len(gx)))(x)
        iy = interp1d(gy, np.arange(len(gy)))(y)
        for ch in range(len(self.ch)):
            array[:, ch] = map_coordinates(self.values[:, :, ch], (ix, iy))

        if ismodulated:
            return fm.modulate(array)
        else:
            return array


    @classmethod
    def setgrid(cls, array, gridsize=10/3600, gcf='besselgauss', reuse_kernel=True):
        if cls.kernel is None or not reuse_kernel:
            # x, y (and their references) of the array
            x  = xr.DataArray(array.x.values, dims='t')
            y  = xr.DataArray(array.y.values, dims='t')
            x0 = array.xref.values
            y0 = array.yref.values

            # x, y of the grid
            gxrel_min = gridsize * np.floor((x-x0).min() / gridsize)
            gyrel_min = gridsize * np.floor((y-y0).min() / gridsize)
            gxrel_max = gridsize * np.ceil((x-x0).max() / gridsize)
            gyrel_max = gridsize * np.ceil((y-y0).max() / gridsize)
            cls.gx = x0 + np.arange(gxrel_min, gxrel_max+gridsize, gridsize)
            cls.gy = y0 + np.arange(gyrel_min, gyrel_max+gridsize, gridsize)

            # grid distances between grid (x,y)s and each (x,y) of array
            # as an xarray.DataArray whose dims = ('x', 'y', 't')
            mgx, mgy = np.meshgrid(cls.gx, cls.gy)
            mgx = xr.DataArray(mgx.T, dims=('x', 'y'))
            mgy = xr.DataArray(mgy.T, dims=('x', 'y'))
            gdist = np.sqrt((mgx-x)**2+(mgy-y)**2) / gridsize

            # grid convolution kernel
            gcf = getattr(cls, 'gcf_{}'.format(gcf))
            cls.kernel = fm.xarrayfunc(gcf)(gdist)
            cls.kernel.values[gdist.values>GDIST_MAX] = 0

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

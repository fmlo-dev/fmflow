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
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import map_coordinates
from scipy.special import j1

# module constants
CHCOORDS = lambda array: OrderedDict([
    ('chid', ('ch', np.zeros(array.shape[0], dtype=int))),
    ('freq', ('ch', np.zeros(array.shape[0], dtype=float))),
])

YCOORDS = lambda array: OrderedDict([
    ('y', ('y', np.zeros(array.shape[1], dtype=float))),
])

XCOORDS = lambda array: OrderedDict([
    ('x', ('x', np.zeros(array.shape[2], dtype=float))),
])

DATACOORDS = lambda array: OrderedDict([
    ('noise', (('ch', 'y', 'x'), np.zeros(array.shape, dtype=float))),
])

SCALARCOORDS = OrderedDict([
    ('type', BaseAccessor.FMCUBE),
    ('status', BaseAccessor.DEMODULATED),
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('chno', 0),
])

R_MAX = 3


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
    def fromarray(cls, array, weights=None, reverse=False, gridsize=10,
                  gridunit='arcsec', gcf='besselgauss'):
        """Create a cube from an array.

        Args:
            array (xarray.DataArray): An array.
            weights (xarray.DataArray, optional): A weight array.
            reverse (bool, optional): If True, the array is reverse-demodulated
                (i.e. -1 * fmch is used for demodulation). Default is False.
            gridsize (float, optional): Grid size in units of `gridunit`.
            gridunit (str, optional): Grid unit of `gridsize`.
            gcf (str, optional): Grid convolution function.

        Returns:
            cube (xarray.DataArray): A cube.

        """
        array = array.copy()

        if weights is None:
            weights = fm.ones_like(array)

        if array.fma.ismodulated:
            array   = fm.demodulate(array, reverse)
            weights = fm.demodulate(weights, reverse)
        else:
            weights.values[np.isnan(array.values)] = np.nan

        # grid parameters
        gridparams = cls.gridparams(array, gridsize, gridunit)

        # make cubes
        array_n = np.ones_like(array.values)
        array_w = np.copy(weights.values)
        array_1 = array_w * array.values
        array_2 = array_1 * array.values

        isnan = np.isnan(array)
        array_n[isnan] = 0
        array_w[isnan] = 0
        array_1[isnan] = 0
        array_2[isnan] = 0

        gcf = getattr(cls, 'gcf_{}'.format(gcf))
        cube_n = cls.convolve(array_n, gcf, **gridparams)
        cube_w = cls.convolve(array_w, gcf, **gridparams)
        cube_1 = cls.convolve(array_1, gcf, **gridparams)
        cube_2 = cls.convolve(array_2, gcf, **gridparams)

        with fm.utils.ignore_numpy_errors():
            # weighted mean and squared mean
            mean_1 = cube_1 / cube_w
            mean_2 = cube_2 / cube_w

            # noise (weighted std)
            noise = np.sqrt((mean_2-mean_1**2) / cube_n)
            noise[cube_n<=2] = np.inf # edge treatment

        # coords
        freq = array.fimg if reverse else array.fsig

        chcoords = deepcopy(array.fma.chcoords)
        chcoords.update({'freq': freq.values})
        ycoords = {'y': gridparams['gy']}
        xcoords = {'x': gridparams['gx']}
        datacoords = {'noise': noise}
        scalarcoords = deepcopy(array.fma.scalarcoords)

        return fm.cube(mean_1, chcoords, ycoords, xcoords,
                       datacoords, scalarcoords)

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

        y, x = array.y.values, array.x.values
        gy, gx = self.y.values, self.x.values
        iy = interp1d(gy, np.arange(len(gy)))(y)
        ix = interp1d(gx, np.arange(len(gx)))(x)

        for ch in range(len(self.ch)):
            array[:, ch] = map_coordinates(self.values[ch], (iy, ix))

        if ismodulated:
            return fm.modulate(array)
        else:
            return array

    @staticmethod
    def gridparams(array, gridsize=10, gridunit='arcsec'):
        """Get grid parameters from an array."""
        # y, x (and their references) of the array
        y, x = array.y.values, array.x.values
        y0, x0 = array.yref.values, array.xref.values

        # parse gridsize (if any)
        if isinstance(gridsize, str):
            gridsize, gridunit = str(u.Unit(gridsize)).split()

        # x, y of the grid
        gs = float(gridsize) * getattr(u, gridunit).to('degree')
        gyrel_min = gs * np.floor((y-y0).min() / gs)
        gxrel_min = gs * np.floor((x-x0).min() / gs)
        gyrel_max = gs * np.ceil((y-y0).max() / gs)
        gxrel_max = gs * np.ceil((x-x0).max() / gs)
        gy = y0 + np.arange(gyrel_min, gyrel_max+gs, gs)
        gx = x0 + np.arange(gxrel_min, gxrel_max+gs, gs)
        mgy, mgx = np.meshgrid(gy, gx, indexing='ij')

        # slices
        iy_0 = np.searchsorted(gy, y)
        ix_0 = np.searchsorted(gx, x)
        iy_min = np.maximum(iy_0-R_MAX, 0)
        ix_min = np.maximum(ix_0-R_MAX, 0)
        iy_max = np.minimum(iy_0+R_MAX+1, len(gy))
        ix_max = np.minimum(ix_0+R_MAX+1, len(gx))
        sy = np.array([slice(*p, 1) for p in zip(iy_min, iy_max)])
        sx = np.array([slice(*p, 1) for p in zip(ix_min, ix_max)])

        return {'y': y, 'x': x, 'sy': sy, 'sx': sx,
                'gy': gy, 'gx': gx, 'gs': gs, 'mgy': mgy, 'mgx': mgx}

    @staticmethod
    def convolve(array, gcf, y, x, sy, sx, gy, gx, gs, mgy, mgx):
        """Convolve NumPy array to cube according to grid parameters."""
        cube = np.zeros((len(gy), len(gx), array.shape[1]))

        for i in range(len(array)):
            syx = sy[i], sx[i]
            r  = (mgy[syx] - y[i])**2
            r += (mgx[syx] - x[i])**2
            r **= 0.5
            r /= gs

            cube[syx] += np.multiply.outer(gcf(r), array[i])

        return cube.transpose(2, 0, 1)

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

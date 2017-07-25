# coding: utf-8

# public items
__all__ = [
    'fmgf',
    'gaussian',
    'lorentzian',
    'mad',
    'pseudovoigt',
    'rollrows',
    'slicewhere',
]

# standard library
from functools import partial

# dependent packages
import numpy as np
from scipy import ndimage


# functions
def fmgf(array, sigma):
    """Apply the FMGF (Fast M-estimation based Gaussian Filter) to a 1D array.

    Args:
        array (numpy.ndarray): A 1D array.
        sigma (int): Standard deviation for Gaussian kernel.

    Returns:
        filtered (numpy.ndarray): A 1D array to which the FMGF is applied.

    References:
        Journal of the Japan Society for Precision Engineering Vol.76 (2010) No.6 P684-688
        A Proposal of Robust Gaussian Filter by Using Fast M-Estimation Method
        http://doi.org/10.2493/jjspe.76.684

    """
    x, y = np.arange(len(array)), array.copy()
    yg = ndimage.filters.gaussian_filter(y, sigma)
    y -= yg

    # digitizing
    m = 101
    dy = 6.0*mad(y) / m
    ybin = np.arange(np.min(y)-5*dy, np.max(y)+5*dy+dy, dy)
    z = np.zeros([len(ybin), len(x)])
    z[np.digitize(y, ybin), x] = 1.0

    # filtering
    g = partial(ndimage.filters.gaussian_filter, sigma=(0,sigma))
    c = partial(ndimage.filters.convolve1d, weights=np.ones(m), axis=0)
    zf = c(c(c(g(z))))

    # estimates
    ym1, y0, yp1 = [ybin[np.argmax(zf,0)+i] for i in (-1,0,1)]
    zm1, z0, zp1 = [zf[np.argmax(zf,0)+i, x] for i in (-1,0,1)]
    t = (zm1-z0) / (zm1-2*z0+zp1)

    filtered = yg + ((1-t)**2)*ym1 + (2*t*(1-t))*y0 + (t**2)*yp1
    return filtered


def gaussian(x, x0=0.0, fwhm=1.0, ampl=1.0):
    """Return Gaussian with given parameters.

    y = ampl * exp(-4ln2 * (x-x0)^2 / fwhm^2)

    Args:
        x (numpy.ndarray): An input 1D array.
        x0 (float): Center of the Gaussian.
        fwhm (float): FWHM of the Gaussian.
        ampl (float): Amplitude of the Gaussian.

    Returns:
        y (numpy.ndarray): An output 1D Gaussian.

    """
    return ampl * np.exp(-4*np.log(2)*((x-x0)/fwhm)**2)


def lorentzian(x, x0=0.0, fwhm=1.0, ampl=1.0):
    """Return Lorentzian with given parameters.

    y = ampl / (1 + 4(x-x0)^2 / fwhm^2)

    Args:
        x (numpy.ndarray): An input 1D array.
        x0 (float): Center of the Lorentzian.
        fwhm (float): FWHM of the Lorentzian.
        ampl (float): Amplitude of the Lorentzian.

    Returns:
        y (numpy.ndarray): An output 1D Lorentzian.

    """
    return ampl * (1+4*((x-x0)/fwhm)**2)**(-1)


def mad(array, axis=None, keepdims=False):
    """Compute the median absolute deviation (MAD) along the given axis.

    Args:
        array (numpy.ndarray): An input array.
        axis (int, optional): Axis along which the MADs are computed.
            The default is to compute the MAD along a flattened version of the array.
        keepdims (bool, optional): If True, the axes which are reduced are left
            in the result as dimensions with size one.

    Returns:
        mad (numpy.ndarray): A new array holding the result.

    """
    ad = np.abs(array - np.median(array, axis, keepdims=True))
    mad = np.median(ad, axis, keepdims=keepdims)
    return mad


def pseudovoigt(x, x0=0.0, fwhm=1.0, ampl=1.0, frac=0.5):
    """Return Pseudo-Voigt with given parameters.

    y = frac * gaussian(x, x0, fwhm) + (1-frac) * lorentzian(x, x0, fwhm)

    Args:
        x (numpy.ndarray): An input 1D array.
        x0 (float): Center of the Pseudo-Voigt.
        fwhm (float): FWHM of the Pseudo-Voigt.
        ampl (float): Amplitude of the Pseudo-Voigt.
        frac (float): Fraction of Gaussian and Lorentzian. Must be 0 <= frac <= 1.

    Returns:
        y (numpy.ndarray): An output 1D Pseudo-Voigt.

    """
    if not 0 <= frac <= 1:
        raise ValueError('frac must be 0 <= frac <= 1')

    return frac*gaussian(x, x0, fwhm, ampl) + (1-frac)*lorentzian(x, x0, fwhm, ampl)


def rollrows(array, shifts):
    """Roll 2D array elements of each row by a given shifts.

    Args:
        array (numpy.ndarray): A 2D array.
        shifts (int of list of int): The number(s) of places
            by which elements of each row are shifted.

    Returns:
        array (numpy.ndarray): An output rolled array.

    """
    array = np.asarray(array)
    shifts = np.asarray(shifts)

    if array.ndim < 2:
        return np.roll(array, shifts)

    if shifts.ndim < 1:
        shifts = np.tile(shifts, array.shape[0])

    rows, cols = np.ogrid[:array.shape[0], :array.shape[1]]
    cols = cols - (shifts%array.shape[1])[:,np.newaxis]
    cols[cols<0] += array.shape[1]
    return array[rows, cols]


def slicewhere(condition):
    """Return slices of regions that fulfill condition.

    Example:
        >>> cond = [False, True, True, False, False, True, False]
        >>> fm.utils.slicewhere(cond)
        [slice(1L, 3L, None), slice(5L, 6L, None)]

    Args:
        condition (numpy.ndarray): An array of booleans.

    Returns:
        slices (list of slice): List of slice objects.

    """
    regions = ndimage.find_objects(ndimage.label(condition)[0])
    return [region[0] for region in regions]

# coding: utf-8

# public items
__all__ = [
    'ctype_to_tform',
    'dtype_to_tform',
]

# standard library
import re

# dependent packages
import numpy as np


# functions
def ctype_to_tform(ctype, shape=None):
    """Convert Python C-type to FITS format.

    Args:
        ctype (str): A C-type format string in Python.
        shape (int or tuple of int, optional): shape of the C-type.

    Returns:
        tform (str): A format string of FITS (TFORM).

    References
        https://docs.python.org/3.6/library/struct.html
        http://docs.astropy.org/en/stable/io/fits/usage/table.html

    """
    count = ''
    if shape is not None:
        prod = np.prod(shape)
        if prod != 1:
            count = str(prod)

    # 32-bit integer
    if re.search('i', ctype):
        return count + 'J'
    # 64-bit integer
    elif re.search('q', ctype):
        return count + 'K'
    # single precision floating point
    elif re.search('f', ctype):
        return count + 'E'
    # double precision floating point
    elif re.search('d', ctype):
        return count + 'D'
    # unsigned byte
    elif re.search('B', ctype):
        return count + 'B'
    # character
    elif re.search('s', ctype):
        num = re.findall('\d+', ctype)[0]
        return count + 'A{}'.format(num)
    # otherwise
    else:
        raise ValueError(ctype)


def dtype_to_tform(dtype, shape=None):
    """Convert NumPy dtype to FITS format.

    Args:
        dtype (str): A dtype string of NumPy.
        shape (int or tuple of int, optional): shape of the C-type.

    Returns:
        tform (str): A format string of FITS (TFORM).

    References
        https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
        http://docs.astropy.org/en/stable/io/fits/usage/table.html

    """
    count = ''
    if shape is not None:
        prod = np.prod(shape)
        if prod != 1:
            count = str(prod)

    # 32-bit integer
    if re.search('i4', dtype):
        return count + 'J'
    # 64-bit integer
    elif re.search('i8', dtype):
        return count + 'K'
    # single precision floating point
    elif re.search('f4', dtype):
        return count + 'E'
    # double precision floating point
    elif re.search('f8', dtype):
        return count + 'D'
    # character or Unicode
    elif re.search('S|U', dtype):
        num = re.findall('\d+', dtype)[0]
        return count + 'A{}'.format(num)
    # otherwise
    else:
        raise ValueError(dtype)

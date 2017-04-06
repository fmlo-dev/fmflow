# coding: utf-8

# imported items
__all__ = ['ctype_to_tform', 'dtype_to_tform']

# standard library
import re


# functions
def ctype_to_tform(ctype):
    """Convert Python C-type to FITS format.

    Args:
        ctype (str): A C-type format string in Python.

    Returns:
        tform (str): A format string of FITS (TFORM).

    References
        https://docs.python.org/3.6/library/struct.html
        http://docs.astropy.org/en/stable/io/fits/usage/table.html

    """
    # 32-bit integer
    if re.search('i', ctype):
        return 'J'
    # 64-bit integer
    elif re.search('q', ctype):
        return 'K'
    # single precision floating point
    elif re.search('f', ctype):
        return 'E'
    # double precision floating point
    elif re.search('d', ctype):
        return 'D'
    # unsigned byte
    elif re.search('B', ctype):
        return 'B'
    # character
    elif re.search('s', ctype):
        num = re.findall('\d+', ctype)[0]
        return 'A{}'.format(num)
    # otherwise
    else:
        raise ValueError(ctype)


def dtype_to_tform(dtype):
    """Convert NumPy dtype to FITS format.

    Args:
        dtype (str): A dtype string of NumPy.

    Returns:
        tform (str): A format string of FITS (TFORM).

    References
        https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
        http://docs.astropy.org/en/stable/io/fits/usage/table.html

    """
    # 32-bit integer
    if re.search('i4', dtype):
        return 'J'
    # 64-bit integer
    elif re.search('i8', dtype):
        return 'K'
    # single precision floating point
    elif re.search('f4', dtype):
        return 'E'
    # double precision floating point
    elif re.search('f8', dtype):
        return 'D'
    # character or Unicode
    elif re.search('S|U', dtype):
        num = re.findall('\d+', dtype)[0]
        return 'A{}'.format(num)
    # otherwise
    else:
        raise ValueError(dtype)

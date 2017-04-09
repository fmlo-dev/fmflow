# coding: utf-8

# imported items
__all__ = ['ctype_to_tform', 'dtype_to_tform']

# standard library
import re


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
    num = ''
    if shape is not None:
        prod = np.prod(shape)
        if prod != 1:
            num = str(prod)

    # 32-bit integer
    if re.search('i', ctype):
        return num + 'J'
    # 64-bit integer
    elif re.search('q', ctype):
        return num + 'K'
    # single precision floating point
    elif re.search('f', ctype):
        return num + 'E'
    # double precision floating point
    elif re.search('d', ctype):
        return num + 'D'
    # unsigned byte
    elif re.search('B', ctype):
        return num + 'B'
    # character
    elif re.search('s', ctype):
        num = re.findall('\d+', ctype)[0]
        return num + 'A{}'.format(num)
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
    num = ''
    if shape is not None:
        prod = np.prod(shape)
        if prod != 1:
            num = str(prod)

    # 32-bit integer
    if re.search('i4', dtype):
        return num + 'J'
    # 64-bit integer
    elif re.search('i8', dtype):
        return num + 'K'
    # single precision floating point
    elif re.search('f4', dtype):
        return num + 'E'
    # double precision floating point
    elif re.search('f8', dtype):
        return num + 'D'
    # character or Unicode
    elif re.search('S|U', dtype):
        num = re.findall('\d+', dtype)[0]
        return num + 'A{}'.format(num)
    # otherwise
    else:
        raise ValueError(dtype)

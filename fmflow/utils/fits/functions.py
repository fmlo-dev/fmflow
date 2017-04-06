# coding: utf-8

# imported items
__all__ = ['ctype_to_tform']

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
    # character
    if re.search('s', ctype):
        return 'A{}'.format(re.findall('\d+', ctype)[0])
    # unsigned byte
    elif re.search('B', ctype):
        return 'B'
    # 32-bit integer
    elif re.search('i', ctype):
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
    # otherwise
    else:
        raise ValueError(ctype)



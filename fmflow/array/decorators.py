# coding: utf-8

# imported items
__all__ = ['arrayfunc']

# standard library
from collections import partial, wraps
from inspect import getargspec

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr


# decorators
def arrayfunc(func):
    """Make a function compatible with array.

    This function should be used as a decorator like::

        >>> @fm.arrayfunc
        >>> def func(array):
        ...     # some operations ...
        ...     return array
        >>>
        >>> result = func(array)

    Args:
        func (function): A function to be wrapped. The first argument
            of the function must be an array to be processed.

    Returns:
        wrapper (function): A wrapped function.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        array = args[0]
        argnames = getargspec(func).args
        if len(args) > 1:
            for i in range(1, len(args)):
                kwargs[argnames[i]] = args[i]

        if isinstance(array, xr.DataArray):
            result = func(array.values, **kwargs)
            return fm.ones_like(array) * result
        else:
            return func(array)

    return wrapper



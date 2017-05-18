# coding: utf-8

# imported items
__all__ = ['arrayfunc', 'numchunk', 'timechunk']

# standard library
from functools import partial, wraps
from inspect import Parameter, signature

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr

# constants
POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD


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

        if isinstance(array, xr.DataArray):
            result = func(array.values, *args[1:], **kwargs)
            return fm.ones_like(array) * result
        else:
            return func(array, *args[1:], **kwargs)

    return wrapper


def numchunk(func):
    """Make a function compatible with multicore numchunk processing.

    This function should be used as a decorator like::

        >>> @fm.numchunk
        >>> def func(array):
        ...     # some operations ...
        ...     return array # do nothing
        >>>
        >>> result = func(array, numchunk=10)

    Args:
        func (function): A function to be wrapped. The first argument
            of the function must be an array to be num-chunked.

    Returns:
        wrapper (function): A wrapped function.

    """
    @wraps(func)
    @arrayfunc
    def wrapper(*args, **kwargs):
        array = args[0]
        params = signature(func).parameters
        for i, key in enumerate(params):
            if params[key].kind == POSITIONAL_OR_KEYWORD:
                if i > 0:
                    kwargs.update({key: args[i]})

        p = fm.utils.MPPool()
        N = kwargs.pop('numchunk', p.processes)
        pfunc = partial(func, **kwargs)
        subarrays = np.array_split(array, N)

        return np.concatenate(p.map(pfunc, subarrays))

    return wrapper


def timechunk(func):
    """Make a function compatible with multicore timechunk processing.

    This function is used as a decorator like::

        >>> @fm.utils.timechunk
        >>> def func(array):
        ...     return array # do nothing
        >>>
        >>> result = func(array, timechunk=100)

    Args:
        func (function): A function to be wrapped. The first argument
            of the function must be an array to be time-chunked.

    Returns:
        wrapper (function): A wrapped function.

    """
    @wraps(func)
    @arrayfunc
    def wrapper(*args, **kwargs):
        array = args[0]
        argnames = getargspec(func).args
        if len(args) > 1:
            for i in range(1, len(args)):
                kwargs[argnames[i]] = args[i]

        p = fm.utils.MPPool()
        T = kwargs.pop('timechunk', len(array))
        N = int(round(len(array) / T))
        pfunc = partial(func, **kwargs)
        subarrays = np.array_split(array, N)

        return np.concatenate(p.map(pfunc, subarrays))

    return wrapper

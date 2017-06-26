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
EMPTY = Parameter.empty
POS_OR_KWD = Parameter.POSITIONAL_OR_KEYWORD


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
        arrays = []
        sequences = []

        params = signature(func).parameters
        for i, key in enumerate(params):
            if params[key].kind == POS_OR_KWD:
                if params[key].default == EMPTY:
                    arrays.append(np.asarray(args[i]))
                else:
                    try:
                        kwargs.update({key: args[i]})
                    except IndexError:
                        kwargs.update({key: params[key].default})

        p = fm.utils.MPPool(kwargs.pop('n_processes', None))
        N = kwargs.pop('numchunk', p.n_processes)
        pfunc = partial(func, **kwargs)
        for i in range(len(arrays)):
            sequences.append(np.array_split(arrays[i], N))

        return np.concatenate(p.map(pfunc, *sequences))

    return wrapper


def timechunk(func):
    """Make a function compatible with multicore timechunk processing.

    This function is used as a decorator like::

        >>> @fm.timechunk
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
        arrays = []
        sequences = []

        params = signature(func).parameters
        for i, key in enumerate(params):
            if params[key].kind == POS_OR_KWD:
                if params[key].default == EMPTY:
                    arrays.append(np.asarray(args[i]))
                else:
                    try:
                        kwargs.update({key: args[i]})
                    except IndexError:
                        kwargs.update({key: params[key].default})

        p = fm.utils.MPPool(kwargs.pop('n_processes', None))
        T = kwargs.pop('timechunk', len(arrays[0]))
        N = int(round(len(arrays[0]) / T))
        pfunc = partial(func, **kwargs)
        for i in range(len(arrays)):
            sequences.append(np.array_split(arrays[i], N))

        return np.concatenate(p.map(pfunc, *sequences))

    return wrapper

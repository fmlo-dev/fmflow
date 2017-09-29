# coding: utf-8

# public items
__all__ = [
    'chunk',
    'numpyfunc',
]

# standard library
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import wraps
from inspect import Parameter, signature, stack
from multiprocessing import cpu_count
from sys import _getframe as getframe

# dependent packages
import fmflow as fm
import numpy as np
import xarray as xr

# module constants
POS_OR_KWD = Parameter.POSITIONAL_OR_KEYWORD
try:
    MAX_WORKERS = cpu_count() - 1
except:
    MAX_WORKERS = 1


# decorators
def numpyfunc(func):
    """Make a function compatible with xarray.DataArray.

    This function is intended to be used as a decorator like::

        >>> @fm.numpyfunc
        >>> def func(array):
        ...     # do something
        ...     return newarray
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
        if any(isinstance(arg, xr.DataArray) for arg in args):
            newargs = []
            for arg in args:
                if isinstance(arg, xr.DataArray):
                    newargs.append(arg.values)
                else:
                    newargs.append(arg)

            return fm.full_like(args[0], func(*newargs, **kwargs))
        else:
            return func(*args, **kwargs)

    return wrapper


def chunk(*argnames, concatfunc=None):
    """Make a function compatible with multicore chunk processing.

    This function is intended to be used as a decorator like::

        >>> @fm.chunk('array')
        >>> def func(array):
        ...     # do something
        ...     return newarray
        >>>
        >>> result = func(array, timechunk=10)

    or you can set a global chunk parameter outside the function::

        >>> timechunk = 10
        >>> result = func(array)

    """
    def _chunk(func):
        depth = [s.function for s in stack()].index('<module>')
        f_globals = getframe(depth).f_globals

        # original (unwrapped) function
        orgname = '_original_' + func.__name__
        orgfunc = fm.utils.copy_function(func, orgname)
        f_globals[orgname] = orgfunc

        @wraps(func)
        def wrapper(*args, **kwargs):
            depth = [s.function for s in stack()].index('<module>')
            f_globals = getframe(depth).f_globals

            # parse args and kwargs
            params = signature(func).parameters
            for i, (key, val) in enumerate(params.items()):
                if not val.kind == POS_OR_KWD:
                    break

                try:
                    kwargs.update({key: args[i]})
                except IndexError:
                    kwargs.setdefault(key, val.default)

            # n_chunks and n_processes
            if not argnames:
                n_chunks = 1
            elif 'numchunk' in kwargs:
                n_chunks = kwargs.pop('numchunk')
            elif 'timechunk' in kwargs:
                length = len(kwargs[argnames[0]])
                n_chunks = round(length / kwargs.pop('timechunk'))
            elif 'numchunk' in f_globals:
                n_chunks = f_globals['numchunk']
            elif 'timechunk' in f_globals:
                length = len(kwargs[argnames[0]])
                n_chunks = round(length / f_globals['timechunk'])
            else:
                n_chunks = 1

            if 'n_processes' in kwargs:
                n_processes = kwargs.pop('n_processes')
            elif 'n_processes' in f_globals:
                n_processes = f_globals['n_processes']
            else:
                n_processes = MAX_WORKERS

            # make chunked args
            chunks = {}
            for name in argnames:
                arg = kwargs.pop(name)
                try:
                    nargs = np.array_split(arg, n_chunks)
                except TypeError:
                    nargs = np.tile(arg, n_chunks)

                chunks.update({name: nargs})

            # run the function
            with fm.utils.one_thread_per_process(), Pool(n_processes) as p:
                futures = []
                for i in range(n_chunks):
                    chunk = {k:v[i] for k,v in chunks.items()}
                    futures.append(p.submit(orgfunc, **{**chunk, **kwargs}))

            results = [future.result() for future in futures]

            # make an output
            if concatfunc is not None:
                return concatfunc(results)

            try:
                return xr.concat(results, 't')
            except TypeError:
                return np.concatenate(results, 0)

        return wrapper

    return _chunk

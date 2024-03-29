# coding: utf-8

# public items
__all__ = [
    "chunk",
    "xarrayfunc",
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
DEFAULT_N_CHUNKS = 1
try:
    MAX_WORKERS = cpu_count() - 1
except:
    MAX_WORKERS = 1


# decorators
def xarrayfunc(func):
    """Make a function compatible with xarray.DataArray.

    This function is intended to be used as a decorator like::

        >>> @fm.xarrayfunc
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
        depth = [s.function for s in stack()].index("<module>")
        f_globals = getframe(depth).f_globals

        # original (unwrapped) function
        orgname = "_original_" + func.__name__
        orgfunc = fm.utils.copy_function(func, orgname)
        f_globals[orgname] = orgfunc

        @wraps(func)
        def wrapper(*args, **kwargs):
            depth = [s.function for s in stack()].index("<module>")
            f_globals = getframe(depth).f_globals

            # parse args and kwargs
            params = signature(func).parameters
            for i, (key, val) in enumerate(params.items()):
                if not val.kind == Parameter.POSITIONAL_OR_KEYWORD:
                    break

                try:
                    kwargs.update({key: args[i]})
                except IndexError:
                    kwargs.setdefault(key, val.default)

            # n_chunks and n_processes
            n_chunks = DEFAULT_N_CHUNKS
            n_processes = MAX_WORKERS
            multiprocess = True

            if argnames:
                length = len(kwargs[argnames[0]])

                if "numchunk" in kwargs:
                    n_chunks = kwargs.pop("numchunk")
                elif "timechunk" in kwargs:
                    n_chunks = round(length / kwargs.pop("timechunk"))
                elif "numchunk" in f_globals:
                    n_chunks = f_globals["numchunk"]
                elif "timechunk" in f_globals:
                    n_chunks = round(length / f_globals["timechunk"])

                if "n_processes" in kwargs:
                    n_processes = kwargs.pop("n_processes")
                elif "n_processes" in f_globals:
                    n_processes = f_globals["n_processes"]

                if "multiprocess" in kwargs:
                    multiprocess = kwargs.pop("multiprocess")
                elif "multiprocess" in f_globals:
                    multiprocess = f_globals["multiprocess"]

            # make chunked args
            chunks = {}
            for name in argnames:
                arg = kwargs.pop(name)
                try:
                    chunks.update({name: np.array_split(arg, n_chunks)})
                except TypeError:
                    chunks.update({name: np.tile(arg, n_chunks)})

            # run the function
            futures = []
            results = []

            if multiprocess:
                with fm.utils.one_thread_per_process(), Pool(n_processes) as p:
                    for i in range(n_chunks):
                        chunk = {key: val[i] for key, val in chunks.items()}
                        futures.append(p.submit(orgfunc, **{**chunk, **kwargs}))

                    for future in futures:
                        results.append(future.result())
            else:
                for i in range(n_chunks):
                    chunk = {key: val[i] for key, val in chunks.items()}
                    results.append(orgfunc(**{**chunk, **kwargs}))

            # make an output
            if concatfunc is not None:
                return concatfunc(results)

            try:
                return xr.concat(results, "t")
            except TypeError:
                return np.concatenate(results, 0)

        return wrapper

    return _chunk

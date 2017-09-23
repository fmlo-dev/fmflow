# coding: utf-8

# public items
__all__ = [
    'copy_function',
    'get_filename',
    'ignore_numpy_errors',
    'one_thread_per_process',
]

# standard library
import tkinter
from contextlib import contextmanager
from tkinter.filedialog import askopenfilename
from types import CodeType, FunctionType

# dependent packages
import numpy as np


# function
def copy_function(func, name=None):
    """Copy a function object with different name.

    Args:
        func (function): A function to be copied.
        name (string, optional): A name of the new function.
            If not spacified, the same name of `func` will be used.

    Returns:
        newfunc (function): A new function with different name.

    """
    code = func.__code__
    newname = name or func.__name__
    newcode = CodeType(
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        code.co_consts,
        code.co_names,
        code.co_varnames,
        code.co_filename,
        newname,
        code.co_firstlineno,
        code.co_lnotab,
        code.co_freevars,
        code.co_cellvars,
    )
    newfunc = FunctionType(
        newcode,
        func.__globals__,
        newname,
        func.__defaults__,
        func.__closure__,
    )
    newfunc.__dict__.update(func.__dict__)
    return newfunc


def get_filename():
    """Get filename by a dialog window."""
    root = tkinter.Tk()
    root.withdraw()
    filename = askopenfilename()
    root.quit()
    return filename


@contextmanager
def ignore_numpy_errors():
    """Return a context manager where all numpy errors are ignored.

    This function is intended to be used as a with statement like::

        >>> with ignore_numpy_errors():
        ...     np.arange(10) / 0 # no errors are displayed

    """
    old_settings = np.seterr(all='ignore')

    try:
        # execute nested block in the with statement
        yield
    finally:
        # revert to the original value
        np.seterr(**old_settings)


@contextmanager
def one_thread_per_process():
    """Return a context manager where only one thread is allocated to a process.

    This function is intended to be used as a with statement like::

        >>> with process_per_thread():
        ...     do_something() # one thread per process

    Notes:
        This function only works when MKL (Intel Math Kernel Library)
        is installed and used in, for example, NumPy and SciPy.
        Otherwise this function does nothing.

    """
    try:
        import mkl
        is_mkl = True
    except ImportError:
        is_mkl = False

    if is_mkl:
        n_threads = mkl.get_max_threads()
        mkl.set_num_threads(1)
        try:
            # execute nested block in the with statement
            yield
        finally:
            # revert to the original value
            mkl.set_num_threads(n_threads)
    else:
        yield

# coding: utf-8

# imported items
__all__ = ['inprogress', 'progressbar']

# dependent packages
import numpy as np


# functions
def inprogress(message='in progress', interval=50):
    """Print a status of 'in progress' by message and rotating bar.

    Example:
        >>> bar = fm.utils.inprogress()
        >>> while True:
        ...     next(bar) # print the next state

    Args:
        message (str, optional): A message. Default is 'in progress'.
        interval (int, optional): refresh interval of the rotating bar. Default is 50.
    """

    i = 0
    while True:
        if i%interval == 0:
            rotator = '|/-\\'[int(i/interval)%4]
            status = '{} {} '.format(message, rotator)
            print('\r'+status, end='', flush=True)

        i += 1
        yield


def progressbar(fraction, width=50):
    """Print a progress bar with progress rate in units of percent.

    Example:
        >>> for frac in np.arange(0.0, 1.0, 0.01):
        ...     fm.utils.progressbar(frac)

    Args:
        fraction (float): progress rate between [1.0, 1.0].
        width (int, optional): width of the progress bar in units of pt.

    """
    N = int(np.ceil(width*fraction))
    fill  = '*' * N
    space = ' ' * (width-N)
    status = '[{}{}] {:.0%} '.format(fill, space, fraction)
    print('\r'+status, end='', flush=True)

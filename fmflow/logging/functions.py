# coding: utf-8

# imported items
__all__ = [
    'setlogfile',
    'setloglevel',
]

# standard library
import logging
import os
import sys

# dependent packages
import fmflow as fm

# constants
DATEFORMAT = '%Y-%m-%d %H:%M:%S'
FORMAT = '%(asctime)s\t%(funcName)s\t[%(levelname)s]\t%(message)s'


# functions
def setlogfile(filename=None, *, logger=None):
    """Create a file where messages will be logged.

    Args:
        filename (str): A file name of logging.
            If not spacified, messages will be printed to stdout.
        logger (logging.Logger, optional): A logger. Default is `fm.logger`.

    """
    if logger is None:
        logger = fm.logger

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    if filename is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(os.path.expanduser(filename))

    formatter = logging.Formatter(FORMAT, DATEFORMAT)
    handler.setFormatter(formatter)
    handler.setLevel(logger.level)
    logger.addHandler(handler)


def setloglevel(level='INFO', *, logger=None):
    """Set a logging level above which messages will be logged.

    Args:
        level (str or int): A logging level. Default is 'INFO'.
        logger (logging.Logger, optional): A logger. Default is `fm.logger`.

    References
        https://docs.python.jp/3/library/logging.html#logging-levels

    """
    if logger is None:
        logger = fm.logger

    logger.setLevel(level.upper())
    for handler in logger.handlers:
        handler.setLevel(level.upper())

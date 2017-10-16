# coding: utf-8

__all__ = [
    'logger',
    'setlogger',
]

# standard library
import logging
from copy import copy
from pathlib import Path

# dependent packages
import fmflow as fm

# module constants
DATEFORMAT = '%Y-%m-%d %H:%M:%S'
LOGFORMAT  = '{asctime} | {levelname:8} | {name}: {message}'
DEFAULTLEVEL = 'INFO'

# default logger
logger = logging.getLogger('fmflow')
logger.addHandler(logging.NullHandler())
logger.propagate = False


# classes
class setlogger(object):
    def __init__(self, level=None, filename=None, overwrite=False, encoding='utf-8'):
        self.oldhandlers = copy(fm.logger.handlers)
        self.oldlevel = copy(fm.logger.level)
        self.sethandlers(filename, overwrite, encoding)
        self.setlevel(level)

    @staticmethod
    def sethandlers(filename, overwrite, encoding):
        for handler in fm.logger.handlers:
            fm.logger.removeHandler(handler)

        if filename is None:
            handler = logging.StreamHandler()
        else:
            filename = str(Path(filename).expanduser())
            mode = 'w' if overwrite else 'a'
            handler = logging.FileHandler(filename, mode, encoding)

        formatter = logging.Formatter(LOGFORMAT, DATEFORMAT, style='{')
        handler.setFormatter(formatter)
        fm.logger.addHandler(handler)

    @staticmethod
    def setlevel(level):
        level = DEFAULTLEVEL if level is None else level.upper()
        fm.logger.setLevel(level)
        for handler in fm.logger.handlers:
            handler.setLevel(level)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        fm.logger.handlers = self.oldhandlers
        fm.logger.level = self.oldlevel

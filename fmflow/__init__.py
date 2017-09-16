# coding: utf-8

__version__ = '0.0'
__author__  = 'snoopython'

from . import utils
from .core import *
from .logging import *
from . import fits
from . import models
del core
del logging

# default logger
import logging
logger = logging.getLogger('fmflow')
logger.propagate = False
setlogfile(logger=logger)
setloglevel(logger=logger)
del logging

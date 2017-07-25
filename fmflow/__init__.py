# coding: utf-8

__version__ = '0.0'
__author__  = 'snoopython'

from .array import *
from .logging import *
from . import fits
from . import models
from . import utils

# default logger
import logging
logger = logging.getLogger('fmflow')
setlogfile(logger=logger)
setloglevel(logger=logger)
del logging

# coding: utf-8

# information
__version__ = 'v0.0'
__author__  = 'Akio Taniguchi'

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

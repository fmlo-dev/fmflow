# coding: utf-8

__version__ = '0.2.6'
__author__ = 'astropenguin'

from . import utils
from .core import *
from .logging import *
from . import fits
from . import models
del core
del logging

# for sphinx build
__all__ = dir()

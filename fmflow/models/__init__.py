# coding: utf-8

# base model
class BaseModel(object):
    def __init__(self, params, logger=None):
        import fmflow as fm
        self.params = params
        self.logger = logger or fm.logger
        self.logger.debug(self)

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        clsname = self.__class__.__name__
        params  = self.params
        return '{0}({1})'.format(clsname, params)


# submodules
from .astrosignal.classes import *
from .astrosignal.functions import *
from .atmosphere.classes import *
from .atmosphere.functions import *
from .commonmode.classes import *
from .commonmode.functions import *
from .gain.classes import *
from .gain.functions import *
del astrosignal
del atmosphere
del commonmode
del gain

# for sphinx build
__all__ = dir()

# coding: utf-8

# dependent packages
import fmflow as fm

# classes
class BaseModel(objects):
    def __init__(self, logger=None):
        self.logger = logger or fm.logger
        self.params = {}
        self.outputs = {}

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        cname = self.__class__.__name__
        params = self.params
        return '{0}({1})'.format(cname, params)

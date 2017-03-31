# coding: utf-8

# imported items
__all__ = ['FMFlowError']


# classes
class FMFlowError(Exception):
    """Error class of FMFlow.

    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)



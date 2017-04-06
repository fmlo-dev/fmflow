# coding: utf-8

# imported items
__all__ = ['DatetimeParser']

# standard library
from datetime import datetime, timedelta

# constants
ISO_8601 = '%Y-%m-%dT%H:%M:%S.%f'
PATTERNS = [
    '%y%m%d%H%M%S',
    '%y%m%d%H%M%S.',
    '%y%m%d%H%M%S.%f',
    '%Y%m%d%H%M%S',
    '%Y%m%d%H%M%S.',
    '%Y%m%d%H%M%S.%f',
]


# classes
class DatetimeParser(object):
    def __init__(self, cutoffsec=True):
        """Initialize a datetime parser object.

        Args:
            cutoffsec (bool, optional): If True, digits smaller than 0.1 second is
                truncated. For example, 0.123 sec becomes 0.100 sec. Default is True.

        """
        self.cutoffsec = cutoffsec
        self.pattern = None

    def __call__(self, string, encoding='utf-8'):
        """Convert a datetime string to that in ISO format.

        Args:
            string (str): A datetime string.
            encoding (str, optional): An encoding with which to decode `string` if it is bytes.

        Returns:
            isostring (str): A datetime string in ISO format.

        """
        if type(string) == bytes:
            string = string.decode(encoding)

        try:
            dt = datetime.strptime(string, self.pattern)
        except:
            self._setpattern(string)
            dt = datetime.strptime(string, self.pattern)

        if self.cutoffsec:
            return dt.strftime(ISO_8601)[:-5] + '00000'
        else:
            return dt.strftime(ISO_8601)

    def _setpattern(self, string):
        for pattern in PATTERNS:
            try:
                dt = datetime.strptime(string, pattern)
                self.pattern = pattern
                break
            except:
                continue

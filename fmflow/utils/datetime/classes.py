# coding: utf-8

# imported items
__all__ = ['DatetimeParser']

# standard library
from datetime import datetime

# dependent packages
import numpy as np

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
    def __init__(self, cutoffsec=True, encoding='utf-8'):
        """Initialize a datetime parser object.

        Args:
            cutoffsec (bool, optional): If True, digits smaller than 0.1 second is
                truncated. For example, 0.123 sec becomes 0.100 sec. Default is True.
            encoding (str, optional): An encoding with which to decode datetime strings
                if their type is bytes. Default is utf-8.

        """
        self.info = {'cutoffsec': cutoffsec, 'encoding': encoding}
        self._pattern = None

    def __call__(self, dt_string):
        """Convert a datetime string to that in ISO format.

        Args:
            dt_string (str): A datetime string.

        Returns:
            isostring (str): A datetime string in ISO format.

        """
        if type(dt_string) == bytes:
            dt_string = dt_string.decode(self.info['encoding'])
        elif type(dt_string) == np.bytes_:
            dt_string = dt_string.tobytes().decode(self.info['encoding'])

        try:
            dt = datetime.strptime(dt_string, self._pattern)
        except:
            self._setpattern(dt_string)
            dt = datetime.strptime(dt_string, self._pattern)

        if self.info['cutoffsec']:
            return dt.strftime(ISO_8601)[:-5] + '00000'
        else:
            return dt.strftime(ISO_8601)

    def _setpattern(self, dt_string):
        for pattern in PATTERNS:
            try:
                dt = datetime.strptime(dt_string, pattern)
                self._pattern = pattern
                break
            except:
                continue

# coding: utf-8

# public items
__all__ = [
    'DatetimeParser'
]

# standard library
from datetime import datetime

# module constants
ISO_8601 = '%Y-%m-%dT%H:%M:%S.%f'
PATTERNS = [
    '%y%m%d%H%M%S',
    '%y%m%d%H%M%S.',
    '%y%m%d%H%M%S.%f',
    '%Y%m%d%H%M%S',
    '%Y%m%d%H%M%S.',
    '%Y%m%d%H%M%S.%f',
    ISO_8601
]


# classes
class DatetimeParser(object):
    def __init__(self, outputiso=True, cutoffsec=True, encoding='utf-8'):
        """Initialize a datetime parser object.

        Args:
            outputiso (bool, optional): If True, the output object is an ISO 8601 string
                (e.g. YYYY-mm-ddTHH:MM:SS.ssssss). Otherwise output is a datetime object.
                Default is True.
            cutoffsec (bool, optional): If True, digits smaller than 0.1 second is
                truncated. For example, 0.123 sec becomes 0.100 sec. Default is True.
            encoding (str, optional): An encoding with which to decode datetime strings
                if their type is bytes. Default is utf-8.

        """
        self.params = {
            'outputiso': outputiso,
            'cutoffsec': cutoffsec,
            'encoding': encoding
        }
        self._pattern = None

    def __call__(self, datetime_like):
        """Convert a datetime string or datetime object to a string in ISO format.

        Args:
            datetime_like (str or datetime): A datetime string or datetime object.

        Returns:
            datetime_like (str or datetime): A datetime string in ISO format.
                if outputiso is True. Otherwise output is a datetime object.

        """
        if issubclass(type(datetime_like), str):
            dt_string = datetime_like
        elif issubclass(type(datetime_like), bytes):
            dt_string = datetime_like.decode(self.encoding)
        elif type(datetime_like) == datetime:
            dt_string = datetime_like.strftime(ISO_8601)
        else:
            raise ValueError(datetime_like)

        try:
            dt = datetime.strptime(dt_string, self._pattern)
        except:
            self._setpattern(dt_string)
            dt = datetime.strptime(dt_string, self._pattern)

        if self.cutoffsec:
            dt_isostring = dt.strftime(ISO_8601)[:-5] + '00000'
        else:
            dt_isostring = dt.strftime(ISO_8601)

        if self.outputiso:
            return dt_isostring
        else:
            return datetime.strptime(dt_isostring, ISO_8601)

    def _setpattern(self, dt_string):
        for pattern in PATTERNS:
            try:
                dt = datetime.strptime(dt_string, pattern)
                self._pattern = pattern
                break
            except ValueError:
                continue

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'DatetimeParser({0})'.format(self.params)

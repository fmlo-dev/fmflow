# coding: utf-8

# public items
__all__ = [
    'CStructReader'
]

# standard library
import json
import re
from collections import deque, OrderedDict
from struct import Struct

# dependent packages
import numpy as np


# classes
class CStructReader(object):
    """Read a binary file to unpack values in a C structure.

    C structures can be expressed as a Python list that stores name, type, and shape
    of each member in a C structure. For example, the following C structure::

        struct structure {
        int a[2];
        float b[2][3];
        char c[10];
        };

    will be expressed as the following Python list::

        structure = [
            ('a', 'i', 2),
            ('b', 'd', (2,3)),
            ('c', '10s')
        ]

    where the first element of each member is name, the second one is the format
    character (https://docs.python.jp/3/library/struct.html#format-characters)
    that represents C type in Python, and the third one (optional) is shape.

    Example:
        >>> structure = [('a', 'i', 2), ('b', 'd', (2,3)), ('c', '10s')]
        >>> reader = fm.utils.CStructReader(structure)
        >>> with open('binaryfile', 'rb') as f:
        ...     reader.read(f)
        >>> reader.data
        OrderedDict([
            ('a', array([0, 1])),
            ('b', array([[0, 1, 2],[3, 4, 5]])),
            ('c', 'some text')
        ])

    Attributes:
        size (int): A byte size of the structure.
        data (OrderedDict): An ordered dictionary that stores unpacked values.
        jsondata (OrderedDict): A JSON string that stores unpacked values.
        params (dict): Stored information about the structure, ignored, and byteorder.

    References:
        https://docs.python.jp/3/library/struct.html#module-struct

    """
    def __init__(self, structure, ignored='$.', byteorder='<', encoding='utf-8'):
        """Initialize a C structure reader.

        Args:
            structure (list of item): A Python list that stores name, type, and shape
                of each member in a C structure. See example for more information.
            ignored (str, optional): A string of regular expression for ignoring
                specific name(s) of data when reading a binary file. Default is '$.'
                (all names of data are not ignored).
            byteorder (str, optional): A format character that indicates the byte ordered
                of a binary file. Default is '<' (little endian). Use '>' for big endian.
            encoding (str, optional): An encoding with which to decode string objects
                if their type is bytes (for the jsondata attribute). Default is utf-8.

        """
        self.params = {
            'ignored': ignored,
            'byteorder': byteorder,
            'encoding': encoding,
        }

        self.params['ctypes'], self.params['shapes'] = self._parse(structure)
        self._data = OrderedDict((name,[]) for name in self.params['ctypes'])
        self._unpacker = Struct(self._joinedctypes())
        self.size = self._unpacker.size

    def read(self, f):
        """Sequentially read a file object to unpack values in a C structure.

        Values are stored in the C structure reader instance as an ordered dictionary.
        Use `data` or `jsondata` attributes to access them.

        Args:
            f (file): A binary file object to be read.
                It must be `open`ed with `b` option.

        """
        bindata = f.read(self._unpacker.size)
        unpdata = deque(self._unpacker.unpack(bindata))
        for name, shape in self.shapes.items():
            datum = [unpdata.popleft() for i in range(np.prod(shape))]
            self._data[name].append(np.asarray(datum))

    @property
    def data(self):
        """An ordered dictionary that stores unpacked values."""
        data = OrderedDict()
        for name, shape in self.shapes.items():
            if re.search(self.ignored, name):
                continue

            _data = self._data[name]
            datum = np.reshape(_data, [len(_data)]+shape)
            if np.prod(datum.shape) == 1:
                data[name] = np.squeeze(datum).item()
            else:
                data[name] = np.squeeze(datum)

        return data

    @property
    def jsondata(self):
        """An JSON string that stores unpacked values."""
        data = self.data
        encoding = self.encoding
        for name, datum in data.items():
            if type(datum) == bytes:
                data[name] = datum.decode(encoding)
            elif type(datum) == np.ndarray:
                if datum.dtype.kind == 'S':
                    data[name] = np.char.decode(datum, encoding).tolist()
                else:
                    data[name] = datum.tolist()
            else:
                continue

        return json.dumps(data)

    def _joinedctypes(self):
        """A Joined C-type string for the Struct class."""
        joinedctypes = self.byteorder
        for name, ctype in self.ctypes.items():
            count = np.prod(self.shapes[name])
            joinedctypes += ctype * count

        return joinedctypes

    def _parse(self, structure):
        """Convert the structure to ctypes and shapes ordered dicts."""
        ctypes = OrderedDict()
        shapes = OrderedDict()
        for item in structure:
            if len(item) == 2:
                (name, ctype), shape = item, [1]
            elif len(item) == 3:
                if type(item[2]) == int:
                    (name, ctype), shape = item[:2], [item[2]]
                elif type(item[2]) in (list, tuple):
                    (name, ctype), shape = item[:2], list(item[2])
                else:
                    raise ValueError(item)
            else:
                raise ValueError(item)

            ctypes[name] = ctype
            shapes[name] = shape

        return ctypes, shapes

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'CStructReader({0})'.format(self.params)

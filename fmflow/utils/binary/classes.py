# coding: utf-8

# imported items
__all__ = ['CStructReader']

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

        struct schedule {
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
        data (OrderedDict): An ordered dictionary that stores unpacked values.
        jsondata (OrderedDict): An JSON string that stores unpacked values.
        fitsformats (OrderedDict): An ordered dictionary of FITS formats corresponding dtypes.
        info (dict): Stored information about the structure, ignored, and byteorder.

    References:
        https://docs.python.jp/3/library/struct.html#module-struct

    """
    def __init__(self, structure, ignored='$.', byteorder='<'):
        """Initialize a C structure reader.

        Args:
            structure (list of item): A Python list that stores name, type, and shape
                of each member in a C structure. See example for more information.
            ignored (str, optional): A string of regular expression for ignoring
                specific name(s) of data when reading a binary file. Default is '$.'
                (all names of data are not ignored).
            byteorder (str, optional): A format character that indicates the byte ordered
                of a binary file. Default is '<' (little endian). Use '>' for big endian.

        """
        self.info = {'ignored': ignored, 'byteorder': byteorder}
        self.info['dtypes'], self.info['shapes'] = self._parse(structure)
        self._data = OrderedDict((name,[]) for name in self.info['dtypes'])
        self._unpacker = Struct(self._joineddtypes())

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
        for name, shape in self.info['shapes'].items():
            datum = [unpdata.popleft() for i in range(np.prod(shape))]
            self._data[name].append(np.asarray(datum))

    @property
    def data(self):
        """An ordered dictionary that stores unpacked values."""
        data = OrderedDict()
        for name, shape in self.info['shapes'].items():
            if re.search(self.info['ignored'], name):
                continue

            _data = self._data[name]
            datum = _data.reshape([len(_data)]+shape)
            if np.prod(datum.shape) == 1:
                data[name] = np.squeeze(datum).item()
            else:
                data[name] = np.squeeze(datum)

        return data

    @property
    def jsondata(self):
        """An JSON string that stores unpacked values."""
        data = self.data
        for name, datum in data.items():
            if isinstance(datum, np.ndarray):
                data[name] = datum.tolist()

        return json.dumps(data)

    @property
    def fitsformats(self):
        """An ordered dictionary of FITS formats corresponding dtypes."""
        fitsformats = OrderedDict()
        for name, dtype in self.info['dtypes'].items():
            count = np.prod(self.info['shapes'][name])

            if re.search('s', dtype):
                code = 'A'
                code += re.findall('\d+', dtype)[0]
            elif re.search('B', dtype):
                code = 'B'
            elif re.search('i', dtype):
                code = 'J'
            elif re.search('d', dtype):
                code = 'D'
            elif re.search('f', dtype):
                code = 'E'
            else:
                raise ValueError(dtype)

            if count == 1:
                fitsformats[name] = code
            else:
                fitsformats[name] = str(count) + code

        return fitsformats

    def _joineddtypes(self):
        joineddtypes = self.info['byteorder']
        for name, dtype in self.info['dtypes'].items():
            count = np.prod(self.info['shapes'][name])
            joineddtypes += dtype * count

        return joineddtypes

    def _parse(self, structure):
        dtypes = OrderedDict()
        shapes = OrderedDict()
        for item in structure:
            if len(item) == 2:
                (name, dtype), shape = item, tuple([1])
            elif len(item) == 3:
                if type(item[2]) == int:
                    (name, dtype), shape = item[:2], (item[2],)
                elif type(item[2]) in (list, tuple):
                    name, dtype, shape = item[:2], tuple(item[2])
                else:
                    raise ValueError(item)
            else:
                raise ValueError(item)

            dtypes[name] = dtype
            shapes[name] = shape

        return dtypes, shapes

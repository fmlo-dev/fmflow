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
    def __init__(self, structure, ignored='$.', byteorder='<'):
        self.ignored = ignored
        self.byteorder = byteorder
        self.dtypes, self.shapes = self._parse(structure)
        self._data = OrderedDict((name,[]) for name in self.dtypes)
        self.unpacker = Struct(self.joineddtypes)

    def read(self, f):
        bindata = f.read(self.unpacker.size)
        unpdata = deque(self.unpacker.unpack(bindata))
        for name, shape in self.shapes.items():
            datum = [unpdata.popleft() for i in range(np.prod(shape))]
            self._data[name].append(np.asarray(datum))

    @property
    def data(self):
        data = OrderedDict()
        for name, shape in self.shapes.items():
            if re.search(self.ignored, name):
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

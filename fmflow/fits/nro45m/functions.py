# coding: utf-8

# imported items
__all__ = ['fromnro45m']

# standard library
import json
import os
import re
from collections import OrderedDict
from pkgutil import get_data

# dependent packages
import yaml
import fmflow as fm
import numpy as np
from astropy import constants
from astropy import coordinates
from astropy import units as u
from astropy.io import fits

# constants
C           = constants.c.value # spped of light in vacuum
D_ASTE      = (45.0 * u.m).value # diameter of the ASTE
EFF_8257D   = 0.92 # exposure / interval time of Agilent 8257D
IGNORED_KEY = '^reserve' # reserved[1|4|8]
LAT_NRO45m  = coordinates.Angle('+35d56m40.9s').deg # latitude of the NRO45m


# functions
def fromnro45m(fmlolog, backendlog, antennalog=None, byteorder='<'):
    """Read logging data of NRO45m and merge them into a FITS object.

    Args:
        fmlolog (str): File name of FMLO logging.
        backendlog (str): File name of backend logging.
        antennalog (str): File name of antenna logging (optional).
        byteorder (str): format string that represents byte order
            of the backendlog. Default is '<' (little-endian).
            If the data in the returned FITS seems to be wrong,
            try to spacify '>' (big-endian).

    Returns:
        hdus (HDUlist): HDU list containing the merged data.

    """
    # PRIMARY HDU
    hdus = fits.HDUList()
    hdus.append(fits.PrimaryHDU())

    # FMLOINFO HDU
    hdus.append(read_fmlolog(fmlolog))

    # BACKEND and OBSINFO HDUs
    backend = check_backend(backendlog, byteorder)

    if backend == b'SAM45':
        hdus.append(read_backendlog_sam45(backendlog, byteorder))
        hdus.insert(1, make_obsinfo_sam45(hdus))
    else:
        raise fm.utils.FMFlowError('invalid logging type')

    # ANTENNA HDU (if any)
    if antennalog is not None:
        hdus.append(read_antennalog(antennalog))

    return hdus


def read_fmlolog(fmlolog):
    """Read a FMLO logging of NRO45m.

    Args:
        fmlolog (str): File name of FMLO logging.

    Returns:
        hdu (BinTableHDU): HDU containing the read FMLO logging.

    """
    # read fmlolog
    fmts = yaml.load(get_data('fmflow', 'fits/nro45m/data/fmlolog.yaml'))
    names, dtypes, units = list(map(list, zip(*fmts)))
    tforms = list(map(fm.utils.dtype_to_tform, dtypes))

    c = {0: fm.utils.DatetimeParser()}
    data = np.genfromtxt(fmlolog, list(zip(names, dtypes)), skip_header=1, converters=c)

    # bintable HDU
    header = fits.Header()
    header['extname'] = 'FMLOLOG'
    header['filename'] = fmlolog

    cols = [fits.Column(n, tforms[i], units[i], array=data[n]) for i, n in enumerate(names)]
    return fits.BinTableHDU.from_columns(cols, header)



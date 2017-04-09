# coding: utf-8

# imported items
__all__ = ['fromaste']

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
D_ASTE      = (10.0 * u.m).value # diameter of the ASTE
EFF_8257D   = 0.92 # exposure / interval time of Agilent 8257D
IGNORED_KEY = '^[a-z]dmy([^_]|$)' # cdmy, cdmy2, ..., except for idmy_flag
LAT_ASTE    = coordinates.Angle('-22d58m17.69447s').deg # latitude of the ASTE


# functions
def fromaste(fmlolog, backendlog, antennalog=None, byteorder='<'):
    """Read logging data of ASTE and merge them into a FITS object.

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

    See Also:
        https://docs.python.jp/3/library/struct.html

    """
    # PRIMARY HDU
    hdus = fits.HDUList()
    hdus.append(fits.PrimaryHDU())

    # FMLOINFO HDU
    hdus.append(read_fmlolog(fmlolog))

    # BACKEND and OBSINFO HDUs
    backend = check_backend(backendlog, byteorder)

    if backend == b'AC45':
        hdus.append(read_backendlog_mac(backendlog, byteorder))
        hdus.insert(1, make_obsinfo_mac(hdus))
    elif backend == b'FFX':
        raise fm.utils.FMFlowError('WHSF logging is not supported yet')
    else:
        raise fm.utils.FMFlowError('invalid logging type')

    # ANTENNA HDU (if any)
    if antennalog is not None:
        hdus.append(read_antennalog(antennalog))

    return hdus



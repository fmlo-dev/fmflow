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


def read_fmlolog(fmlolog):
    """Read a FMLO logging of ASTE.

    Args:
        fmlolog (str): File name of FMLO logging.

    Returns:
        hdu (BinTableHDU): HDU containing the read FMLO logging.

    """
    # read fmlolog
    fmts = yaml.load(get_data('fmflow', 'fits/aste/data/fmlolog.yaml'))
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


def read_antennalog(antennalog):
    """Read an antenna logging of ASTE.

    Args:
        antennalog (str): File name of antenna logging.

    Returns:
        hdu (BinTableHDU): HDU containing the read antenna logging.

    """
    # read fmlolog
    fmts = yaml.load(get_data('fmflow', 'fits/aste/data/antennalog.yaml'))
    names, dtypes, units = list(map(list, zip(*fmts)))
    tforms = list(map(fm.utils.dtype_to_tform, dtypes))

    c = {0: fm.utils.DatetimeParser()}
    d = np.genfromtxt(antennalog, list(zip(names, dtypes)), skip_header=1, converters=c)

    # RA, Dec real
    sind = lambda deg: np.sin(np.deg2rad(deg))
    cosd = lambda deg: np.cos(np.deg2rad(deg))
    q = -np.arcsin(sind(d['az_prog']) * cosd(LAT_ASTE) / cosd(d['dec_prog']))

    ra_error  = -np.cos(q)*d['az_error'] + np.sin(q)*d['el_error']
    dec_error = +np.sin(q)*d['az_error'] + np.cos(q)*d['el_error']
    ra_real   = d['ra_prog'] - ra_error
    dec_real  = d['dec_prog'] - ra_error

    # bintable HDU
    header = fits.Header()
    header['extname'] = 'ANTENNA'
    header['filename'] = antennalog

    cols = [fits.Column(n, tforms[i], units[i], array=d[n]) for i, n in enumerate(names)]
    cols.append(fits.Column('ra', 'D', 'deg', array=ra_real))
    cols.append(fits.Column('dec', 'D', 'deg', array=dec_real))
    cols.append(fits.Column('ra_error', 'D', 'deg', array=ra_error))
    cols.append(fits.Column('dec_error', 'D', 'deg', array=dec_error))
    return fits.BinTableHDU.from_columns(cols, header)



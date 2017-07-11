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
D_NRO45m      = (45.0 * u.m).value # diameter of the NRO45m
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
    fmts = yaml.load(get_data('fmflow', 'fits/data/nro45m_fmlolog.yaml'))
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
    """Read an antenna logging of NRO45m.

    Args:
        antennalog (str): File name of antenna logging.

    Returns:
        hdu (BinTableHDU): HDU containing the read antenna logging.

    """
    # read fmlolog
    fmts = yaml.load(get_data('fmflow', 'fits/data/nro45m_antennalog.yaml'))
    names, dtypes, units = list(map(list, zip(*fmts)))
    tforms = list(map(fm.utils.dtype_to_tform, dtypes))

    c = {0: fm.utils.DatetimeParser()}
    d = np.genfromtxt(antennalog, list(zip(names, dtypes)), skip_header=1, converters=c)

    # RA, Dec real
    sind = lambda deg: np.sin(np.deg2rad(deg))
    cosd = lambda deg: np.cos(np.deg2rad(deg))
    q = -np.arcsin(sind(d['az_prog']) * cosd(LAT_NRO45m) / cosd(d['dec_prog']))

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


def check_backend(backendlog, byteorder):
    """Check backend type from a backend logging of NRO45m.

    Args:
        backendlog (str): File name of backend logging.
        byteorder (str): format string that represents byte order
            of the backendlog. Default is '<' (little-endian).
            If the data in the returned FITS seems to be wrong,
            try to spacify '>' (big-endian).

    Returns:
        backend (str): Backend type.

    """
    com = yaml.load(get_data('fmflow', 'fits/data/nro45m_backendlog_common.yaml'))
    head = fm.utils.CStructReader(com['head'], IGNORED_KEY, byteorder)
    ctl  = fm.utils.CStructReader(com['ctl'], IGNORED_KEY, byteorder)

    # read backendlog
    with open(backendlog, 'rb') as f:
        head.read(f)
        ctl.read(f)

    return ctl.data['cbe_type']


def read_backendlog_sam45(backendlog, byteorder):
    """Read a backend logging of NRO45m/SAM45.

    Args:
        backendlog (str): File name of backend logging.
        byteorder (str): format string that represents byte order
            of the backendlog. Default is '<' (little-endian).
            If the data in the returned FITS seems to be wrong,
            try to spacify '>' (big-endian).

    Returns:
        hdu (BinTableHDU): HDU containing the read backend logging.

    """
    com = yaml.load(get_data('fmflow', 'fits/data/nro45m_backendlog_common.yaml'))
    mac = yaml.load(get_data('fmflow', 'fits/data/nro45m_backendlog_sam45.yaml'))

    head = fm.utils.CStructReader(com['head'], IGNORED_KEY, byteorder)
    ctl  = fm.utils.CStructReader(com['ctl'], IGNORED_KEY, byteorder)
    obs  = fm.utils.CStructReader(mac['obs'], IGNORED_KEY, byteorder)
    dat  = fm.utils.CStructReader(mac['dat'], IGNORED_KEY, byteorder)

    def eof(f):
        head.read(f)
        return (head._data['crec_type'][-1][0] == b'ED')

    # read backendlog
    fsize = os.path.getsize(backendlog)
    with open(backendlog, 'rb') as f:
        eof(f)
        ctl.read(f)
        eof(f)
        obs.read(f)

        i = 0
        while not eof(f):
            i += 1
            dat.read(f)
            if i%100 == 0:
                frac = f.tell() / fsize
                fm.utils.progressbar(frac)

    # edit data
    data = dat.data
    data['starttime'] = data.pop('cint_sttm')
    data['arrayid']   = data.pop('cary_name')
    data['scantype']  = data.pop('cscan_type')
    data['arraydata'] = data.pop('fary_data')

    ## starttime
    p = fm.utils.DatetimeParser()
    data['starttime'] = np.array([p(t) for t in data['starttime']])

    ## scantype (bug?)
    data['scantype'][data['scantype']==b'ON\x00O'] = b'ON'
    data['scantype'][data['scantype']==b'R\x00RO'] = b'R'

    ## arraydata
    usefg    = np.array(obs.data['iary_usefg'], dtype=bool)
    ifatt    = np.array(obs.data['iary_ifatt'], dtype=float)[usefg]
    islsb    = np.array(obs.data['csid_type'] == b'LSB')[usefg]
    arrayids = np.unique(data['arrayid'])

    for i, arrayid in enumerate(arrayids):
        flag = (data['arrayid'] == arrayid)

        ## slices of each scantype
        ons  = fm.utils.slicewhere(flag & (data['scantype'] == b'ON'))
        rs   = fm.utils.slicewhere(flag & (data['scantype'] == b'R'))
        skys = fm.utils.slicewhere(flag & (data['scantype'] == b'SKY'))
        zero = fm.utils.slicewhere(flag & (data['scantype'] == b'ZERO'))[0]

        ## apply ZERO to ON data
        for on in ons:
            data['arraydata'][on] -= data['arraydata'][zero]

        ## apply ZERO and ifatt to R data
        for r in rs:
            data['arraydata'][r] -= data['arraydata'][zero]
            data['arraydata'][r] *= 10.0**(ifatt[i]/10.0)

        ## apply ZERO to SKY data
        for sky in skys:
            data['arraydata'][sky] -= data['arraydata'][zero]

        ## reverse array (if LSB)
        if arrayid in arrayids[islsb]:
            data['arraydata'][flag] = data['arraydata'][flag,::-1]

    # read and edit formats
    names  = list(dat.info['ctypes'].keys())
    ctypes = list(dat.info['ctypes'].values())
    shapes = list(dat.info['shapes'].values())
    tforms = map(fm.utils.ctype_to_tform, ctypes, shapes)
    fmts = OrderedDict(item for item in zip(names, tforms))

    fmts['starttime'] = 'A26'
    fmts['arraydata'] = fmts.pop('fary_data')
    fmts['arrayid']   = fmts.pop('cary_name')
    fmts['scantype']  = fmts.pop('cscan_type')

    # bintable HDU
    header = fits.Header()
    header['extname']  = 'BACKEND'
    header['filename'] = backendlog
    header['ctlinfo']  = ctl.jsondata
    header['obsinfo']  = obs.jsondata

    cols = [fits.Column(key, fmts[key], array=data[key]) for key in data]
    return fits.BinTableHDU.from_columns(cols, header)


def make_obsinfo_sam45(hdus):
    """Make a OBSINFO HDU of NRO45m/SAM45.

    Args:
        hdus (HDUList): FITS object containing FMLOINFO, BACKEND HDUs.

    Returns:
        hdu (BinTableHDU): OBSINFO HDU containing the formatted observational info.

    """
    # read info
    ctlinfo = json.loads(hdus['backend'].header['ctlinfo'])
    obsinfo = json.loads(hdus['backend'].header['obsinfo'])
    datinfo = hdus['backend'].data

    # bintable HDU
    N = obsinfo['iary_num']
    p = fm.utils.DatetimeParser()
    flag = np.array(obsinfo['iary_usefg'], dtype=bool)

    fmts = yaml.load(get_data('fmflow', 'fits/data/obsinfo.yaml'))
    names, dtypes, units = list(map(list, zip(*fmts)))
    tforms = list(map(fm.utils.dtype_to_tform, dtypes))

    header = fits.Header()
    header['extname']  = 'OBSINFO'
    header['fitstype'] = 'FMFITS'
    header['telescop'] = 'NRO45m'
    header['date-obs'] = p(obsinfo['clog_id'])[:-3]
    header['observer'] = obsinfo['cobs_user']
    header['object']   = obsinfo['cobj_name']
    header['ra']       = obsinfo['dsrc_pos'][0][0]
    header['dec']      = obsinfo['dsrc_pos'][1][0]
    header['equinox']  = float(re.findall('\d+', obsinfo['cepoch'])[0])
    header['fmflow']   = fm.__version__

    data = OrderedDict()
    data['arrayid']   = np.unique(datinfo['arrayid'])
    data['sideband']  = np.array(obsinfo['csid_type'])[flag]
    data['frontend']  = np.array(obsinfo['cfe_type'])[flag]
    data['backend']   = np.tile(ctlinfo['cbe_type'], N)
    data['numchan']   = np.array(obsinfo['ichannel'])[flag]
    data['restchan']  = np.array(obsinfo['ichannel'])[flag]/2 - 0.5
    data['restfreq']  = np.array(obsinfo['dcent_freq'])[flag]
    data['intmfreq']  = np.array(obsinfo['dflif'])[flag]
    data['bandwidth'] = np.array(obsinfo['dbebw'])[flag]
    data['chanwidth'] = np.array(obsinfo['dbechwid'])[flag]
    data['interval']  = np.tile(obsinfo['diptim'], N)
    data['integtime'] = np.tile(obsinfo['diptim']*EFF_8257D, N)
    data['beamsize']  = np.rad2deg(1.2*C/D_NRO45m) / data['restfreq']

    cols = [fits.Column(n, tforms[i], units[i], array=data[n]) for i, n in enumerate(names)]
    return fits.BinTableHDU.from_columns(cols, header)

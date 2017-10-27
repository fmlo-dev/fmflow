# coding: utf-8

# public items
__all__ = ['fromnro45m']

# standard library
import json
import re
from collections import OrderedDict
from pathlib import Path
from pkgutil import get_data

# dependent packages
import yaml
import fmflow as fm
import numpy as np
from astropy import constants
from astropy import coordinates
from astropy import units as u
from astropy.io import fits
from tqdm import tqdm

# module constants
C                = constants.c.value # spped of light in vacuum
D_NRO45m         = (45.0 * u.m).value # diameter of the NRO45m
LON_NRO45m       = coordinates.Angle('+138d28m21.2s').deg # longitude of the NRO45m
LAT_NRO45m       = coordinates.Angle('+35d56m40.9s').deg # latitude of the NRO45m
EFF_8257D        = 0.92 # exposure / interval time of Agilent 8257D
IGNORED_KEY      = '^reserve' # reserved[1|4|8]
CONF_OBSINFO     = get_data('fmflow', 'data/obsinfo.yaml')
CONF_FMLOLOG     = get_data('fmflow', 'data/nro45m_fmlolog.yaml')
CONF_BACKEND_COM = get_data('fmflow', 'data/nro45m_backendlog_common.yaml')
CONF_BACKEND_SAM = get_data('fmflow', 'data/nro45m_backendlog_sam45.yaml')
CONF_ANTENNA     = get_data('fmflow', 'data/nro45m_antennalog.yaml')
BAR_FORMAT       = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

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
        raise ValueError('invalid logging type')

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
    # path
    fmlolog = Path(fmlolog).expanduser()

    # read fmlolog
    fmts = yaml.load(CONF_FMLOLOG)
    names, dtypes, units = list(map(list, zip(*fmts)))
    tforms = list(map(fm.utils.dtype_to_tform, dtypes))

    c = {0: fm.utils.DatetimeParser()}
    data = np.genfromtxt(fmlolog, list(zip(names, dtypes)), skip_header=1, converters=c)

    # bintable HDU
    header = fits.Header()
    header['extname'] = 'FMLOLOG'
    header['filename'] = str(fmlolog)

    cols = [fits.Column(n, tforms[i], units[i], array=data[n]) for i, n in enumerate(names)]
    return fits.BinTableHDU.from_columns(cols, header)


def read_antennalog(antennalog):
    """Read an antenna logging of NRO45m.

    Args:
        antennalog (str): File name of antenna logging.

    Returns:
        hdu (BinTableHDU): HDU containing the read antenna logging.

    """
    # path
    antennalog = Path(antennalog).expanduser()

    # read antennalog
    fmts = yaml.load(CONF_ANTENNA)
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
    header['filename'] = str(antennalog)

    cols = [fits.Column(n, tforms[i], units[i], array=d[n]) for i, n in enumerate(names)]
    cols.append(fits.Column('az', 'D', 'deg', array=d['az_real']))
    cols.append(fits.Column('el', 'D', 'deg', array=d['el_real']))
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
    # path
    backendlog = Path(backendlog).expanduser()

    com = yaml.load(CONF_BACKEND_COM)
    head = fm.utils.CStructReader(com['head'], IGNORED_KEY, byteorder)
    ctl  = fm.utils.CStructReader(com['ctl'], IGNORED_KEY, byteorder)

    # read backendlog
    with backendlog.open('rb') as f:
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
    # path
    backendlog = Path(backendlog).expanduser()

    com = yaml.load(CONF_BACKEND_COM)
    sam = yaml.load(CONF_BACKEND_SAM)
    head = fm.utils.CStructReader(com['head'], IGNORED_KEY, byteorder)
    ctl  = fm.utils.CStructReader(com['ctl'], IGNORED_KEY, byteorder)
    obs  = fm.utils.CStructReader(sam['obs'], IGNORED_KEY, byteorder)
    dat  = fm.utils.CStructReader(sam['dat'], IGNORED_KEY, byteorder)

    def eof(f):
        head.read(f)
        bar.update(head.size)
        return (head._data['crec_type'][-1][0] == b'ED')

    # read backendlog
    with backendlog.open('rb') as f:
        total = backendlog.stat().st_size
        with tqdm(total=total, bar_format=BAR_FORMAT) as bar:
            eof(f)
            ctl.read(f)
            bar.update(ctl.size)
            eof(f)
            obs.read(f)
            bar.update(obs.size)

            while not eof(f):
                dat.read(f)
                bar.update(dat.size)

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
    names  = list(dat.ctypes.keys())
    ctypes = list(dat.ctypes.values())
    shapes = list(dat.shapes.values())
    tforms = map(fm.utils.ctype_to_tform, ctypes, shapes)
    fmts = OrderedDict(item for item in zip(names, tforms))

    fmts['starttime'] = 'A26'
    fmts['arraydata'] = fmts.pop('fary_data')
    fmts['arrayid']   = fmts.pop('cary_name')
    fmts['scantype']  = fmts.pop('cscan_type')

    # bintable HDU
    header = fits.Header()
    header['extname']  = 'BACKEND'
    header['filename'] = str(backendlog)
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

    fmts = yaml.load(CONF_OBSINFO)
    names, dtypes, units = list(map(list, zip(*fmts)))
    tforms = list(map(fm.utils.dtype_to_tform, dtypes))

    header = fits.Header()
    header['extname']  = 'OBSINFO'
    header['fitstype'] = 'FMFITSv0'
    header['telescop'] = 'NRO45m'
    header['sitelon']  = LON_NRO45m
    header['sitelat']  = LAT_NRO45m
    header['date-obs'] = p(obsinfo['clog_id'])[:-3]
    header['observer'] = obsinfo['cobs_user']
    header['object']   = obsinfo['cobj_name']
    header['ra']       = obsinfo['dsrc_pos'][0][0]
    header['dec']      = obsinfo['dsrc_pos'][1][0]
    header['equinox']  = float(re.findall('\d+', obsinfo['cepoch'])[0])

    data = OrderedDict()
    data['arrayid']   = np.unique(datinfo['arrayid'])
    data['sideband']  = np.array(obsinfo['csid_type'])[flag]
    data['frontend']  = np.array(obsinfo['cfe_type'])[flag]
    data['backend']   = np.tile(ctlinfo['cbe_type'], N)
    data['offsetaz']  = np.tile(0.0, N) # not implemented yet
    data['offsetel']  = np.tile(0.0, N) # not implemented yet
    data['chtotaln']  = np.array(obsinfo['ichannel'])[flag]
    data['chcenter']  = data['chtotaln']/2 + 0.5
    data['rfcenter']  = np.array(obsinfo['dcent_freq'])[flag]
    data['ifcenter']  = np.array(obsinfo['dflif'])[flag]
    data['ifcenter']  += np.array(obsinfo['dcent_freq'])[flag]
    data['ifcenter']  -= np.array(obsinfo['dtrk_freq'])[flag]
    data['chwidth']   = np.array(obsinfo['dbechwid'])[flag]
    data['bandwidth'] = np.array(obsinfo['dbebw'])[flag]
    data['interval']  = np.tile(obsinfo['diptim'], N)
    data['integtime'] = data['interval'] * EFF_8257D
    data['beamsize']  = np.rad2deg(1.2*C/D_NRO45m) / data['rfcenter']

    cols = [fits.Column(n, tforms[i], units[i], array=data[n]) for i, n in enumerate(names)]
    return fits.BinTableHDU.from_columns(cols, header)

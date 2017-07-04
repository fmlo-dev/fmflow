# coding: utf-8

# imported items
__all__ = ['getarray']

# standard library
import os
from datetime import timedelta
from functools import reduce

# dependent packages¬
import numpy as np
import fmflow as fm
from astropy.io import fits


# functions
def getarray(fitsname, arrayid, scantype, offsetsec=0.0):
    """Create a modulated array from a FMFITS.

    Args:
        fitsname (str): File name of a FMFITS.
        arrayid (str): An array ID with which the output fmarray is created.
        scantype (str): A scan type with which the output fmarray is created.
        offsetsec (float, optional): A float value of FM offset time in units of sec.

    Returns:
        array (xarray.DataArray): A modulated array of the spacified `arrayid` and `scantype`.

    """
    with fits.open(os.path.expanduser(fitsname)) as f:
        # fits data
        fmlo = f['fmlolog'].data
        be = f['backend'].data
        if 'antenna' in f:
            ant = f['antenna'].data

        # info for *coords
        d = f['obsinfo'].data
        info = dict(zip(d.names, d[d['arrayid']==arrayid][0]))
        info.update(f['obsinfo'].header)

        # ptcoords
        ptcoords = {
            'xref': info['RA'],
            'yref': info['DEC'],
            'coordsys': 'RADEC',
            'status': 'MODULATED',
        }

        # chcoords
        step = info['chanwidth']
        start = info['restfreq'] - step*info['restchan']
        end = start + step*info['numchan']

        chcoords = {
            'fsig': np.arange(start, end, step),
            'fimg': np.arange(start, end, step)[::-1] - 2*info['intmfreq'],
        }

        # tcoords and flags
        tcoords = {}

        if scantype == 'ON':
            t, flag_fmlo, flag_be, flag_ant = makeflags(f, arrayid, scantype, offsetsec)

            tcoords.update({
                'fmch': (fmlo['FMFREQ'][flag_fmlo]/step).astype(int),
                'vrad': fmlo['VRAD'][flag_fmlo],
                'time': t,
            })

            if 'antenna' in f:
                tcoords.update({
                    'xrel': ant['RA'][flag_ant] - ptcoords['xref'],
                    'yrel': ant['DEC'][flag_ant] - ptcoords['yref'],
                })
        else:
            flag_be = (be['arrayid']==arrayid) & (be['scantype']==scantype)

        # data
        data = be['arraydata'][flag_be]
        return fm.array(data, tcoords, chcoords, ptcoords).squeeze()


def flag(f, arrayid, scantype, offsetsec=0.0):
    p = fm.utils.DatetimeParser(False)
    c = lambda dt: np.vectorize(p)(np.asarray(dt))
    t_list = []

    # fmlolog
    fmlo = f['fmlolog'].data
    t_fmlo = c(fmlo['starttime']) + timedelta(seconds=offsetsec)
    f_fmlo = (fmlo['scantype']==scantype)
    t_list.append(t_fmlo[f_fmlo])

    # backend
    be = f['backend'].data
    t_be = c(be['starttime'])
    f_be = (be['scantype']==scantype) & (be['arrayid']==arrayid)
    t_list.append(t_be[f_be])

    # antenna
    if 'antenna' in f:
        ant = f['antenna'].data
        t_ant = c(ant['starttime'])
        t_list.append(t_ant)

    # flags
    t_com = reduce(lambda t, s: np.intersect1d(t, s), t_list)
    flag_fmlo = np.in1d(t_fmlo, t_com) & f_fmlo
    flag_be   = np.in1d(t_be, t_com) & f_be
    if 'antenna' in f:
        flag_ant = np.in1d(t_ant, t_com)
        return t_com, flag_fmlo, flag_be, flag_ant
    else:
        return t_com, flag_fmlo, flag_be, None

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
        be   = f['backend'].data
        if 'antenna' in f:
            ant = f['antenna'].data

        # data other than ON (e.g. R, SKY, ZERO)
        if not scantype == 'ON':
            flag_be = (be['arrayid']==arrayid) & (be['scantype']==scantype)
            return np.squeeze(be['arraydata'][flag_be])

        # fmarray data
        t_com, flag_fmlo, flag_be, flag_ant = flag(f, arrayid, scantype, offsetsec)
        data = np.squeeze(be['arraydata'][flag_be])

        # fmarray coords
        info = f['obsinfo'].data
        flag_info = (info['arrayid']==arrayid)
        info = dict(zip(info.names, info[flag_info][0]))
        info.update(f['obsinfo'].header)

        tcoords = {'time': t_com}
        chcoords = {}
        ptcoords = {'status': 'MODULATED'}

        ## fmch, vrad
        fmfreq = fmlo['FMFREQ'][flag_fmlo]
        vrad = fmlo['VRAD'][flag_fmlo]
        fmch = (fmfreq / info['chanwidth']).astype(int)
        tcoords.update({'fmch': fmch, 'vrad': vrad})

        ## x/yrel, x/yref (if any)
        if 'antenna' in f:
            xref, yref = info['RA'], info['DEC']
            xrel = ant['RA'][flag_ant] - xref
            yrel = ant['DEC'][flag_ant] - yref
            tcoords.update({'xrel': xrel, 'yrel': yrel})
            ptcoords.update({'xref': xref, 'yref': yref, 'coordsys': 'RADEC'})

        ## fsig, fimg
        step = info['chanwidth']
        start = info['restfreq'] - info['restchan']*step
        end = start + info['numchan']*step
        fsig = np.arange(start, end, step)
        fimg = fsig[::-1] - 2*info['intmfreq']
        chcoords.update({'fsig': fsig, 'fimg': fimg})

        return fm.array(data, tcoords, chcoords, ptcoords)


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

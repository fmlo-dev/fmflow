# coding: utf-8


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

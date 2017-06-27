# coding: utf-8

# imported items
__all__ = [
    'OzoneLines',
]

# standard library
import os
from subprocess import PIPE, run

# dependent packages
import yaml
import fmflow as fm
import numpy as np
from astropy import constants
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.ndimage import filters
from scipy.optimize import curve_fit

# constants
C = constants.c.value
DATA_DIR = os.path.join(fm.__path__[0], 'models', 'data')
with open(os.path.join(DATA_DIR, 'am.yaml')) as f:
    d = yaml.load(f)
    AMCONFIG = d['amc']
    AMLAYERS = d['layers']


# classes
class OzoneLines(object):
    info = {
        'amconfig': AMCONFIG,
        'amlayers': AMLAYERS,
        'computed': False,
    }
    freq = None
    taus = None
    Tbs  = None

    def __init__(self, fitmode='normal', smooth=50):
        self.info = {
            'fitmode': fitmode,
            'smooth': smooth,
        }
        self._setclassattrs()

    def fit(self, array, weights=None):
        if not self.computed:
            self.compute(array)

        freq = fm.getfreq(array, unit='GHz')
        spec = fm.getspec(array, weights=weights)
        vrad = array.vrad.values.mean() # m/s
        frad = freq * vrad/C # GHz

        if self.fitmode == 'normal':
            model = self._fit(freq-frad, spec)
        elif self.fitmode == 'diff':
            model = self._dfit(freq-frad, spec, smooth)
        else:
            raise fm.utils.FMFlowError('invalid mode')

        array_ = fm.demodulate(array)
        return fm.modulate(fm.zeros_like(array_)+model)

    def compute(self, array):
        freq = fm.getfreq(array, unit='GHz') # GHz
        step = 1e3 * np.diff(freq).mean() # MHz

        fmin = np.floor(np.min(freq))
        fmax = np.ceil(np.max(freq))
        fstep  = float('{:.0e}'.format(0.5*step))
        params = {'fmin': fmin, 'fmax': fmax, 'fstep': fstep}

        amfreq = None
        amtaus, amTbs = [], []
        N = len(self.amlayers)
        for n in range(N):
            fm.utils.progressbar((n+1)/N)

            params.update(**self.amlayers[n])
            amc = self.amconfig.format(**params)

            cp = run(['am', '-'], input=amc.encode('utf-8'), stdout=PIPE)
            stdout = cp.stdout.decode('utf-8')
            output = np.loadtxt(stdout.split('\n'))

            if amfreq is None:
                amfreq = output[:, 0]

            amtaus.append(output[:, 1])
            amTbs.append(output[:, 2])

        OzoneLines.freq = amfreq
        OzoneLines.taus = np.array(amtaus)
        OzoneLines.Tbs  = np.array(amTbs) - 2.7
        OzoneLines.info['computed'] = True
        self._getclassattrs()

    def _setclassattrs(self):
        self.info.update(OzoneLines.info)
        self.freq = OzoneLines.freq
        self.taus = OzoneLines.taus
        self.Tbs  = OzoneLines.Tbs

    def _fit(self, freq, spec):
        Tbs = interp1d(self.freq, self.Tbs, axis=1)(freq)

        def func(freq, *coeffs):
            coeffs = np.asarray(coeffs)[:,np.newaxis]
            return np.sum(coeffs * Tbs, 0)

        p0 = np.full(len(Tbs), 0.5)
        bs = (0.0, 1.0)
        popt, pcov = curve_fit(func, freq, spec, p0, bounds=bs)
        return func(freq, *popt)

    def _dfit(self, freq, spec, smooth):
        Tbs = interp1d(self.freq, self.Tbs, axis=1)(freq)
        dTbs = np.gradient(Tbs, axis=1)

        dspec = np.gradient(spec)
        dspec -= filters.gaussian_filter(dspec, self.smooth)

        def func(freq, *coeffs):
            coeffs = np.asarray(coeffs)[:,np.newaxis]
            return np.sum(coeffs * dTbs, 0)

        p0 = np.zeros(len(dTbs))
        bs = (0.0, 1.0)
        popt, pcov = curve_fit(func, freq, dspec, p0, bounds=bs)
        return np.cumsum(func(freq, *popt))

    def __getattr__(self, name):
        return self.info[name]

    def __repr__(self):
        string = str.format(
            'OzoneLines(fitmode={0}, computed={1})',
            self.fitmode,
            self.computed,
        )

        return string

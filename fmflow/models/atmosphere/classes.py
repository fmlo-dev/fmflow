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
from scipy.optimize import curve_fit

# constants
C = constants.c.value
DATA_DIR = os.path.join(fm.__path__[0], 'models', 'data')

AMCMD = ['am', '-']
with open(os.path.join(DATA_DIR, 'am.yaml')) as f:
    am = yaml.load(f)
    AMCONFIG = am['config']
    AMLAYERS = am['layers']


# classes
class OzoneLines(object):
    params = {
        'amconfig': AMCONFIG,
        'amlayers': AMLAYERS,
        'computed': False,
    }
    freq = None
    taus = None
    tbs  = None

    def __init__(self):
        self._setclassattrs()

    def fit(self, freq, spec, vrad=0.0):
        self.compute(freq)
        frad = np.median(freq) * vrad/C
        return self._fit(freq-frad, spec)

    def generate(self, freq, vrad=0.0):
        self.compute(freq)
        frad = np.median(freq) * vrad/C
        return self._generate(freq-frad)

    def compute(self, freq, forcibly=False):
        if (not self.computed) or (self.computed and forcibly):
            self._compute(freq)

    def _fit(self, freq, spec):
        tbs = interp1d(self.freq, self.tbs, axis=1)(freq)

        def func(freq, *coeffs):
            coeffs = np.asarray(coeffs)[:,np.newaxis]
            return (coeffs*tbs).sum(0)

        p0 = np.full(len(tbs), 0.5)
        bs = (0.0, 1.0)
        popt, pcov = curve_fit(func, freq, spec, p0, bounds=bs)
        return func(freq, *popt)

    def _generate(self, freq):
        tbs = interp1d(self.freq, self.tbs, axis=1)(freq)
        return tbs.sum(0)

    def _compute(self, freq):
        params = {
            'fmin': np.floor(np.min(freq)),
            'fmax': np.ceil(np.max(freq)),
            'fstep': float('{:.0e}'.format(0.5*np.diff(freq).mean())),
        }

        amfreq = None
        amtaus, amtbs = [], []

        N = len(self.amlayers)
        for n in range(N):
            fm.utils.progressbar((n+1)/N)

            params.update(**self.amlayers[n])
            amc = self.amconfig.format(**params)

            cp = run(AMCMD, input=amc.encode('utf-8'), stdout=PIPE)
            stdout = cp.stdout.decode('utf-8')
            output = np.loadtxt(stdout.split('\n'))

            if amfreq is None:
                amfreq = output[:, 0]

            amtaus.append(output[:, 1])
            amtbs.append(output[:, 2])

        OzoneLines.freq = amfreq
        OzoneLines.taus = np.array(amtaus)
        OzoneLines.tbs  = np.array(amtbs) - 2.7
        OzoneLines.params['computed'] = True
        self._setclassattrs()

    def _setclassattrs(self):
        self.params = OzoneLines.params
        self.freq = OzoneLines.freq
        self.taus = OzoneLines.taus
        self.tbs  = OzoneLines.tbs

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'OzoneLines(fitmode={0}, computed={1})',
            self.fitmode, self.computed,
        )

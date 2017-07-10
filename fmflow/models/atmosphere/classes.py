# coding: utf-8

# imported items
__all__ = [
    'AtmosLines',
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
class AtmosLines(object):
    amconfig = AMCONFIG
    amlayers = AMLAYERS
    computed = False
    freq = None
    taus = None
    tbs  = None
    tbbg = 2.7

    def __init__(self, ch_tolerance=5):
        self.params = {
            'ch_tolerance': ch_tolerance,
        }

    def fit(self, freq, spec, vrad=0.0, *, forcibly=False):
        self.compute(freq, forcibly=forcibly)
        frad = np.median(freq) * vrad/C
        fstep = np.diff(freq).mean()

        amspecs = []
        for ch in range(-self.ch_tolerance, self.ch_tolerance+1):
            amspecs.append(self._fit(freq-frad-ch*fstep, spec))

        amspecs = np.array(amspecs)
        bestfit = np.argmin(np.sum((amspecs-spec)**2, 1))
        return amspecs[bestfit]

    def generate(self, freq, vrad=0.0, coeffs=None, *, forcibly=False):
        self.compute(freq, forcibly=forcibly)
        frad = np.median(freq) * vrad/C
        return self._generate(freq-frad, coeffs)

    @classmethod
    def _fit(cls, freq, spec):
        tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)

        def func(freq, *coeffs):
            coeffs = np.asarray(coeffs)[:,np.newaxis]
            return (coeffs*tbs).sum(0)

        p0 = np.full(len(tbs), 0.5)
        bs = (0.0, 2.0)
        popt, pcov = curve_fit(func, freq, spec, p0, bounds=bs)
        return func(freq, *popt)

    @classmethod
    def _generate(cls, freq, coeffs=None):
        tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)

        if coeffs is None:
            coeffs = np.ones(len(cls.tbs), dtype=float)

        return (coeffs * tbs).sum(0)

    @classmethod
    def compute(cls, freq, *, forcibly=False):
        if forcibly or (not cls.computed):
            params = {
                'fmin': np.floor(np.min(freq)),
                'fmax': np.ceil(np.max(freq)),
                'fstep': float('{:.0e}'.format(0.5*np.diff(freq).mean())),
            }

            amfreq = None
            amtaus, amtbs = [], []

            N = len(cls.amlayers)
            for n in range(N):
                fm.utils.progressbar((n+1)/N)

                params.update(**cls.amlayers[n])
                amc = cls.amconfig.format(**params)

                cp = run(AMCMD, input=amc.encode('utf-8'), stdout=PIPE)
                stdout = cp.stdout.decode('utf-8')
                output = np.loadtxt(stdout.split('\n'))

                if amfreq is None:
                    amfreq = output[:, 0]

                amtaus.append(output[:, 1])
                amtbs.append(output[:, 2])

            cls.freq = amfreq
            cls.taus = np.array(amtaus)
            cls.tbs  = np.array(amtbs) - cls.tbbg
            cls.computed = True

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'OzoneLines(ch_tolerance={0})',
            self.ch_tolerance
        )

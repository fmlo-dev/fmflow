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
    coeffs = None
    freq = None
    taus = None
    tbs = None

    def __init__(self, ch_tolerance=5, *, logger=None):
        self.params = {
            'ch_tolerance': ch_tolerance,
        }

        self.logger = logger or fm.logger

    def fit(self, freq, spec, vrad=0.0, *, forcibly=False):
        self._compute(freq, forcibly=forcibly, logger=self.logger)
        frad = np.median(freq) * vrad/C
        fstep = np.diff(freq).mean()

        amfreqs = []
        amspecs = []
        for ch in range(-self.ch_tolerance, self.ch_tolerance+1):
            amfreq = freq - frad - ch*fstep
            self._fit(amfreq, spec)
            amspec = self._generate(amfreq)
            amfreqs.append(amfreq)
            amspecs.append(amspec)

        amspecs = np.array(amspecs)
        index = np.argmin(np.sum((amspecs-spec)**2, 1))
        self._fit(amfreqs[index], spec)
        return amspecs[index]

    def generate(self, freq, vrad=0.0, coeffs=None, *, forcibly=False):
        self._compute(freq, forcibly=forcibly, logger=self.logger)
        frad = np.median(freq) * vrad/C
        return self._generate(freq-frad, coeffs)

    @classmethod
    def _fit(cls, freq, spec):
        tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)

        def func(freq, *coeffs):
            coeffs = np.asarray(coeffs)
            return (coeffs[:, np.newaxis]*tbs).sum(0)

        p0, bounds = np.full(len(tbs), 0.5), (0.0, 1.0)
        cls.coeffs = curve_fit(func, freq, spec, p0, bounds=bounds)[0]

    @classmethod
    def _generate(cls, freq, coeffs=None):
        tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)

        if coeffs is None:
            if cls.coeffs is None:
                coeffs = np.ones(len(cls.tbs), dtype=float)
            else:
                coeffs = cls.coeffs
        else:
            coeffs = np.asarray(coeffs)

        return (coeffs[:, np.newaxis]*tbs).sum(0)

    @classmethod
    def _compute(cls, freq, *, forcibly=False, logger=None):
        logger = fm.logger if logger is None else logger

        if forcibly or (not cls.computed):
            params = {
                'fmin': np.floor(np.min(freq)),
                'fmax': np.ceil(np.max(freq)),
                'fstep': float('{:.0e}'.format(0.5*np.diff(freq).mean())),
            }

            logger.info('computing am')
            logger.info('this may take several minutes')
            logger.debug(params)

            amtaus, amtbs = [], []
            N = len(cls.amlayers)
            for n in range(N):
                logger.debug('computing layer {0}/{1}'.format(n+1, N))

                params.update(**cls.amlayers[n])
                amc = cls.amconfig.format(**params)

                cp = run(AMCMD, input=amc.encode('utf-8'), stdout=PIPE)
                stdout = cp.stdout.decode('utf-8')
                output = np.loadtxt(stdout.split('\n'))

                amtaus.append(output[:, 1])
                amtbs.append(output[:, 2])

            cls.computed = True
            cls.freq = output[:, 0]
            cls.taus = np.array(amtaus)
            cls.tbs = np.array(amtbs) - 2.7
            logger.info('computing finished')

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return str.format(
            'OzoneLines(ch_tolerance={0})',
            self.ch_tolerance
        )

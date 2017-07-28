# coding: utf-8

# public items
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

# module constants
C = constants.c.value
DATA_DIR = os.path.join(fm.__path__[0], 'models', 'data')

# am settings
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
    tbs = None

    def __init__(self, ch_tolerance=5, *, logger=None):
        self.params = {
            'ch_tolerance': ch_tolerance,
        }

        self.logger = logger or fm.logger

    def fit(self, freq, spec, vrad=0.0):
        frad = np.median(freq) * vrad/C
        fstep = np.diff(freq).mean()

        taus, tbs = [], []
        for ch in range(-self.ch_tolerance, self.ch_tolerance+1):
            tau, tb = self._fit(freq-frad-ch*fstep, spec)
            taus.append(tau)
            tbs.append(tb)

        index = np.argmin(np.sum((np.array(tbs)-spec)**2, 1))
        return taus[index], tbs[index]

    def generate(self, freq, vrad=0.0):
        frad = np.median(freq) * vrad/C
        return self._generate(freq-frad)

    def _fit(self, freq, spec):
        try:
            taus = interp1d(self.freq, self.taus, axis=1)(freq)
            tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)
        except:
            self._compute(freq, logger=self.logger)
            taus = interp1d(self.freq, self.taus, axis=1)(freq)
            tbs = interp1d(self.freq, self.tbs, axis=1)(freq)

        def f_tau(freq, *coeffs):
            coeffs = np.asarray(coeffs)
            return (coeffs[:, np.newaxis]*taus).sum(0)

        def f_tb(freq, *coeffs):
            coeffs = np.asarray(coeffs)
            return (coeffs[:, np.newaxis]*tbs).sum(0)

        p0, bounds = np.full(len(tbs), 0.5), (0.0, 1.0)
        coeffs = curve_fit(f_tb, freq, spec, p0, bounds=bounds)[0]
        return f_tau(freq, *coeffs), f_tb(freq, *coeffs)

    def _generate(self, freq):
        try:
            taus = interp1d(self.freq, self.taus, axis=1)(freq)
            tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)
        except:
            self._compute(freq, logger=self.logger)
            taus = interp1d(self.freq, self.taus, axis=1)(freq)
            tbs = interp1d(self.freq, self.tbs, axis=1)(freq)

        return taus.sum(0), tbs.sum(0)

    @classmethod
    def _compute(cls, freq, *, logger=None):
        logger = logger or fm.logger

        if not cls.computed:
            params = {
                'fmin': np.min(freq) - 0.1*np.ptp(freq),
                'fmax': np.max(freq) + 0.1*np.ptp(freq),
                'fstep': 0.5*np.mean(np.diff(freq)),
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
        return 'AtmosLines({0})'.format(self.params)

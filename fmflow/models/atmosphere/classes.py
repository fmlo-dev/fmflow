# coding: utf-8

# public items
__all__ = [
    'AtmosLines',
]

# standard library
import os
import warnings
from subprocess import PIPE, run

# dependent packages
import yaml
import fmflow as fm
import numpy as np
from astropy import constants
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)

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
    freq = None
    taus = None
    tbs = None

    def __init__(self, snr_threshold=5, ch_tolerance=5, *, logger=None):
        self.params = {
            'snr_threshold': snr_threshold,
            'ch_tolerance': ch_tolerance,
        }

        self.logger = logger or fm.logger

    def fit(self, freq, spec, noise, vrad=0.0):
        frad = np.median(freq) * vrad/C
        fstep = np.diff(freq).mean()

        tbs = []
        for ch in range(-self.ch_tolerance, self.ch_tolerance+1):
            _freq = freq - frad - ch*fstep
            tbs.append(self._fit(_freq, spec, logger=self.logger))

        tb = tbs[np.argmin(((tbs-spec)**2).sum(1))]

        if np.max(tb/noise) < self.snr_threshold:
            return np.zeros_like(tb)
        else:
            return tb

    def generate(self, freq, vrad=0.0):
        frad = np.median(freq) * vrad/C
        return self._generate(freq-frad, logger=self.logger)

    @classmethod
    def _fit(cls, freq, spec, *, logger=None):
        try:
            tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)
        except:
            cls._compute(freq, logger=logger)
            tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)

        def func(freq, *coeffs):
            coeffs = np.asarray(coeffs)
            return (coeffs[:, np.newaxis]*tbs).sum(0)

        p0, bounds = np.full(len(tbs), 0.5), (0.0, 1.0)
        coeffs = curve_fit(func, freq, spec, p0, bounds=bounds)
        return func(freq, *coeffs[0])

    @classmethod
    def _generate(cls, freq, *, logger=None):
        try:
            tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)
        except:
            cls._compute(freq, logger=logger)
            tbs = interp1d(cls.freq, cls.tbs, axis=1)(freq)

        return tbs.sum(0)

    @classmethod
    def _compute(cls, freq, *, logger=None):
        params = {
            'fmin': np.min(freq) - 0.1*np.ptp(freq),
            'fmax': np.max(freq) + 0.1*np.ptp(freq),
            'fstep': np.abs(0.5*np.mean(np.diff(freq))),
        }

        logger.info('computing am')
        logger.info('this may take several minutes')
        logger.debug(params)

        amtaus, amtbs = [], []
        n_layers = len(cls.amlayers)
        for n in range(n_layers):
            logger.debug('computing am layer {0}/{1}'.format(n+1, n_layers))

            params.update(**cls.amlayers[n])
            amc = cls.amconfig.format(**params)
            cp = run(AMCMD, input=amc.encode('utf-8'), stdout=PIPE)

            stdout = cp.stdout.decode('utf-8')
            output = np.loadtxt(stdout.split('\n'))
            amtaus.append(output[:, 1])
            amtbs.append(output[:, 2])

        cls.freq = output[:, 0]
        cls.taus = np.array(amtaus)
        cls.tbs = np.array(amtbs) - 2.7
        logger.info('computing finished')

    def __getattr__(self, name):
        return self.params[name]

    def __repr__(self):
        return 'AtmosLines({0})'.format(self.params)

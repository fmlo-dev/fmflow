# coding: utf-8

# public items
__all__ = [
    'AtmosLines',
]

# standard library
import warnings
from subprocess import PIPE, run
from pkgutil import get_data

# dependent packages
import yaml
import fmflow as fm
import numpy as np
from .. import BaseModel
from astropy import constants
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)

# module constants
C = constants.c.value
AMCMD = ['am', '-']
AMDATA = get_data('fmflow', 'data/am.yaml')
AMCONFIG = yaml.load(AMDATA)['config']
AMLAYERS = yaml.load(AMDATA)['layers']


# classes
class AtmosLines(BaseModel):
    amconfig = AMCONFIG
    amlayers = AMLAYERS
    amfreqs  = []
    amtaus   = []
    amtbs    = []

    def __init__(self, snr_threshold=10, ch_tolerance=5, *, logger=None):
        params = {
            'snr_threshold': snr_threshold,
            'ch_tolerance': ch_tolerance,
        }
        super().__init__(params, logger)


    def fit(self, freq, spec, noise, vrad=0.0, freqlim=None):
        freq  = np.asarray(freq)
        spec  = np.asarray(spec)
        noise =  np.asarray(noise)

        # freq limits
        if freqlim is not None:
            spec[freq<freqlim[0]] = 0
            spec[freq>freqlim[1]] = 0

        spec[np.isnan(spec)] = 0
        noise[np.isnan(noise)] = np.inf

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
        freq  = np.asarray(freq)

        frad = np.median(freq) * vrad/C
        return self._generate(freq-frad, logger=self.logger)

    @classmethod
    def _fit(cls, freq, spec, *, logger=None):
        for amfreq, amtb in zip(cls.amfreqs, cls.amtbs):
            try:
                tb = interp1d(amfreq, amtb, axis=1)(freq)
                break
            except:
                continue
        else:
            cls._compute(freq, logger=logger)
            amfreq, amtb = cls.amfreqs[-1], cls.amtbs[-1]
            tb = interp1d(amfreq, amtb, axis=1)(freq)

        def func(freq, *coeffs):
            coeffs = np.asarray(coeffs)
            return (coeffs[:, np.newaxis]*tb).sum(0)

        p0, bounds = np.full(len(tb), 0.5), (0.0, 1.0)
        coeffs = curve_fit(func, freq, spec, p0, bounds=bounds)
        return func(freq, *coeffs[0])

    @classmethod
    def _generate(cls, freq, *, logger=None):
        for amfreq, amtb in zip(cls.amfreqs, cls.amtbs):
            try:
                tb = interp1d(amfreq, amtb, axis=1)(freq)
                break
            except:
                continue
        else:
            cls._compute(freq, logger=logger)
            amfreq, amtb = cls.amfreqs[-1], cls.amtbs[-1]
            tb = interp1d(amfreq, amtb, axis=1)(freq)

        return tb.sum(0)

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

        amtau, amtb = [], []
        n_layers = len(cls.amlayers)
        for n in range(n_layers):
            logger.debug('computing am layer {0}/{1}'.format(n+1, n_layers))

            params.update(**cls.amlayers[n])
            amc = cls.amconfig.format(**params)
            cp = run(AMCMD, input=amc.encode('utf-8'), stdout=PIPE)

            stdout = cp.stdout.decode('utf-8')
            output = np.loadtxt(stdout.split('\n'))
            amtau.append(output[:, 1])
            amtb.append(output[:, 2])

        amfreq = output[:, 0]
        amtau  = np.array(amtau)
        amtb   = np.array(amtb) - 2.7
        amtb  -= amtb.min(1)[:, np.newaxis]

        cls.amfreqs.append(amfreq)
        cls.amtaus.append(amtau)
        cls.amtbs.append(amtb)
        logger.info('computing finished')

# coding: utf-8

# standard library
from setuptools import setup

# module constants
REQUIRES = ['astropy',
            'numba',
            'numpy',
            'pyyaml',
            'scikit-learn',
            'scipy',
            'tqdm',
            'xarray']

PACKAGES = [
    'fmflow',
    'fmflow.core',
    'fmflow.core.array',
    'fmflow.core.cube',
    'fmflow.core.spectrum',
    'fmflow.fits',
    'fmflow.fits.nro45m',
    'fmflow.fits.aste',
    'fmflow.logging',
    'fmflow.models',
    'fmflow.models.astrosignal',
    'fmflow.models.atmosphere',
    'fmflow.models.commonmode',
    'fmflow.models.gain',
    'fmflow.utils',
    'fmflow.utils.binary',
    'fmflow.utils.convergence',
    'fmflow.utils.datetime',
    'fmflow.utils.fits',
    'fmflow.utils.misc',
    'fmflow.utils.ndarray'
]

# main
setup(
    name = 'fmflow',
    description = 'Data analysis package for FMLO',
    version = '0.2.1',
    author = 'astropenguin',
    author_email = 'taniguchi@a.phys.nagoya-u.ac.jp',
    url = 'https://github.com/fmlo-dev/fmflow',
    install_requires = REQUIRES,
    packages = PACKAGES,
    package_data = {'fmflow': ['data/*.yaml']},
)

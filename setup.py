# coding: utf-8

"Data Analysis Package for FMLO"

# standard library
from setuptools import setup

# module constants
# INSTALL_REQUIRES = [
#     'xarray >= 0.9.6',
# ]

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
    description = __doc__,
    version = '0.1',
    author = 'FMLO software team',
    author_email = 'ataniguchi@ioa.s.u-tokyo.ac.jp',
    url = 'https://github.com/fmlo-dev/fmflow',
    # install_requires = INSTALL_REQUIRES,
    packages = PACKAGES,
    package_data = {'fmflow': ['data/*.yaml']},
)

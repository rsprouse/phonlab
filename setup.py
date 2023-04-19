#!/usr/bin/env python

from distutils.core import setup

setup(
  name = 'phonlab',
  version = '0.10.0',
  packages = ['phonlab', 'phonlab.third_party'],
  install_requires=[
    'importlib_resources; python_version < "3.9"',
    'numpy',
    'pandas',
    'praat-parselmouth',
    'scipy',
  ],
  scripts = [
  ],
  include_package_data=True,
  classifiers = [
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Multimedia :: Sound/Audio :: Speech'
  ]
)

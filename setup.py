#!/usr/bin/env python

from distutils.core import setup

setup(
  name = 'phonlab',
  packages = ['phonlab', 'phonlab.third_party'],
  scripts = [
  ],
  package_data = {'phonlab': ['data/phonemap.txt']},
  classifiers = [
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering'
  ]
)

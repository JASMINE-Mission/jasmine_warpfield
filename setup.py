#!/usr/bin/env python
# -*- coding: utf-8 -*-
from glob import glob
from setuptools import setup
import os,sys,re


with open('README.md', 'r') as fd:
  version = '0.1'
  author = 'Ryou Ohsawa'
  email = 'ohsawa@ioa.s.u-tokyo.ac.jp'
  description = 'An experimental code to simulate a warped focal plane for small-JASMINE.'
  long_description = fd.read()
  license = 'MIT'

classifiers = [
  'Development Status :: 3 - Alpha',
  'Environment :: Console',
  'Intended Audience :: Science/Research',
  'License :: OSI Approved :: MIT License',
  'Operating System :: POSIX :: Linux',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.7',
  'Topic :: Scientific/Engineering :: Astronomy']


if __name__ == '__main__':
  setup(
    name='jasmine_warpfield',
    package_dir={"":"warpfield"},
    version=version,
    author=author,
    author_email=email,
    maintainer=author,
    maintainer_email=email,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/xr0038/jasmine_warpfield',
    license=license,
    classifiers=classifiers,
    install_requires=['numpy','scipy','astropy', 'astroquery'])
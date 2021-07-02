#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import pandas as pd
import astropy.units as u
import warpfield as w

description='''
This script generates Case-5 astrometry challenges. The telescope used
in this challenge is affected by distortion. Astronomical sources are retrived
from the Gaia EDR3 catalog, while observations fields are not in the Galactic
center region. In total, 10 fields are obtained for each distortion pattern.
A list of source positions on the focal plane and original ICRS coordinates
is provided. Solve the position angle and distortion parameters.
'''.strip()



seed = 42
np.random.seed(seed)


def generate_challenge(filename):
  lon = Longitude(np.random.uniform(0,36.0)*u.deg)
  lat = Latitude(np.random.uniform(-90,90)*u.deg)
  center = SkyCoord(lon, lat, frame='galactic')

  scale = np.random.normal(1.00,0.01)
  K = np.random.normal(0,5e1,size=1)
  S = np.random.normal(0,3e0,size=2)
  T = np.random.normal(0,5e1,size=1)
  distortion = w.distortion_generator(K,S,T,scale=scale)

  fields = np.arange(10)
  catalog = []
  tel_ra  = []
  tel_dec = []
  tel_pa  = []
  for n in fields:
    separation = np.random.uniform(0,2)*u.deg
    direction = Angle(np.random.uniform(0,360)*u.deg)
    pa = Angle(np.random.uniform(0,360)*u.deg)
    pointing = center.directional_offset_by(direction, separation)
    jasmine = w.Telescope(pointing, pa)
    jasmine.set_distortion(distortion)

    radius = Angle(0.3*u.deg)
    sources = w.retrieve_gaia_sources(pointing,radius)

    position = jasmine.observe(sources)[0]
    position['field'] = n
    catalog.append(position)
    tel_ra.append(pointing.icrs.ra.deg)
    tel_dec.append(pointing.icrs.ra.deg)
    tel_pa.append(pa.deg)

  catalog = pd.concat(catalog)

  table = QTable(
    [
      catalog.x*u.um,
      catalog.y*u.um,
      catalog.ra*u.deg,
      catalog.dec*u.deg,
      catalog.field,
    ],
    names = ('x','y','ra','dec','field'),
    meta = {
      'keywords': {
        'distortion_K': {'value': tuple(K)},
        'distortion_S': {'value': tuple(S)},
        'distortion_T': {'value': tuple(T)},
        'distortion_scale': {'value': scale},
      },
      'comments': [description,]
    })
  print(table)
  table.write(filename, format='ascii.ipac', overwrite=True)

  table = QTable(
    [
      fields,
      np.array(tel_ra)*u.deg,
      np.array(tel_dec)*u.deg,
      np.array(tel_pa)*u.deg,
    ],
    names = ('field','ra','dec','pa'),
    meta = {
      'comments': ['The attitudes of the telescope.']
    })
  print(table)
  filename = filename.replace('.txt','_pointing.txt')
  table.write(filename, format='ascii.ipac', overwrite=True)


if __name__ == '__main__':
  parser = ap(description='Generate Case-5 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')

  args = parser.parse_args()

  for n in range(args.num):
    filename=f'case5_challenge_{n:02d}.txt'
    generate_challenge(filename)

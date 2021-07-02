#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import astropy.units as u
import warpfield as w

description='''
This script generates Case-4 astrometry challenges. The telescope used
in this challenge is affected by distortion. Astronomical sources are retrived
from the Gaia EDR3 catalog. Observation fields are set up around the Galactic
center region. A list of source positions on the focal plane and original ICRS
coordinates is provided. Solve the position angle and distortion parameters.
'''.strip()



seed = 42
np.random.seed(seed)


def generate_challenge(filename):
  lon = Longitude(np.random.uniform(-1.0,1.0)*u.deg)
  lat = Latitude(np.random.uniform(-0.5,0.5)*u.deg)
  pa = Angle(np.random.uniform(0,360)*u.deg)

  pointing = SkyCoord(lon, lat, frame='galactic')
  jasmine = w.Telescope(pointing, pa)

  radius = Angle(0.3*u.deg)
  sources = w.retrieve_gaia_sources(pointing,radius)

  scale = np.random.normal(1.00,0.01)
  K = np.random.normal(0,5e1,size=1)
  S = np.random.normal(0,3e0,size=2)
  T = np.random.normal(0,5e1,size=1)
  distortion = w.distortion_generator(K,S,T,scale=scale)
  jasmine.set_distortion(distortion)

  position = jasmine.observe(sources)[0]

  table = QTable(
    [
      position.x*u.um,
      position.y*u.um,
      position.ra*u.deg,
      position.dec*u.deg,
    ],
    names = ('x','y','ra','dec'),
    meta = {
      'keywords': {
        'pointing_ra'   : {'value': lon.deg},
        'pointing_dec'  : {'value': lat.deg},
        'position_angle': {'value': pa.deg},
        'distortion_K': {'value': tuple(K)},
        'distortion_S': {'value': tuple(S)},
        'distortion_T': {'value': tuple(T)},
        'distortion_scale': {'value': scale},
      },
      'comments': [description,]
    })
  print(table)
  table.write(filename, format='ascii.ipac', overwrite=True)


if __name__ == '__main__':
  parser = ap(description='Generate Case-4 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')

  args = parser.parse_args()

  for n in range(args.num):
    filename=f'case4_challenge_{n:02d}.txt'
    generate_challenge(filename)
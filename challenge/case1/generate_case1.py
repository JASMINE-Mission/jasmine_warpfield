#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import astropy.units as u
import warpfield as w

description='''
This script generates Case-1 astrometry challenges. The telescope used
in this challenge is distortion-free. Sources are retrieved from the
Gaia EDR3 catalog. A list of source positions on the focal plane and
ICRS coordinates is provided. Solve the field center and the position angle.
'''.strip()



seed = 42
np.random.seed(seed)


def generate_challenge(filename):
  lon = Longitude(np.random.uniform(0,360)*u.deg)
  lat = Latitude(np.random.uniform(-90,90)*u.deg)
  pa = Angle(np.random.uniform(0,360)*u.deg)

  pointing = SkyCoord(lon, lat, frame='icrs')
  jasmine = w.Telescope(pointing, pa)

  radius = Angle(0.3*u.deg)
  sources = w.retrieve_gaia_sources(pointing,radius)

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
      },
      'comments': [description,]
    })
  print(table)
  table.write(filename, format='ascii.ipac', overwrite=True)


if __name__ == '__main__':
  parser = ap(description='Generate Case-1 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')

  args = parser.parse_args()

  for n in range(args.num):
    filename=f'case1_challenge_{n:02d}.txt'
    generate_challenge(filename)

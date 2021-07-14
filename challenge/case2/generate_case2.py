#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import astropy.units as u
import warpfield as w

description='''
This script generates Case-2 astrometry challenges. The telescope used
in this challenge is distortion-free. Sources are retrieved from the
Gaia EDR3 catalog. A list of source positions on the focal plane is provided.
Corresponding stellar coordinates are not given.
Instead, a list of the coordinates
Solve the field center and the position angle.
'''.strip()



seed = 42
np.random.seed(seed)


def generate_source_catalog(pointing, radius, filename):
  sources = w.retrieve_gaia_sources(pointing,radius)
  icrs = sources.transform_to('icrs')
  catalog_info = '''
  The master source list of the Case-2 challenges. All the sources are
  retrieved from the Gaia EDR3 catalog. The coordinates of the search circle
  is (ra, dec) = ({:.2f}, {:.2f}) with the radius of {:.2f} deg.
  '''.format(pointing.icrs.ra.deg,pointing.icrs.dec.deg,radius.deg).strip()

  table = QTable(
    [
      sources.ra.deg.array*u.deg,
      sources.dec.deg.array*u.deg,
    ],
    names = ('ra','dec'),
    meta = {
      'comments': [catalog_info]
  })
  print(table)
  table.write(filename, format='ascii.ipac', overwrite=True)


def generate_challenge(pointing, radius, catalog, filename):
  separation = (radius-0.3*u.deg)*np.random.uniform(0,1)
  direction = Angle(np.random.uniform(0,360)*u.deg)
  pa = Angle(np.random.uniform(0,360)*u.deg)

  pointing = pointing.directional_offset_by(direction, separation)
  jasmine = w.Telescope(pointing, pa)

  table = QTable.read(catalog, format='ascii.ipac')
  sources = SkyCoord(table['ra'], table['dec'], frame='icrs')
  position = jasmine.observe(sources)[0]

  table = QTable(
    [
      position.x.array*u.um,
      position.y.array*u.um,
    ],
    names = ('x','y'),
    meta = {
      'keywords': {
        'pointing_ra'   : {'value': lon.deg},
        'pointing_dec'  : {'value': lat.deg},
        'position_angle': {'value': pa.deg},
      },
      'comments': [description,]
    })
  print(table)
  table.write(filename, format='ascii.ipac',overwrite=True)


if __name__ == '__main__':
  parser = ap(description='Generate Case-2 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')
  parser.add_argument(
    '--starlist', action='store_true',
    help='generate source list')
  parser.add_argument(
    '--catalog', type=str, default='case2_source_list.txt',
    help='filename of the source list')

  args = parser.parse_args()

  lon = Longitude(np.random.uniform(0,360)*u.deg)
  lat = Latitude(np.random.uniform(-90,90)*u.deg)
  pointing = SkyCoord(lon, lat, frame='icrs')
  radius = Angle(2.0*u.deg)

  if args.starlist is True:
    generate_source_catalog(pointing, radius, args.catalog)
    exit()

  for n in range(args.num):
    filename=f'case2_challenge_{n:02d}.txt'
    generate_challenge(pointing, radius, args.catalog, filename)

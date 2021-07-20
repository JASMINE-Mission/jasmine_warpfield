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
This script generates Case-6 astrometry challenges. The telescope used
in this challenge is affected by distortion. Astronomical sources are retrived
from the Gaia EDR3 catalog, while observations fields are not in the Galactic
center region. In total, 10 fields are obtained for each distortion pattern.
A list of source positions on the focal plane and original ICRS coordinates
is provided. Solve the position angle and distortion parameters.
'''.strip()



seed = 42
np.random.seed(seed)


def generate_source_catalog(pointing, radius, filename):
  sources = w.retrieve_gaia_sources(pointing,radius)
  icrs = sources.transform_to('icrs')
  catalog_info = '''
  The master source list of the Case-6 challenges. All the sources are
  retrieved from the Gaia EDR3 catalog. The coordinates of the search circle
  is (ra, dec) = ({:.2f}, {:.2f}) with the radius of {:.2f} deg.
  '''.format(pointing.icrs.ra.deg,pointing.icrs.dec.deg,radius.deg).strip()

  table = QTable(
    [
      np.arange(len(icrs)),
      icrs.ra.deg*u.deg,
      icrs.dec.deg*u.deg,
    ],
    names = ('catalog_id','ra','dec'),
    meta = {
      'comments': [catalog_info]
  })
  print(table)
  table.write(filename, format='ascii.ipac', overwrite=True)


def generate_challenge(pointing, radius, catalog, stride, filename):
  table = QTable.read(catalog, format='ascii.ipac')
  sources = SkyCoord(table['ra'],table['dec'], frame='icrs')

  separation = (radius-0.3*u.deg)*np.random.uniform(0,1)
  direction = Angle(np.random.uniform(0,360)*u.deg)
  pa = Angle(np.random.uniform(0,360)*u.deg)
  pointing = pointing.directional_offset_by(direction, separation)

  arr = (stride.deg)*np.arange(-2,3)
  ll,bb = np.meshgrid(arr,arr)

  fields  = []
  catalog = []
  tel_ra  = []
  tel_dec = []
  tel_pa  = []

  for n,(l,b) in enumerate(zip(ll.flat,bb.flat)):
    dl = np.random.normal(0,5)*u.arcsec
    db = np.random.normal(0,5)*u.arcsec
    lon = Longitude(pointing.galactic.l+l*u.deg+dl)
    lat = Latitude(pointing.galactic.b+b*u.deg+db)
    center = SkyCoord(lon, lat, frame='galactic')
    pa = Angle(np.random.normal(0,1)*u.degree)

    jasmine = w.Telescope(center, pa)
    position = jasmine.observe(sources)[0]
    position['catalog_id'] = position.index.to_series()
    position['field'] = n
    position = position.reset_index(drop=True)

    catalog.append(position)
    fields.append(n)
    tel_ra.append(center.icrs.ra.deg)
    tel_dec.append(center.icrs.dec.deg)
    tel_pa.append(pa.deg)

  catalog = pd.concat(catalog)

  table = QTable(
    [
      catalog.x.array*u.um,
      catalog.y.array*u.um,
      catalog.catalog_id.array,
      catalog.ra.array*u.deg,
      catalog.dec.array*u.deg,
      catalog.field.array,
    ],
    names = ('x','y','catalog_id','ra','dec','field'),
    meta = {
      'keywords': {
        'pointing_ra'   : {'value': tel_ra[12]},
        'pointing_dec'  : {'value': tel_dec[12]},
        'position_angle': {'value': tel_pa[12]},
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
  parser = ap(description='Generate Case-6 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')
  parser.add_argument(
    '--generate-catalog', action='store_true',
    help='generate source list')
  parser.add_argument(
    '--catalog', type=str, default='case6_source_list.txt',
    help='filename of the source list')

  args = parser.parse_args()

  lon = Longitude(0.0*u.deg)
  lat = Latitude(0.0*u.deg)
  pointing = SkyCoord(lon, lat, frame='galactic')
  radius = Angle(2.0*u.deg)

  if args.generate_catalog is True:
    generate_source_catalog(pointing, radius, args.catalog)
    exit()

  for n in range(args.num):
    catalog = args.catalog
    stride = Angle(0.1*(n+1)*u.deg)
    filename=f'case6_challenge_{n:02d}.txt'
    generate_challenge(pointing, radius, catalog, stride, filename)

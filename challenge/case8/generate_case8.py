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
This script generates Case-8 astrometry challenges. The telescope used
in this challenge is affected by distortion. Astronomical sources are retrived
from the Gaia EDR3 catalog, while observations fields are not in the Galactic
center region. In total, 4 fields are obtained for each distortion pattern.
A list of source positions on the focal plane and original ICRS coordinates
is provided. Solve the position angle and distortion parameters.
'''.strip()



seed = 42
np.random.seed(seed)


class PositionAngle(Longitude):
  pass


def distortion_generator(c, K):
  from functools import reduce
  from operator import add
  cx,cy = c
  r0 = max((cx-20000.)**2+(cy-20000.)**2, (cx+20000.)**2+(cy-20000.)**2,
           (cx-20000.)**2+(cy+20000.)**2, (cx+20000.)**2+(cy+20000.)**2)
  def distortion(position):
    position = np.array(position)
    center = np.array(c).reshape((2,1))
    ka = K.reshape((-1,2))
    r2 = np.square(position-center).sum(axis=0,keepdims=True)/r0
    Kp = 1-reduce(add,[k.reshape((2,1))*(r2**(n+1)) for n,k in enumerate(ka)])
    return (position-center)/Kp+center
  return distortion


def generate_source_catalog(pointing, radius, filename):
  sources = w.retrieve_gaia_sources(pointing,radius)
  icrs = sources.transform_to('icrs')
  catalog_info = '''
  The master source list of the Case-7 challenges. All the sources are
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

  ## calculate the position angle offset due to the cooridinate conversion.
  pos = pointing.directional_offset_by(0.0*u.deg, 0.1*u.deg)
  pa0 = pointing.icrs.position_angle(pos)

  separation = (radius-3*u.arcmin)*np.random.uniform(0,1)
  direction = Angle(np.random.uniform(0,360)*u.deg)
  pointing = pointing.directional_offset_by(direction, separation)

  arr = (stride.deg)*np.array([-1,1])
  ll,bb = np.meshgrid(arr,arr)


  c = np.random.uniform(-50000,50000,size=2)
  K = np.random.normal(0, 0.01, size=6)
  distortion = distortion_generator(c, K)

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
    pa  = Angle(np.random.normal(0,1)*u.degree)

    ## calculate the position angle offset due to the cooridinate conversion.
    pos = center.directional_offset_by(0.0*u.deg, 0.1*u.deg)
    dpa = center.icrs.position_angle(pos)

    jasmine = w.Telescope(center, pa)
    vanilla = jasmine.optics.imaging(sources)
    vanilla['catalog_id'] = vanilla.index.to_series()
    vanilla = vanilla.loc[:,['x','y','catalog_id']]
    jasmine.set_distortion(distortion)
    distorted = jasmine.observe(sources)[0]
    distorted['catalog_id'] = distorted.index.to_series()
    distorted['field'] = n
    distorted = distorted.reset_index(drop=True)
    position = distorted.merge(vanilla, on='catalog_id', suffixes=('','_orig'))

    catalog.append(position)
    fields.append(n)
    tel_ra.append(center.icrs.ra.deg)
    tel_dec.append(center.icrs.dec.deg)
    tel_pa.append(PositionAngle(pa+dpa).deg)

  catalog = pd.concat(catalog)

  keywords = {
    'pointing_ra'   : {'value': pointing.icrs.ra.deg},
    'pointing_dec'  : {'value': pointing.icrs.dec.deg},
    'position_angle': {'value': pa0.deg},
    'distortion_xc'  : {'value': c[0]},
    'distortion_yc'  : {'value': c[1]},
  }
  for n,k in enumerate(K):
    keywords[f'distortion_K[{n}]'] = {'value': k}

  table = QTable(
    [
      catalog.x.array*u.um,
      catalog.y.array*u.um,
      catalog.catalog_id.array,
      catalog.x_orig.array*u.um,
      catalog.y_orig.array*u.um,
      catalog.ra.array*u.deg,
      catalog.dec.array*u.deg,
      catalog.field.array,
    ],
    names = ('x','y','catalog_id','x_orig','y_orig','ra','dec','field'),
    meta = {
      'keywords': keywords,
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
  parser = ap(description='Generate Case-8 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')
  parser.add_argument(
    '--generate-catalog', action='store_true',
    help='generate source list')
  parser.add_argument(
    '--catalog', type=str, default='case8_source_list.txt',
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
    stride = Angle(4*u.arcmin)
    filename=f'case8_challenge_{n:02d}.txt'
    generate_challenge(pointing, radius, catalog, stride, filename)

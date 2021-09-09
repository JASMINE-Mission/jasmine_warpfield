#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import pandas as pd
import astropy.units as u
import warpfield as w
from warpfield.DUMMY import get_jasmine

description='''
This script generates Case-7 astrometry challenges. The telescope used
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


def sip_distortion_generator(sip_x, sip_y):
  ''' Generate a distortion function compatible with the SIP notation.

  - c[m,n] = sip_x[m,n]
  - d[m,n] = sip_y[m,n]

  Values for (m,n) = (0,0), (1,0), (0,1) are ignored.
  '''
  def apply_sip(x,y,param,norm=1e6):
    d = np.zeros_like(x)
    narr = np.arange(param.size)
    for m,n in np.ndindex(param.shape):
      if n==0 and m==0: continue
      if n==1 and m==0: continue
      if n==0 and m==1: continue
      d += param[m,n]*(x/norm)**m*(y/norm)**n*(norm**(m+n))
    return d

  def distortion(position):
    position = np.array(position)
    x,y = position[0].copy(), position[1].copy()
    position[0] += apply_sip(x,y,sip_x)
    position[1] += apply_sip(x,y,sip_y)
    return position

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


  sip_n = np.zeros((3,3))
  for m,n in np.ndindex(sip_n.shape):
    sip_n[m,n] = 10**(3-6*(m+n))
  sip_x = np.random.normal(0,10,size=(3,3))*sip_n
  sip_y = np.random.normal(0,10,size=(3,3))*sip_n
  distortion = sip_distortion_generator(sip_x,sip_y)

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

    jasmine = get_jasmine(center, pa)
    jasmine.set_distortion(distortion)
    position = jasmine.observe(sources)[0]
    position['catalog_id'] = position.index.to_series()
    position['field'] = n
    position = position.reset_index(drop=True)

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
  }
  for m,n in np.ndindex(sip_x.shape):
    if m==0 and n==0: continue
    if m==0 and n==1: continue
    if m==1 and n==0: continue
    keywords[f'sip_c[{m},{n}]'] = {'value': sip_x[m,n]}
  for m,n in np.ndindex(sip_y.shape):
    if m==0 and n==0: continue
    if m==0 and n==1: continue
    if m==1 and n==0: continue
    keywords[f'sip_d[{m},{n}]'] = {'value': sip_y[m,n]}

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
  parser = ap(description='Generate Case-7 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')
  parser.add_argument(
    '--generate-catalog', action='store_true',
    help='generate source list')
  parser.add_argument(
    '--catalog', type=str, default='case7_source_list.txt',
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
    filename=f'case7_challenge_{n:02d}.txt'
    generate_challenge(pointing, radius, catalog, stride, filename)

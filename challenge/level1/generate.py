#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import pandas as pd
import astropy.units as u
import warpfield as w
from warpfield.distortion import SipModDistortion
from warpfield.DUMMY import get_jasmine

description='''
JASMINE astrometry challenge for LEVEL-1

- The telescope used in this challenge is affected by distortion.
- Astronomical sources are retrivedfrom the Gaia EDR3 catalog,
  while observations fields are not in the Galactic center region.
- In total, 4xN fields are obtained for each distortion pattern.
- Source positions on the focal plane and ICRS coordinates is provided.
- Solve the position angle and distortion parameters.
'''.strip()



seed = 42
np.random.seed(seed)


class PositionAngle(Longitude):
  pass


def get_field():
  l,b = Longitude(0.2*u.deg), Latitude(0.2*u.deg)
  return SkyCoord(l, b, frame='galactic')


def generate_challenge(pointing, catalog, plate, stride, filename):
  table = QTable.read(catalog, format='ascii.ipac')
  sources = SkyCoord(table['ra'],table['dec'], frame='icrs')

  ## calculate the position angle offset due to the cooridinate conversion.
  pos = pointing.directional_offset_by(0.0*u.deg, 0.1*u.deg)
  pa0 = pointing.icrs.position_angle(pos)

  arr = (stride.deg)*np.array([-1,1])
  ll,bb = np.meshgrid(arr,arr)

  n_sip = 5
  sip_z = np.zeros((n_sip+1,n_sip+1))
  for m,n in np.ndindex(sip_z.shape):
    sip_z[m,n] = 10**(3-6*(m+n)) if (1 < m+n < n_sip+1) else 0.0
  sip_x = np.random.normal(0,1,size=(n_sip+1,n_sip+1))*sip_z
  sip_y = np.random.normal(0,1,size=(n_sip+1,n_sip+1))*sip_z
  sip_c = np.random.uniform(-3000,3000,2)
  distortion = SipModDistortion(n_sip,sip_c,sip_x,sip_y)

  fields  = []
  blocks  = []
  plates  = []
  catalog = []
  tel_ra  = []
  tel_dec = []
  tel_pa  = []

  for n,(l,b) in enumerate(zip(ll.flat,bb.flat)):
    for m in range(plate):
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
      vanilla = jasmine.optics.imaging(sources)
      vanilla['catalog_id'] = vanilla.index.to_series()
      vanilla = vanilla.loc[:,['x','y','catalog_id']]
      jasmine.set_distortion(distortion)
      distorted = jasmine.observe(sources)[0]
      distorted['catalog_id'] = distorted.index.to_series()
      distorted['field_id'] = n
      distorted['block_id'] = m
      distorted['plate_id'] = m+n*plate
      distorted = distorted.reset_index(drop=True)
      position = distorted.merge(
        vanilla, on='catalog_id', suffixes=('','_orig'))

      dx = position.x-position.x_orig
      dy = position.y-position.y_orig
      print(dx.max(),dx.min(),dx.mean(),dx.median(),dx.std())
      print(dy.max(),dy.min(),dy.mean(),dy.median(),dy.std())

      catalog.append(position)
      fields.append(n)
      blocks.append(m)
      plates.append(m+n*plate)
      tel_ra.append(center.icrs.ra.deg)
      tel_dec.append(center.icrs.dec.deg)
      tel_pa.append(PositionAngle(pa+dpa).deg)

  catalog = pd.concat(catalog)

  keywords = {
    'pointing_ra'   : {'value': pointing.icrs.ra.deg},
    'pointing_dec'  : {'value': pointing.icrs.dec.deg},
    'position_angle': {'value': pa0.deg},
  }

  table = QTable(
    [
      catalog.x.array*u.um,
      catalog.y.array*u.um,
      catalog.catalog_id.array,
      catalog.field_id.array,
      catalog.block_id.array,
      catalog.plate_id.array,
      catalog.x_orig.array*u.um,
      catalog.y_orig.array*u.um,
      catalog.ra.array*u.deg,
      catalog.dec.array*u.deg,
    ],
    names = (
      'x','y','catalog_id',
      'field_id','block_id','plate_id',
      'x_orig','y_orig','ra','dec'),
    meta = {
      'keywords': keywords,
      'comments': [description,]
    })
  print(table)
  table.write(filename, format='ascii.ipac', overwrite=True)

  table = QTable(
    [
      fields,
      blocks,
      plates,
      np.array(tel_ra)*u.deg,
      np.array(tel_dec)*u.deg,
      np.array(tel_pa)*u.deg,
    ],
    names = ('field','block','plate','ra','dec','pa'),
    meta = {
      'comments': ['The attitudes of the telescope.']
    })
  print(table)
  filename = filename.replace('.txt','_pointing.txt')
  table.write(filename, format='ascii.ipac', overwrite=True)

  with open('sip.npz', 'wb') as f:
    np.savez(f, sip_c=sip_c, sip_x=sip_x, sip_y=sip_y)


if __name__ == '__main__':
  parser = ap(description='Generate Level-1 challenges')
  parser.add_argument(
    '--catalog', type=str, default='source_list.txt',
    help='filename of the source list')

  args = parser.parse_args()

  pointing = get_field()

  plate = 32
  catalog = args.catalog
  stride = Angle(4*u.arcmin)
  filename=f'level1_challenge.txt'
  generate_challenge(pointing, catalog, plate, stride, filename)

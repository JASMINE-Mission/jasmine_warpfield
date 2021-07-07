#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import astropy.units as u
import warpfield as w

description='''
This script generates Case-3 astrometry challenges. The telescope used
in this challenge is affected by distortion. Artificial sources tiled in a
grid pattern are generated. A list of source positions on the focal plane and
original ICRS coordinates is provided.
Solve the position angle and distortion parameters.
'''.strip()



seed = 42
np.random.seed(seed)


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


def generate_challenge(filename):
  lon = Longitude(0.0*u.deg)
  lat = Latitude(0.0*u.deg)
  pa = Angle(np.random.uniform(0,360)*u.deg)

  pointing = SkyCoord(lon, lat, frame='icrs')
  jasmine = w.Telescope(pointing, pa)

  arr = np.linspace(-0.3,0.3,50)
  xx,yy = np.meshgrid(arr,arr)
  ra = [x*u.deg for x in xx.flat]
  dec = [y*u.deg for y in yy.flat]
  sources = SkyCoord(ra, dec, frame='icrs')

  sip_x = np.random.normal(0,3e-15,size=(3,3))
  sip_y = np.random.normal(0,3e-15,size=(3,3))
  print(sip_x)
  print(sip_y)
  distortion = sip_distortion_generator(sip_x,sip_y)
  jasmine.set_distortion(distortion)

  # jasmine.display_focal_plane(sources)
  position = jasmine.observe(sources)[0]

  keywords = {
    'pointing_ra'   : {'value': lon.deg},
    'pointing_dec'  : {'value': lat.deg},
    'position_angle': {'value': pa.deg},
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
      position.x*u.um,
      position.y*u.um,
      position.ra*u.deg,
      position.dec*u.deg,
    ],
    names = ('x','y','ra','dec'),
    meta = {
      'keywords': keywords,
      'comments': [description,]
    })
  print(table)
  table.write(filename, format='ascii.ipac', overwrite=True)


if __name__ == '__main__':
  parser = ap(description='Generate Case-3 challenges')
  parser.add_argument(
    '-n', '--num', type=int, default=5,
    help='the number of challenges')

  args = parser.parse_args()

  for n in range(args.num):
    filename=f'case3_challenge_{n+5:02d}.txt'
    generate_challenge(filename)

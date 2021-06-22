#!/usr/bin/env python
# -*- coding: utf-8 -*-
from astropy.coordinates import SkyCoord, Longitude, Latitude
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u


def get_projection(lon, lat, frame='icrs'):
  proj = WCS(naxis=2)
  proj.wcs.crpix = [50.5, 50.5]
  proj.wcs.cdelt = [10/3600., 10/3600.]
  proj.wcs.cunit = ['deg', 'deg']
  proj.wcs.crval = [lon, lat]
  if frame == 'galactic':
    proj.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
  elif frame in ('icrs', 'fk5', 'fk4'):
    proj.wcs.ctype = ['RA---TAN', 'DEC--TAN']
  print(proj)
  print(proj.wcs)

  return proj

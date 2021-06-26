#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from astropy.coordinates import Longitude, Latitude
from astropy.wcs import WCS


__debug_mode__ = False


def get_projection(pointing, scale=1.0):
  ''' Obtain the gnomonic projection instance.

  Parameters:
    pointing (SkyCoord): the center of the search point.
    scale (float)      : the conversion factor to calculate the position on
                         the focal plane from the angular distance on the sky.
  '''
  proj = WCS(naxis=2)
  proj.wcs.crpix = [0., 0.]
  if pointing.frame.name == 'galactic':
    lon = pointing.galactic.l.deg
    lat = pointing.galactic.b.deg
    proj.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
  else:
    lon = pointing.icrs.ra.deg
    lat = pointing.icrs.dec.deg
    proj.wcs.ctype = ['RA---TAN', 'DEC--TAN']
  proj.wcs.crval = [lon, lat]
  proj.wcs.cdelt = [-scale/3600., scale/3600.]
  proj.wcs.cunit = ['deg', 'deg']

  if __debug_mode__ is True:
    print(proj)
    print(proj.wcs)

  return proj

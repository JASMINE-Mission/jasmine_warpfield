#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from astropy.coordinates import Longitude, Latitude
from astropy.wcs import WCS


__debug_mode__ = False


class Frame(Enum):
  ''' The reference frame of the coordinate system. '''
  icrs     = 'icrs'
  fk5      = 'fk5'
  galactic = 'galactic'


def get_projection(lon, lat, scale=1.0, frame=Frame.icrs):
  ''' Obtain the gnomonic projection instance.

  Parameters:
    lon (float or Longitude): the longitude of the frame center in degree.
    lat (float or Latitude) : the latitude of the frame center in degree.
    frame (Frame): the reference coordinate system.
    scale (float): the conversion factor to calculate the position on the
                   focal plane from the angular distance on the sky.
  '''
  if isinstance(lon, Longitude): lon = lon.degree
  if isinstance(lat, Latitude): lat = lat.degree
  proj = WCS(naxis=2)
  proj.wcs.crpix = [0., 0.]
  proj.wcs.cdelt = [scale/3600., scale/3600.]
  proj.wcs.cunit = ['deg', 'deg']
  proj.wcs.crval = [lon, lat]
  if frame is Frame.galactic:
    proj.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
  else:
    proj.wcs.ctype = ['RA---TAN', 'DEC--TAN']
  if __debug_mode__ is True:
    print(proj)
    print(proj.wcs)

  return proj

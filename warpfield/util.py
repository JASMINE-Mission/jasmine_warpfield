#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum
from astropy.wcs import WCS


__debug_mode__ = True


class Frame(Enum):
  ''' The reference frame of the coordinate system. '''
  icrs     = 'icrs'
  fk5      = 'fk5'
  galactic = 'galactic'


def get_projection(lon, lat, frame=Frame.icrs):
  ''' Obtain the gnomonic projection instance.

  Parameters:
    lon (float)  : the longitude of the frame center in degree.
    lat (float)  : the latitude of the frame center in degree.
    frame (Frame): the reference coordinate system.
  '''
  proj = WCS(naxis=2)
  proj.wcs.crpix = [50.5, 50.5]
  proj.wcs.cdelt = [10/3600., 10/3600.]
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

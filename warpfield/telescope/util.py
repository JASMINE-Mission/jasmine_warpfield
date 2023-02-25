#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' miscellaneous tools
'''

from astropy.wcs import WCS
import numpy as np
import sys

__arcsec_to_um__ = 1.0 / 3600


def eprint(message):
    print(message, file=sys.stderr)


def estimate_frame_from_ctype(ctype):
    ''' Estimate a coordinate frame from CTYPE.

    Arguments:
      ctype (tuple):
          A tuple of CTYPE strings.

    Returns:
      A string to specify the coordinate frame.
    '''
    ctype1, ctype2 = ctype

    if 'GLON' in ctype1 and 'GLAT' in ctype2:
        return 'galactic'
    elif 'RA' in ctype1 and 'DEC' in ctype2:
        return 'icrs'
    else:
        raise ValueError(f'unsupported CTYPE: {ctype}')


def get_axis_name(frame):
    ''' Estimate the axes names from the coordinate frame name

    Arguments:
      frame (str):
          The name of the cooridante frame.

    Returns:
      xlabel: the name of the Longitude
      ylabel: the name of the Latitude
    '''
    if frame in ('galactic'):
        xlabel = 'Galactic Longitude'
        ylabel = 'Galactic Latitude'
    else:
        xlabel = 'Right Ascension'
        ylabel = 'Declination'

    return xlabel, ylabel


def frame_conversion(skycoord, frame):
    ''' Return functions to retrieve spherical coordinates

    Arguments:
      frame (str):
          The name of the cooridante frame.

    Returns:
      A converted skycoord object.
    '''
    if frame == ('gcrs'):
        func = lambda x: getattr(x, 'gcrs')
    elif frame in ('icrs', 'barycentric'):
        func = lambda x: getattr(x, 'icrs')
    elif frame in ('gcgrs', 'galactic'):
        func = lambda x: getattr(x, 'gcgrs')
    else:
        raise ValueError(f'unsupported frame: {frame}')

    return func(skycoord)


def get_projection(
        pointing,
        scale=__arcsec_to_um__,
        rotation=0.0,
        left_hand_system=True,
        projection='TAN'):
    ''' Obtain the gnomonic projection instance.

    Arguments:
      pointing (SkyCoord):
          The coordinates of the projection center.
      scale (float):
          The conversion factor to calculate the position on the focal plane
          from the angular distance on the sky in units of degree/um.
      lhcs (bool, optional):
          Set True if the coordinate is left-handded.

    Returns:
      An object (astropy.wcs.WCS) for coordinate conversion.
    '''

    # This projection instance is used to map celestrical coordinates onto
    # a telescope focal plane. The conversion function `SkyCoord.to_pixel()`
    # requires the `origin` parameter.
    # The coordinates `CRVAL` denotes the center of the field-of-view. Thus,
    # the projection instance should convert the `CRVAL` coordinates into
    # the origin (0,0) on the focal plane. The convination of `crpix = [1, 1]`
    # and `origin = 0` will fullfil the requirements.
    proj = WCS(naxis=2)
    proj.wcs.crpix = [1., 1.]
    lon = pointing.spherical.lon.deg
    lat = pointing.spherical.lat.deg
    if pointing.frame.name == 'galactic':
        proj.wcs.ctype = [f'GLON-{projection}', f'GLAT-{projection}']
    else:
        proj.wcs.ctype = [f'RA---{projection}', f'DEC--{projection}']
    proj.wcs.crval = [lon, lat]
    rot = np.array([
        [+np.cos(rotation), +np.sin(rotation)],
        [-np.sin(rotation), +np.cos(rotation)]])
    delt = np.diag([-scale, scale] if left_hand_system else [scale, scale])
    proj.wcs.cd = rot @ delt
    proj.wcs.cunit = ['deg', 'deg']

    return proj

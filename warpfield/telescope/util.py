#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" miscellaneous tools
"""

from astropy.wcs import WCS

__debug_mode__ = False


def get_projection(pointing, scale=1.0 / 3600., lhcs=True):
    """ Obtain the gnomonic projection instance.

    Arguments:
      pointing (SkyCoord):
          the coordinates of the projection center.
      scale (float):
          the conversion factor to calculate the position on the focal plane
          from the angular distance on the sky in units of degree/um.
      lhcs (bool, optional):
          set True if the coordinate is left-handded.

    Returns:
      an object (astropy.wcs.WCS) for coordinate conversion.
    """

    # This projection instance is used to map celestrical coordinates onto
    # a telescope focal plane. The conversion function `SkyCoord.to_pixel()`
    # requires the `origin` parameter.
    # The coordinates `CRVAL` denotes the center of the field-of-view. Thus,
    # the projection instance should convert the `CRVAL` coordinates into
    # the origin (0,0) on the focal plane. The convination of `crpix = [1, 1]`
    # and `origin = 0` will fullfil the requirements.
    proj = WCS(naxis=2)
    proj.wcs.crpix = [1., 1.]
    if pointing.frame.name == 'galactic':
        lon = pointing.galactic.l.deg
        lat = pointing.galactic.b.deg
        proj.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
    else:
        lon = pointing.icrs.ra.deg
        lat = pointing.icrs.dec.deg
        proj.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    proj.wcs.crval = [lon, lat]
    if lhcs is True:
        proj.wcs.cdelt = [-scale, scale]
    else:
        proj.wcs.cdelt = [scale, scale]
    proj.wcs.cunit = ['deg', 'deg']

    if __debug_mode__ is True:
        print(proj)
        print(proj.wcs)

    return proj

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp

from ..conversion import degree_to_radian, rotation_matrix


def sptrig_cosr(tel_ra, tel_dec, ra, dec):
    ''' Calculate cos(r), the cosine of the distance r

    Arguments:
      tel_ra: A right ascension of the telescope center in radian.
      tel_dec: A declinatoin of the telescope center in radian.
      ra: A right ascension of the target in radian.
      dec: A declination of the target in radian.

    Return:
      cos(r) where r is the true distance from the telescope to the target.
    '''
    return jnp.sin(tel_dec) * jnp.sin(dec) \
        + jnp.cos(tel_dec) * jnp.cos(dec) * jnp.cos(ra - tel_ra)


def generate_conversion(xfunc, yfunc):
    def conversion(tel_ra, tel_dec, ra, dec):
        X = -xfunc(tel_ra, tel_dec, ra, dec) * 180.0 / jnp.pi
        Y = +yfunc(tel_ra, tel_dec, ra, dec) * 180.0 / jnp.pi
        return X, Y
    return conversion


def generate_projection(func):
    def inner_func(tel_ra, tel_dec, tel_pa, ra, dec, sx, sy):
        ''' Gnomonic projection of the spherical coordinates

        Arguments:
          tel_ra: A right ascension of the telescope center in degree.
          tel_dec: A declinatoin of the telescope center in degree.
          tel_pa: A position angle of the telescope in degree.
          ra: A right ascension of the target in degree.
          dec: A declination of the target in degree.
          sx: A physical scale of the focal plane along x (mm/deg).
          sx: A physical scale of the focal plane along y (mm/deg).

        Returns:
          Converted coordinates on the focal plane
        '''
        tel_ra = degree_to_radian(tel_ra)
        tel_dec = degree_to_radian(tel_dec)
        tel_pa = degree_to_radian(tel_pa)
        ra = degree_to_radian(ra)
        dec = degree_to_radian(dec)
        X, Y = func(tel_ra, tel_dec, ra, dec)
        return (rotation_matrix(-tel_pa) @ jnp.stack([sx * X, sy * Y])).T
    return inner_func

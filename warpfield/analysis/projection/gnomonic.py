#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Gnomonic projection '''

import jax.numpy as jnp
from jax import vmap


def degree_to_radian(theta):
    ''' Convert degree to radian '''
    return theta * jnp.pi / 180.


def rotation_matrix(theta):
    ''' Calculate rotation matrix R '''
    rot = [jnp.cos(theta), -jnp.sin(theta), jnp.sin(theta), jnp.cos(theta)]
    return jnp.array(rot).reshape([2, 2])


def rsinr(rho):
    ''' Expansion of r/sin(r) in terms of rho = cos(r)

    Calculate an approximation of f(r) = r/sin(r) as a 6th order polynomial
    function of cos(r). The expansion coefficients are calculated by the
    Wolfram|Alpha.

    This approximation is valid for rho > ~0.8.
    '''
    p = jnp.array([240, -1960, 7344, -17150, 29392, -43365, 70544])
    return jnp.polyval(p, rho) / 45045


def gnomonic_cosr(tel_ra, tel_dec, ra, dec):
    ''' Calculate cos(r), the cosine of the distance r '''
    return jnp.sin(tel_dec) * jnp.sin(dec) \
        + jnp.cos(tel_dec) * jnp.cos(dec) * jnp.cos(ra - tel_ra)


def gnomonic_rsint(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate x '''
    rho = gnomonic_cosr(tel_ra, tel_dec, ra, dec)
    return rsinr(rho) \
        * jnp.sin(ra - tel_ra) * jnp.cos(dec)


def gnomonic_rcost(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate y '''
    rho = gnomonic_cosr(tel_ra, tel_dec, ra, dec)
    return rsinr(rho) \
        * (jnp.sin(dec) - rho * jnp.sin(tel_dec)) / jnp.cos(tel_dec)


def gnomonic_conversion(tel_ra, tel_dec, ra, dec):
    X = -gnomonic_rsint(tel_ra, tel_dec, ra, dec) * 180.0 / jnp.pi
    Y = +gnomonic_rcost(tel_ra, tel_dec, ra, dec) * 180.0 / jnp.pi
    return X, Y


def gnomonic(tel_ra, tel_dec, tel_pa, ra, dec, scale):
    ''' Gnomonic projection of the spherical coordinates

    Arguments:
      tel_ra: A right ascension of the telescope center.
      tel_dec: A declinatoin of the telescope center.
      tel_pa: A position angle of the telescope.
      ra: A right ascension of the target.
      dec: A declination of the target.
      scale: A physical scale of the focal plane (mm/deg).

    Returns:
      Converted coordinates on the focal plane
    '''
    tel_ra = degree_to_radian(tel_ra)
    tel_dec = degree_to_radian(tel_dec)
    tel_pa = degree_to_radian(tel_pa)
    ra = degree_to_radian(ra)
    dec = degree_to_radian(dec)
    X, Y = gnomonic_conversion(tel_ra, tel_dec, ra, dec)
    return (rotation_matrix(-tel_pa) @ jnp.stack([X, Y])).T * scale


projection = vmap(gnomonic, (0, 0, 0, 0, 0, 0), 0)

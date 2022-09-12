#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Gnomonic projection '''

import jax.numpy as jnp
from jax import vmap

from .util import sptrig_cosr, generate_projection


def equidistant_rsinr(rho):
    ''' Expansion of r/sin(r) in terms of rho = cos(r)

    Calculate an approximation of f(r) = r/sin(r) as a 6th order polynomial
    function of cos(r). The expansion coefficients are calculated by the
    Wolfram|Alpha.

    This approximation is valid for rho > ~0.8.
    '''
    p = jnp.array([240, -1960, 7344, -17150, 29392, -43365, 70544])
    return jnp.polyval(p, rho) / 45045


def equidistant_rsint(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate x '''
    rho = sptrig_cosr(tel_ra, tel_dec, ra, dec)
    return equidistant_rsinr(rho) \
        * jnp.sin(ra - tel_ra) * jnp.cos(dec)


def equidistant_rcost(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate y '''
    rho = sptrig_cosr(tel_ra, tel_dec, ra, dec)
    return equidistant_rsinr(rho) \
        * (jnp.sin(dec) - rho * jnp.sin(tel_dec)) / jnp.cos(tel_dec)


def equidistant_conversion(tel_ra, tel_dec, ra, dec):
    X = -equidistant_rsint(tel_ra, tel_dec, ra, dec) * 180.0 / jnp.pi
    Y = +equidistant_rcost(tel_ra, tel_dec, ra, dec) * 180.0 / jnp.pi
    return X, Y


equidistant = generate_projection(equidistant_conversion)


projection = vmap(equidistant, (0, 0, 0, 0, 0, 0), 0)

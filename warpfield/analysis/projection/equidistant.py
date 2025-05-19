#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Gnomonic projection '''

import jax.numpy as jnp
from jax import vmap

from .util import sptrig_cosr, generate_conversion, generate_projection


def equidistant_rsinr(rho):
    ''' Expansion of r/sin(r) in terms of rho = cos(r)

    Calculate an approximation of f(r) = r/sin(r) as a 6th order polynomial
    function of cos(r). The expansion coefficients are calculated by the
    Wolfram|Alpha.

    This approximation is valid for rho > ~0.8.
    '''
    p = jnp.array([
        1, 1 / 3,
        2 / 15, 2 / 35,
        8 / 315, 8 / 693,
        16 / 3003, 16 / 6435,
        128 / 109395, 128 / 230945,
        256 / 969969, 256 / 2028117,
        1024 / 16900975, 1024 / 35102025,
        2048 / 145422675, 2048 / 300540195,
        32768 / 9917826435, 32768 / 20419054425,
        655356 / 83945001525
    ])[::-1]
    return jnp.polyval(p, 1 - rho)


def equidistant_rsint(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate x '''
    rho = sptrig_cosr(tel_ra, tel_dec, ra, dec)
    return equidistant_rsinr(rho) \
        * jnp.sin(ra - tel_ra) * jnp.cos(dec)


def equidistant_rcost(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate y

    Note that this function does not work when the telescope is pointed
    around the celestial poles where cos(tel_dec) is extremely small.
    '''
    rho = sptrig_cosr(tel_ra, tel_dec, ra, dec)
    return equidistant_rsinr(rho) \
        * (jnp.sin(dec) - rho * jnp.sin(tel_dec)) / jnp.cos(tel_dec)


equidistant_conversion = \
    generate_conversion(equidistant_rsint, equidistant_rcost)

equidistant = generate_projection(equidistant_conversion)

projection = vmap(equidistant, (0, 0, 0, 0, 0, 0), 0)

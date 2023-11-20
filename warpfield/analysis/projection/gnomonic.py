#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Gnomonic projection '''

import jax.numpy as jnp
from jax import vmap

from .util import sptrig_cosr, generate_conversion, generate_projection


def gnomonic_Rsint(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate x '''
    return jnp.sin(ra - tel_ra) * jnp.cos(dec) \
        / sptrig_cosr(tel_ra, tel_dec, ra, dec)


def gnomonic_Rcost(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate y '''
    return (jnp.sin(dec) * jnp.cos(tel_dec)
        - jnp.sin(tel_dec) * jnp.cos(dec) * jnp.cos(ra - tel_ra)) \
        / sptrig_cosr(tel_ra, tel_dec, ra, dec)


gnomonic_conversion = \
    generate_conversion(gnomonic_Rsint, gnomonic_Rcost)

gnomonic = generate_projection(gnomonic_conversion)

projection = vmap(gnomonic, (0, 0, 0, 0, 0, 0), 0)

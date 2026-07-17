#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Orthographic projection '''

import jax.numpy as jnp
from jax import vmap

from .base import Projection
from .util import _generate_conversion, _generate_projection


__all__ = ['OrthographicProjection']


def _orthographic_Rsint(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate x '''
    del tel_dec
    return jnp.sin(ra - tel_ra) * jnp.cos(dec)


def _orthographic_Rcost(tel_ra, tel_dec, ra, dec):
    ''' Calculate the projected coordinate y '''
    return jnp.sin(dec) * jnp.cos(tel_dec) \
        - jnp.sin(tel_dec) * jnp.cos(dec) * jnp.cos(ra - tel_ra)


_orthographic_conversion = \
    _generate_conversion(_orthographic_Rsint, _orthographic_Rcost)

_orthographic = _generate_projection(_orthographic_conversion)

_projection = vmap(_orthographic, (0, 0, 0, 0, 0, 0), 0)


class OrthographicProjection(Projection):
    ''' Orthographic projection represented as a callable PyTree '''

    def __call__(self, tel_ra, tel_dec, tel_pa, ra, dec, scale):
        return _projection(tel_ra, tel_dec, tel_pa, ra, dec, scale)

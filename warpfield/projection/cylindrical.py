#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Plate Carree cylindrical projection '''

import jax.numpy as jnp
from jax import vmap

from .base import Projection
from .util import _generate_conversion, _generate_projection


__all__ = ['CylindricalProjection']


def _cylindrical_lon(tel_ra, tel_dec, ra, dec):
    ''' Calculate the wrapped longitude offset '''
    del tel_dec, dec
    delta_ra = ra - tel_ra
    return jnp.arctan2(jnp.sin(delta_ra), jnp.cos(delta_ra))


def _cylindrical_lat(tel_ra, tel_dec, ra, dec):
    ''' Calculate the latitude offset '''
    del tel_ra, ra
    return dec - tel_dec


_cylindrical_conversion = \
    _generate_conversion(_cylindrical_lon, _cylindrical_lat)

_cylindrical = _generate_projection(_cylindrical_conversion)

_projection = vmap(_cylindrical, (0, 0, 0, 0, 0, 0), 0)


class CylindricalProjection(Projection):
    ''' Plate Carree projection represented as a callable PyTree '''

    def __call__(self, tel_ra, tel_dec, tel_pa, ra, dec, scale):
        return _projection(tel_ra, tel_dec, tel_pa, ra, dec, scale)

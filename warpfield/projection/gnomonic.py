#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gnomonic projection"""

import jax.numpy as jnp
from jax import vmap

from .base import Projection
from .util import _sptrig_cosr, _generate_conversion, _generate_projection


__all__ = ['GnomonicProjection']


def _gnomonic_Rsint(tel_ra, tel_dec, ra, dec):
    """Calculate the projected coordinate x"""
    return (
        jnp.sin(ra - tel_ra)
        * jnp.cos(dec)
        / _sptrig_cosr(tel_ra, tel_dec, ra, dec)
    )


def _gnomonic_Rcost(tel_ra, tel_dec, ra, dec):
    """Calculate the projected coordinate y"""
    return (
        jnp.sin(dec) * jnp.cos(tel_dec)
        - jnp.sin(tel_dec) * jnp.cos(dec) * jnp.cos(ra - tel_ra)
    ) / _sptrig_cosr(tel_ra, tel_dec, ra, dec)


_gnomonic_conversion = _generate_conversion(_gnomonic_Rsint, _gnomonic_Rcost)

_gnomonic = _generate_projection(_gnomonic_conversion)

_projection = vmap(_gnomonic, (0, 0, 0, 0, 0, 0), 0)


class GnomonicProjection(Projection):
    """Gnomonic projection represented as a callable PyTree"""

    def __call__(self, tel_ra, tel_dec, tel_pa, ra, dec, scale):
        return _projection(tel_ra, tel_dec, tel_pa, ra, dec, scale)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gnomonic projection"""

import jax.numpy as jnp
from jax import vmap

from .base import Projection
from .util import _sptrig_cosr, _generate_conversion, _generate_projection


__all__ = ['EquidistantProjection']


def _equidistant_rsinr(rho):
    """Expansion of r/sin(r) in terms of rho = cos(r)

    Calculate an approximation of f(r) = r/sin(r) as a 6th order polynomial
    function of cos(r). The expansion coefficients are calculated by the
    Wolfram|Alpha.

    This approximation is valid for rho > ~0.8.
    """
    p = jnp.array([
        1,
        1 / 3,
        2 / 15,
        2 / 35,
        8 / 315,
        8 / 693,
        16 / 3003,
        16 / 6435,
        128 / 109395,
        128 / 230945,
        256 / 969969,
        256 / 2028117,
        1024 / 16900975,
        1024 / 35102025,
        2048 / 145422675,
        2048 / 300540195,
        32768 / 9917826435,
        32768 / 20419054425,
        655356 / 83945001525,
    ])[::-1]
    return jnp.polyval(p, 1 - rho)


def _equidistant_rsint(tel_ra, tel_dec, ra, dec):
    """Calculate the projected coordinate x"""
    rho = _sptrig_cosr(tel_ra, tel_dec, ra, dec)
    return _equidistant_rsinr(rho) * jnp.sin(ra - tel_ra) * jnp.cos(dec)


def _equidistant_rcost(tel_ra, tel_dec, ra, dec):
    """Calculate the projected coordinate y"""
    rho = _sptrig_cosr(tel_ra, tel_dec, ra, dec)
    return _equidistant_rsinr(rho) * (
        jnp.sin(dec) * jnp.cos(tel_dec)
        - jnp.sin(tel_dec) * jnp.cos(dec) * jnp.cos(ra - tel_ra)
    )


_equidistant_conversion = _generate_conversion(
    _equidistant_rsint, _equidistant_rcost
)

_equidistant = _generate_projection(_equidistant_conversion)

_projection = vmap(_equidistant, (0, 0, 0, 0, 0, 0), 0)


class EquidistantProjection(Projection):
    """Equidistant projection represented as a callable PyTree"""

    def __call__(self, tel_ra, tel_dec, tel_pa, ra, dec, scale):
        return _projection(tel_ra, tel_dec, tel_pa, ra, dec, scale)

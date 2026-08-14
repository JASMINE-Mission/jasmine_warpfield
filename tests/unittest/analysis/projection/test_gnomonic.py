#!/usr/bin/env python
# -*- coding: utf-8 -*-
import equinox as eqx
import jax
import jax.numpy as jnp
from pytest import approx

from warpfield.projection.gnomonic import (
    GnomonicProjection,
    _gnomonic,
    _gnomonic_conversion,
    _projection,
)


def test_gnomonic_conversion_zero():
    X, Y = _gnomonic_conversion(0.0, 0.0, 0.0, 0.0)
    assert X == approx(0.0)
    assert Y == approx(0.0)


def test_gnomonic_conversion_lon():
    args = [-0.1, 0.0, 0.1]
    for lon in args:
        Y = _gnomonic_conversion(0.0, 0.0, lon, 0.0)[1]
        assert Y == approx(0.0)


def test_gnomonic_conversion_lat():
    args = [-0.1, 0.0, 0.1]
    for lat in args:
        X = _gnomonic_conversion(0.0, 0.0, 0.0, lat)[0]
        assert X == approx(0.0)


def test_gnomonic_rotate():
    def gnomonic_rotate(pa):
        a0 = 266.415  # Right Ascension of the Galactic Center
        d0 = -29.006  # Declination of the Galactic Center
        return _gnomonic(a0, d0, pa, a0 + 1.0, d0 + 1.0, 1.0)

    X0, Y0 = gnomonic_rotate(0.0)

    X1, Y1 = gnomonic_rotate(90.0)
    assert X0 + Y1 == approx(0.0)
    assert Y0 - X1 == approx(0.0)

    X1, Y1 = gnomonic_rotate(180.0)
    assert X0 + X1 == approx(0.0)
    assert Y0 + Y1 == approx(0.0)


def test_gnomonic_projection_model():
    model = GnomonicProjection()
    tel_ra = jnp.array([266.4, 266.5])
    tel_dec = jnp.array([-29.0, -29.1])
    tel_pa = jnp.array([0.0, 5.0])
    ra = jnp.array([266.5, 266.4])
    dec = jnp.array([-28.9, -29.2])
    scale = jnp.array([[1.0, 1.0], [2.0, 3.0]])
    expected = _projection(tel_ra, tel_dec, tel_pa, ra, dec, scale)

    assert model(tel_ra, tel_dec, tel_pa, ra, dec, scale) == approx(expected)
    assert eqx.filter_jit(model)(
        tel_ra, tel_dec, tel_pa, ra, dec, scale) == approx(expected)
    assert len(jax.tree_util.tree_leaves(model)) == 0

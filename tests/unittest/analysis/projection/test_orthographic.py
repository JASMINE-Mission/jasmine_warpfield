#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pytest import approx

from warpfield.projection.orthographic import (
    OrthographicProjection,
    _orthographic,
    _orthographic_conversion,
    _projection,
)


def test_orthographic_conversion_zero():
    X, Y = _orthographic_conversion(0.3, -0.2, 0.3, -0.2)
    assert X == approx(0.0, abs=1.0e-6)
    assert Y == approx(0.0, abs=1.0e-6)


def test_orthographic_conversion_lon():
    longitude = 0.1
    X, Y = _orthographic_conversion(0.0, 0.0, longitude, 0.0)
    assert X == approx(-np.sin(longitude) * 180.0 / np.pi)
    assert Y == approx(0.0)


def test_orthographic_conversion_lat():
    latitude = 0.1
    X, Y = _orthographic_conversion(0.0, 0.0, 0.0, latitude)
    assert X == approx(0.0)
    assert Y == approx(np.sin(latitude) * 180.0 / np.pi)


def test_orthographic_rotate():
    def orthographic_rotate(pa):
        return _orthographic(0.0, 0.0, pa, 1.0, 1.0, 1.0)

    X0, Y0 = orthographic_rotate(0.0)
    X1, Y1 = orthographic_rotate(90.0)
    assert X0 + Y1 == approx(0.0, abs=1.0e-6)
    assert Y0 - X1 == approx(0.0, abs=1.0e-6)


def test_orthographic_projection_model():
    model = OrthographicProjection()
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

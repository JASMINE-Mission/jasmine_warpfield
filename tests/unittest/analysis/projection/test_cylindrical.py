#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pytest import approx

from warpfield.projection.cylindrical import (
    CylindricalProjection,
    _cylindrical,
    _cylindrical_conversion,
    _projection,
)


def test_cylindrical_conversion_zero():
    X, Y = _cylindrical_conversion(0.3, -0.2, 0.3, -0.2)
    assert X == approx(0.0, abs=1.0e-6)
    assert Y == approx(0.0, abs=1.0e-6)


def test_cylindrical_conversion_plate_carree():
    X, Y = _cylindrical_conversion(0.1, -0.2, 0.3, 0.4)
    assert X == approx(-0.2 * 180.0 / np.pi)
    assert Y == approx(+0.6 * 180.0 / np.pi)


def test_cylindrical_conversion_wrap():
    X, Y = _cylindrical_conversion(
        np.deg2rad(179.0),
        0.0,
        np.deg2rad(-179.0),
        0.0,
    )
    assert X == approx(-2.0, abs=1.0e-5)
    assert Y == approx(0.0)


def test_cylindrical_rotate():
    def cylindrical_rotate(pa):
        return _cylindrical(0.0, 0.0, pa, 1.0, 1.0, 1.0)

    X0, Y0 = cylindrical_rotate(0.0)
    X1, Y1 = cylindrical_rotate(90.0)
    assert X0 + Y1 == approx(0.0, abs=1.0e-6)
    assert Y0 - X1 == approx(0.0, abs=1.0e-6)


def test_cylindrical_projection_model():
    model = CylindricalProjection()
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax.numpy as jnp
from pytest import approx, raises

from warpfield.analysis import Optics
from warpfield.analysis.distortion import (
    IdentityDistortion,
    LegendreDistortion,
)
from warpfield.analysis.projection import GnomonicProjection


def generate_coordinates():
    return (
        jnp.array([266.4, 266.5]),
        jnp.array([-29.0, -29.1]),
        jnp.array([0.0, 5.0]),
        jnp.array([266.5, 266.4]),
        jnp.array([-28.9, -29.2]),
        jnp.array([[1.0, 1.0], [2.0, 3.0]]),
    )


def test_identity_optics():
    projection = GnomonicProjection()
    optics = Optics(projection, IdentityDistortion())
    coordinates = generate_coordinates()
    expected = projection(*coordinates)

    assert optics(*coordinates) == approx(expected)
    assert eqx.filter_jit(optics)(*coordinates) == approx(expected)


def test_distorted_optics():
    projection = GnomonicProjection()
    distortion = LegendreDistortion(jnp.ones(18), jnp.ones(18), 10.0)
    optics = Optics(projection, distortion)
    coordinates = generate_coordinates()
    ideal = projection(*coordinates)

    assert optics(*coordinates) == approx(ideal + distortion(ideal))


def test_optics_zodiax_update():
    distortion = LegendreDistortion(jnp.ones(18), jnp.ones(18), 10.0)
    optics = Optics(GnomonicProjection(), distortion)
    updated = optics.set('distortion.coeff_x', jnp.zeros(18))

    assert optics.get('distortion.coeff_x') == approx(jnp.ones(18))
    assert updated.get('distortion.coeff_x') == approx(jnp.zeros(18))


def test_optics_validation():
    with raises(TypeError, match='Projection'):
        Optics(object(), IdentityDistortion())
    with raises(TypeError, match='Distortion'):
        Optics(GnomonicProjection(), object())

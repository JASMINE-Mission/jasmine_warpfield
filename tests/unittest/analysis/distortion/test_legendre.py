#!/usr/bin/env python
# -*- coding: utf-8 -*-
import equinox as eqx
import jax
from pytest import approx, raises
import numpy.polynomial.legendre as legendre
import zodiax as zdx


from .util import *
from warpfield.distortion.legendre import (
    LegendreDistortion,
    _distortion,
    _legval,
    _legval2d,
)


def test_1d_legendre(y, random):
    c = random.normal(size=(10))
    assert _legval(y, c) == approx(legendre.legval(y, c))


def test_2d_legendre(x, y, random):
    c = random.normal(size=(10, 10))
    assert _legval2d(x, y, c) == approx(legendre.legval2d(x, y, c))


def test_distortion(xy, random):
    coeff_a = random.normal(size=(18))
    coeff_b = random.normal(size=(18))
    d = _distortion(coeff_a, coeff_b, xy)
    assert at_origin(d) == approx(0.0)

    d = _distortion(0 * coeff_a, coeff_b, xy)
    assert d[:, 0] == approx(0.0)

    d = _distortion(coeff_a, 0 * coeff_b, xy)
    assert d[:, 1] == approx(0.0)


def test_legendre_distortion(xy, random):
    coeff_x = random.normal(size=(18))
    coeff_y = random.normal(size=(18))
    model = LegendreDistortion(coeff_x, coeff_y)

    assert isinstance(model, zdx.Base)
    assert model(xy) == approx(_distortion(coeff_x, coeff_y, xy))
    assert eqx.filter_jit(model)(xy) == approx(model(xy))
    assert len(jax.tree_util.tree_leaves(model)) == 2


def test_legendre_distortion_zodiax_update(xy):
    model = LegendreDistortion(jnp.ones(18), jnp.ones(18))
    updated = model.set('coeff_x', jnp.zeros(18))

    assert model.get('coeff_x') == approx(jnp.ones(18))
    assert updated.get('coeff_x') == approx(jnp.zeros(18))
    assert updated(xy)[:, 0] == approx(0.0)


def test_legendre_distortion_gradient(xy):
    model = LegendreDistortion(jnp.ones(18), jnp.ones(18))

    def loss(candidate):
        return jnp.sum(candidate(xy)**2)

    gradient = eqx.filter_grad(loss)(model)

    assert jnp.isfinite(gradient.coeff_x).all()
    assert jnp.isfinite(gradient.coeff_y).all()
    assert len(jax.tree_util.tree_leaves(gradient)) == 2


def test_legendre_distortion_validation(xy):
    with raises(ValueError, match='coeff_x'):
        LegendreDistortion(jnp.zeros(17), jnp.zeros(18))
    with raises(ValueError, match='coeff_y'):
        LegendreDistortion(jnp.zeros(18), jnp.zeros(17))
    with raises(ValueError, match='N_coordinate'):
        LegendreDistortion(jnp.zeros(18), jnp.zeros(18))(xy[:, 0])

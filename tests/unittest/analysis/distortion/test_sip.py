#!/usr/bin/env python
# -*- coding: utf-8 -*-
import equinox as eqx
import jax
from pytest import approx, raises
import zodiax as zdx

from .util import *
from warpfield.analysis.distortion.sip import (
    SIPDistortion,
    _distortion,
    _polymap,
)


def test_polymap(xy):
    coeff = jnp.array([0.0, 0.0, 0.4])
    _polymap(coeff, xy)


def test_distortion(xy, random):
    coeff_a = random.normal(size=(18))
    coeff_b = random.normal(size=(18))
    d = _distortion(coeff_a, coeff_b, xy)
    assert at_origin(d) == approx(0.0)

    d = _distortion(0 * coeff_a, 0 * coeff_b, xy)
    assert d == approx(jnp.zeros_like(xy))

    d = _distortion(0 * coeff_a, coeff_b, xy)
    assert d[:, 0] == approx(0.0)

    d = _distortion(coeff_a, 0 * coeff_b, xy)
    assert d[:, 1] == approx(0.0)


def test_sip_distortion(xy, random):
    coeff_x = random.normal(size=(18))
    coeff_y = random.normal(size=(18))
    model = SIPDistortion(coeff_x, coeff_y)

    assert isinstance(model, zdx.Base)
    assert model(xy) == approx(_distortion(coeff_x, coeff_y, xy))
    assert eqx.filter_jit(model)(xy) == approx(model(xy))
    assert len(jax.tree_util.tree_leaves(model)) == 2


def test_sip_distortion_zodiax_update(xy):
    model = SIPDistortion(jnp.ones(18), jnp.ones(18))
    updated = model.set('coeff_x', jnp.zeros(18))

    assert model.get('coeff_x') == approx(jnp.ones(18))
    assert updated.get('coeff_x') == approx(jnp.zeros(18))
    assert updated(xy)[:, 0] == approx(0.0)


def test_sip_distortion_gradient(xy):
    model = SIPDistortion(jnp.ones(18), jnp.ones(18))

    def loss(candidate):
        return jnp.sum(candidate(xy)**2)

    gradient = eqx.filter_grad(loss)(model)

    assert jnp.isfinite(gradient.coeff_x).all()
    assert jnp.isfinite(gradient.coeff_y).all()


def test_sip_distortion_validation(xy):
    with raises(ValueError, match='coeff_x'):
        SIPDistortion(jnp.zeros(17), jnp.zeros(18))
    with raises(ValueError, match='coeff_y'):
        SIPDistortion(jnp.zeros(18), jnp.zeros(17))
    with raises(ValueError, match='N_coordinate'):
        SIPDistortion(jnp.zeros(18), jnp.zeros(18))(xy[:, 0])

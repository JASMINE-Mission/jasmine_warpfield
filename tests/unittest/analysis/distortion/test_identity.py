#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax
import jax.numpy as jnp
from pytest import approx, raises

from warpfield.distortion import Distortion, IdentityDistortion


def test_identity_distortion():
    model = IdentityDistortion()
    xy = jnp.array([[1.0, 2.0], [-3.0, 4.0]])

    assert isinstance(model, Distortion)
    assert model(xy) == approx(jnp.zeros_like(xy))
    assert eqx.filter_jit(model)(xy) == approx(jnp.zeros_like(xy))
    assert len(jax.tree_util.tree_leaves(model)) == 0


def test_identity_distortion_validation():
    with raises(ValueError, match='N_coordinate'):
        IdentityDistortion()(jnp.ones(2))

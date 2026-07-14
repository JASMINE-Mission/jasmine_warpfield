#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import Pointing


def generate_pointing():
    return Pointing(
        ra=[10.0, 20.0],
        dec=[-10.0, -20.0],
        position_angle=[1.0, 2.0],
        scale=[[3.0, 4.0], [5.0, 6.0]],
    )


def test_pointing():
    pointing = generate_pointing()
    ra, dec, position_angle, scale = pointing.take(jnp.array([1, 0]))

    assert isinstance(pointing, zdx.Base)
    assert len(pointing) == 2
    assert ra == approx([20.0, 10.0])
    assert dec == approx([-20.0, -10.0])
    assert position_angle == approx([2.0, 1.0])
    assert scale == approx(jnp.array([[5.0, 6.0], [3.0, 4.0]]))
    assert len(jax.tree_util.tree_leaves(pointing)) == 4


def test_pointing_zodiax_update():
    pointing = generate_pointing()
    updated = pointing.add('position_angle', 1.0)

    assert pointing.get('position_angle') == approx([1.0, 2.0])
    assert updated.get('position_angle') == approx([2.0, 3.0])


def test_pointing_shape_validation():
    with raises(ValueError, match='same shape'):
        Pointing([1.0], [2.0, 3.0], [4.0], [[5.0, 6.0]])
    with raises(ValueError, match='N_pointing'):
        Pointing([1.0], [2.0], [3.0], [4.0])

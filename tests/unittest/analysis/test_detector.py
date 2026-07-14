#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import Detector
from warpfield.analysis.transform.affine import transform


def generate_detector():
    return Detector(
        rotation=[0.0, 90.0],
        offset=[[0.0, 0.0], [1.0, 2.0]],
        pixel_scale=[[1.0, 1.0], [0.5, 2.0]],
    )


def test_detector():
    detector = generate_detector()
    rotation, offset, pixel_scale = detector.take(jnp.array([1, 0]))

    assert isinstance(detector, zdx.Base)
    assert len(detector) == 2
    assert rotation == approx([90.0, 0.0])
    assert offset == approx(jnp.array([[1.0, 2.0], [0.0, 0.0]]))
    assert pixel_scale == approx(jnp.array([[0.5, 2.0], [1.0, 1.0]]))
    assert len(jax.tree_util.tree_leaves(detector)) == 3


def test_detector_transform():
    detector = generate_detector()
    xy = jnp.array([[1.0, -2.0], [3.0, 4.0], [-1.0, 2.0]])
    index = jnp.array([0, 1, 0])

    rotation, offset, pixel_scale = detector.take(index)
    expected = transform(xy, rotation, offset, pixel_scale)

    assert detector(xy, index) == approx(expected)
    assert eqx.filter_jit(detector)(xy, index) == approx(expected)


def test_detector_gradient():
    detector = generate_detector()
    xy = jnp.array([[1.0, -2.0], [3.0, 4.0]])
    index = jnp.array([0, 1])

    def loss(model):
        return jnp.sum(model(xy, index)**2)

    gradient = eqx.filter_grad(loss)(detector)

    assert jnp.isfinite(gradient.rotation).all()
    assert jnp.isfinite(gradient.offset).all()
    assert jnp.isfinite(gradient.pixel_scale).all()


def test_detector_zodiax_update():
    detector = generate_detector()
    updated = detector.set('offset', jnp.ones((2, 2)))

    expected = jnp.array([[0.0, 0.0], [1.0, 2.0]])
    assert detector.get('offset') == approx(expected)
    assert updated.get('offset') == approx(jnp.ones((2, 2)))


def test_detector_shape_validation():
    with raises(ValueError, match='N_detector'):
        Detector([0.0], [0.0, 0.0], [[1.0, 1.0]])
    with raises(ValueError, match='N_detector'):
        Detector([0.0], [[0.0, 0.0]], [1.0, 1.0])


def test_detector_input_validation():
    detector = generate_detector()

    with raises(ValueError, match='N_observation'):
        detector(jnp.ones(2), jnp.array([0]))
    with raises(ValueError, match='same length'):
        detector(jnp.ones((2, 2)), jnp.array([0]))
    with raises(ValueError, match='contain integers'):
        detector(jnp.ones((2, 2)), jnp.array([0.0, 1.0]))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import Detector
from warpfield.analysis.utils import _affine_transform


def generate_detector():
    return Detector(
        rotation=90.0,
        offset=[1.0, 2.0],
        pixel_scale=[0.5, 2.0],
    )


def test_detector():
    detector = generate_detector()

    assert isinstance(detector, zdx.Base)
    assert detector.rotation == approx(90.0)
    assert detector.offset == approx([1.0, 2.0])
    assert detector.pixel_scale == approx([0.5, 2.0])
    assert detector.shape == (1024, 1024)
    assert len(jax.tree_util.tree_leaves(detector)) == 3


def test_detector_transform():
    detector = generate_detector()
    xy = jnp.array([[1.0, -2.0], [3.0, 4.0], [-1.0, 2.0]])
    expected = _affine_transform(
        xy,
        jnp.full(3, detector.rotation),
        jnp.tile(detector.offset, (3, 1)),
        jnp.tile(detector.pixel_scale, (3, 1)),
    )

    assert detector(xy) == approx(expected)
    assert eqx.filter_jit(detector)(xy) == approx(expected)


def test_detector_gradient():
    detector = generate_detector()
    xy = jnp.array([[1.0, -2.0], [3.0, 4.0]])

    def loss(model):
        return jnp.sum(model(xy)**2)

    gradient = eqx.filter_grad(loss)(detector)

    assert jnp.isfinite(gradient.rotation).all()
    assert jnp.isfinite(gradient.offset).all()
    assert jnp.isfinite(gradient.pixel_scale).all()


def test_detector_zodiax_update():
    detector = generate_detector()
    updated = detector.set('offset', jnp.ones(2))

    assert detector.get('offset') == approx([1.0, 2.0])
    assert updated.get('offset') == approx(jnp.ones(2))


def test_detector_shape_validation():
    with raises(ValueError, match='scalar'):
        Detector([0.0], [0.0, 0.0], [1.0, 1.0])
    with raises(ValueError, match='shape'):
        Detector(0.0, [[0.0, 0.0]], [1.0, 1.0])
    with raises(ValueError, match='shape'):
        Detector(0.0, [0.0, 0.0], [[1.0, 1.0]])
    with raises(TypeError, match='tuple of two integers'):
        Detector(0.0, [0.0, 0.0], [1.0, 1.0], [1024, 1024])
    with raises(ValueError, match='positive'):
        Detector(0.0, [0.0, 0.0], [1.0, 1.0], (0, 1024))


def test_detector_input_validation():
    detector = generate_detector()

    with raises(ValueError, match='N_coordinate'):
        detector(jnp.ones(2))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import astropy.units as u
import numpy as np
from pytest import approx, fixture, mark, raises

from warpfield.utils import (
    _affine_transform,
    _degree_to_radian,
    _rotation_matrix,
    plate_scale,
)


@fixture
def xy():
    return jnp.array([[+1, -2], ] * 5)


def generate(rotation, offset, pixel_scale):
    rotation = jnp.array([rotation] * 5)
    offset = jnp.array([offset, ] * 5)
    pixel_scale = jnp.array([pixel_scale, ] * 5)
    return rotation, offset, pixel_scale


def test_plate_scale():
    expected = 1000 * np.pi / 180

    assert plate_scale(1.0 * u.m) == approx([expected, expected])
    assert plate_scale(100.0 * u.cm) == approx([expected, expected])


def test_plate_scale_validation():
    with raises(TypeError, match='Astropy Quantity'):
        plate_scale(1000.0)
    with raises(ValueError, match='length units'):
        plate_scale(1.0 * u.deg)
    with raises(ValueError, match='scalar'):
        plate_scale([1.0, 2.0] * u.m)
    with raises(ValueError, match='finite and positive'):
        plate_scale(0.0 * u.m)


def test_degree_to_radian():
    for theta in [-180, -90, 0, 90, 180]:
        assert _degree_to_radian(theta) == approx(theta * np.pi / 180)


def test_rotation_matrix():
    assert _rotation_matrix(0.0).ravel() == approx([1, 0, 0, 1])
    assert _rotation_matrix(np.pi / 2).ravel() == approx([0, -1, 1, 0])


def test_affine_identity(xy):
    args = generate(0.0, [0.0, 0.0], [1.0, 1.0])
    assert _affine_transform(xy, *args) == approx(xy)


@mark.parametrize(('rotation', 'value'), [
    (90.0, [+2, +1]),
    (180.0, [-1, +2]),
])
def test_affine_rotation(xy, rotation, value):
    args = generate(rotation, [0.0, 0.0], [1.0, 1.0])
    expected = jnp.array([value] * 5)
    assert _affine_transform(xy, *args) == approx(expected)


@mark.parametrize(('offset', 'value'), [
    ([1.0, 0.0], [0, -2]),
    ([0.0, 2.0], [+1, -4]),
])
def test_affine_offset(xy, offset, value):
    args = generate(0.0, offset, [1.0, 1.0])
    expected = jnp.array([value] * 5)
    assert _affine_transform(xy, *args) == approx(expected)


@mark.parametrize(('pixel_scale', 'value'), [
    ([0.5, 1.0], [+2, -2]),
    ([1.0, 2.0], [+1, -1]),
])
def test_affine_pixel_scale(xy, pixel_scale, value):
    args = generate(0.0, [0.0, 0.0], pixel_scale)
    expected = jnp.array([value] * 5)
    assert _affine_transform(xy, *args) == approx(expected)

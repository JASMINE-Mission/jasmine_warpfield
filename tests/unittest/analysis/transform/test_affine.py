#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import jax.numpy as jnp

from warpfield.analysis.transform.affine import *


@fixture
def array():
    return jnp.array([[+1, -2], ] * 5)


def generate(rot, offset, scale):
    rot = jnp.array([rot] * 5)
    offset = jnp.array([offset, ] * 5)
    scale = jnp.array([scale, ] * 5)
    return rot, offset, scale


def test_affine_identity(array):
    rot, offset, scale = generate(0.0, [0.0, 0.0], [1.0, 1.0])
    res = affine(array, rot, offset, scale)

    correct = jnp.array([[+1, -2], ] * 5)
    assert res == approx(correct)


def test_affine_rot90(array):
    rot, offset, scale = generate(90.0, [0.0, 0.0], [1.0, 1.0])
    res = affine(array, rot, offset, scale)

    correct = jnp.array([[+2, +1], ] * 5)
    assert res == approx(correct)


def test_affine_rot180(array):
    rot, offset, scale = generate(180.0, [0.0, 0.0], [1.0, 1.0])
    res = affine(array, rot, offset, scale)

    correct = jnp.array([[-1, +2], ] * 5)
    assert res == approx(correct)


def test_affine_trans_x(array):
    rot, offset, scale = generate(0.0, [1.0, 0.0], [1.0, 1.0])
    res = affine(array, rot, offset, scale)

    correct = jnp.array([[+0, -2], ] * 5)
    assert res == approx(correct)


def test_affine_trans_y(array):
    rot, offset, scale = generate(0.0, [0.0, 2.0], [1.0, 1.0])
    res = affine(array, rot, offset, scale)

    correct = jnp.array([[+1, -4], ] * 5)
    assert res == approx(correct)


def test_affine_scale_x(array):
    rot, offset, scale = generate(0.0, [0.0, 0.0], [0.5, 1.0])
    res = affine(array, rot, offset, scale)

    correct = jnp.array([[+2, -2], ] * 5)
    assert res == approx(correct)


def test_affine_scale_y(array):
    rot, offset, scale = generate(0.0, [0.0, 0.0], [1.0, 2.0])
    res = affine(array, rot, offset, scale)

    correct = jnp.array([[+1, -1], ] * 5)
    assert res == approx(correct)

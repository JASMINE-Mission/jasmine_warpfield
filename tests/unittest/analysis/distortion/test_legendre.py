#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from warpfield.analysis.distortion.legendre import *


@fixture
def x():
    return jnp.linspace(-1, 1, 201)


@fixture
def random():
    return np.random.default_rng(seed=42)


def test_2d_legendre(x, random):
    c = random.normal(size=(10, 10))
    legval2d(x, x, c)


def test_distortion(x, random):
    xy = jnp.stack([x, x]).T
    coeff_a = random.normal(size=(18))
    coeff_b = random.normal(size=(18))
    d = distortion(coeff_a, coeff_b, xy)
    assert d[100] == approx(0.0)

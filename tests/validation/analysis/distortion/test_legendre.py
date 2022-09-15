#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, settings
from hypothesis.strategies import integers, floats
from hypothesis.strategies import lists, tuples, composite
import jax.numpy as jnp
import numpy as np
import numpy.polynomial.legendre as legendre

from warpfield.analysis.distortion.legendre import *


def seeds():
    return integers(0, 2**32 - 1)


@composite
def array_1d(draw):
    return jnp.array(draw(lists(floats(-1, 1), min_size=1, max_size=1000)))


@composite
def array_2d(draw):
    return jnp.array(draw(
        lists(tuples(floats(-1, 1), floats(-1, 1)), min_size=1, max_size=100)))


@composite
def generators(draw):
    return np.random.default_rng(seed=draw(seeds()))


@settings(deadline=None)
@given(array_1d(), generators())
def test_legval(x, gen):
    coeff = gen.normal(size=(16))
    assert legval(x, coeff) == approx(legendre.legval(x, coeff))


@settings(deadline=None)
@given(array_2d(), generators())
def test_legval2d(xy, gen):
    coeff = gen.normal(size=(10, 10))
    assert legval2d(xy[:, 0], xy[:, 1], coeff) \
        == approx(legendre.legval2d(xy[:, 0], xy[:, 1], coeff))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, settings
from hypothesis.strategies import integers, lists, floats, composite
import jax.numpy as jnp
import numpy as np
import numpy.polynomial.legendre as legendre

from warpfield.analysis.distortion.legendre import *


def seeds():
    return integers(0, 2**32 - 1)


@composite
def arrays(draw):
    return jnp.array(draw(lists(floats(-1, 1), min_size=1, max_size=10)))


@composite
def rngs(draw):
    return np.random.default_rng(seed=draw(seeds()))


@settings(deadline=None)
@given(arrays(), rngs())
def test_legval(x, rng):
    coeff = rng.normal(size=(16))
    assert legval(x, coeff) == approx(legendre.legval(x, coeff))


@settings(deadline=None)
@given(arrays(), rngs())
def test_legval2d(x, rng):
    coeff = rng.normal(size=(16))
    assert legval2d(x, x, coeff) == approx(legendre.legval2d(x, x, coeff))

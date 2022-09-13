#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, settings
from hypothesis.strategies import integers, lists, floats
import jax.numpy as jnp
import numpy as np
import numpy.polynomial.legendre as legendre

from warpfield.analysis.distortion.legendre import *


def seeds():
    return integers(0, 2**32 - 1)


def arrays():
    return lists(floats(-1, 1), min_size=1, max_size=1000)


def random(seed):
    return np.random.default_rng(seed=seed)


@settings(deadline=None)
@given(arrays(), seeds())
def test_legval(arr, seed):
    x = jnp.array(arr)
    coeff = random(seed).normal(size=(16))
    assert legval(x, coeff) == approx(legendre.legval(x, coeff))

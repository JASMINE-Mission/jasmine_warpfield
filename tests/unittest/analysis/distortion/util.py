#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import fixture
import jax.numpy as jnp
import numpy as np


@fixture
def x():
    return jnp.linspace(-1, 1, 201)


@fixture
def xy(x):
    return jnp.stack([x, x]).T


@fixture
def random():
    return np.random.default_rng(seed=42)


def at_origin(arr):
    return arr[int(arr.shape[0] / 2)]

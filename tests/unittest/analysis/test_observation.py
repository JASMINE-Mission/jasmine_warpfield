#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import Observation


def generate_observation(uncertainty=None):
    return Observation(
        xy=[[1.0, 2.0], [3.0, 4.0]],
        source_index=[1, 0],
        pointing_index=[0, 1],
        detector_index=[1, 1],
        uncertainty=uncertainty,
    )


def test_observation():
    observation = generate_observation([[0.1, 0.2], [0.3, 0.4]])

    assert isinstance(observation, zdx.Base)
    assert len(observation) == 2
    assert observation.xy == approx(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert observation.source_index == approx([1, 0])
    assert observation.pointing_index == approx([0, 1])
    assert observation.detector_index == approx([1, 1])
    assert observation.uncertainty == approx(
        jnp.array([[0.1, 0.2], [0.3, 0.4]]))
    assert len(jax.tree_util.tree_leaves(observation)) == 5


def test_observation_without_uncertainty():
    observation = generate_observation()

    assert observation.uncertainty is None
    assert len(jax.tree_util.tree_leaves(observation)) == 4


def test_observation_zodiax_update():
    observation = generate_observation()
    updated = observation.set('xy', jnp.zeros((2, 2)))

    assert observation.get('xy') == approx(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert updated.get('xy') == approx(jnp.zeros((2, 2)))


def test_observation_shape_validation():
    with raises(ValueError, match='N_observation'):
        Observation([1.0, 2.0], [0], [0], [0])
    with raises(ValueError, match='one-dimensional'):
        Observation([[1.0, 2.0]], [[0]], [0], [0])
    with raises(ValueError, match='length N_observation'):
        Observation([[1.0, 2.0]], [], [0], [0])
    with raises(ValueError, match='N_observation'):
        Observation([[1.0, 2.0]], [0], [0], [0], [0.1, 0.2])


def test_observation_index_validation():
    with raises(ValueError, match='contain integers'):
        Observation([[1.0, 2.0]], [0.0], [0], [0])
    with raises(ValueError, match='contain integers'):
        Observation([[1.0, 2.0]], [0], [0.0], [0])
    with raises(ValueError, match='contain integers'):
        Observation([[1.0, 2.0]], [0], [0], [0.0])

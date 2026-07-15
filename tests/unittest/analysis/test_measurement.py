#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import Measurement


def generate_measurement(uncertainty=None):
    return Measurement(
        xy=[[1.0, 2.0], [3.0, 4.0]],
        source_index=[1, 0],
        exposure_index=[0, 1],
        detector_index=[1, 1],
        uncertainty=uncertainty,
    )


def test_measurement():
    measurement = generate_measurement([[0.1, 0.2], [0.3, 0.4]])

    assert isinstance(measurement, zdx.Base)
    assert len(measurement) == 2
    assert measurement.xy == approx(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert measurement.source_index == approx([1, 0])
    assert measurement.exposure_index == approx([0, 1])
    assert measurement.detector_index == approx([1, 1])
    assert measurement.uncertainty == approx(
        jnp.array([[0.1, 0.2], [0.3, 0.4]]))
    assert len(jax.tree_util.tree_leaves(measurement)) == 5


def test_measurement_without_uncertainty():
    measurement = generate_measurement()

    assert measurement.uncertainty is None
    assert len(jax.tree_util.tree_leaves(measurement)) == 4


def test_measurement_is_immutable():
    measurement = generate_measurement()
    updated = measurement.set('xy', jnp.zeros((2, 2)))

    assert measurement.get('xy') == approx(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert updated.get('xy') == approx(jnp.zeros((2, 2)))


def test_measurement_shape_validation():
    with raises(ValueError, match='N_measurement'):
        Measurement([1.0, 2.0], [0], [0], [0])
    with raises(ValueError, match='one-dimensional'):
        Measurement([[1.0, 2.0]], [[0]], [0], [0])
    with raises(ValueError, match='length N_measurement'):
        Measurement([[1.0, 2.0]], [], [0], [0])
    with raises(ValueError, match='N_measurement'):
        Measurement([[1.0, 2.0]], [0], [0], [0], [0.1, 0.2])


def test_measurement_index_validation():
    with raises(ValueError, match='contain integers'):
        Measurement([[1.0, 2.0]], [0.0], [0], [0])
    with raises(ValueError, match='contain integers'):
        Measurement([[1.0, 2.0]], [0], [0.0], [0])
    with raises(ValueError, match='contain integers'):
        Measurement([[1.0, 2.0]], [0], [0], [0.0])

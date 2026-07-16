#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from astropy.table import QTable
import astropy.units as u
from pytest import approx, raises
import zodiax as zdx

from warpfield import Measurement


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


def test_measurement_qtable_roundtrip():
    measurement = generate_measurement(
        [[0.1, 0.2], [0.3, 0.4]])

    table = measurement.to_qtable()
    restored = Measurement.from_qtable(table)

    assert table.colnames == [
        'measurement_id',
        'x',
        'y',
        'source_id',
        'exposure_id',
        'detector_id',
        'x_error',
        'y_error',
    ]
    assert table['measurement_id'].tolist() == [0, 1]
    assert table['x'].unit == u.pix
    assert restored.xy == approx(measurement.xy)
    assert restored.source_index == approx(measurement.source_index)
    assert restored.exposure_index == approx(measurement.exposure_index)
    assert restored.detector_index == approx(measurement.detector_index)
    assert restored.uncertainty == approx(measurement.uncertainty)


def test_measurement_qtable_without_uncertainty():
    restored = Measurement.from_qtable(
        generate_measurement().to_qtable())

    assert restored.uncertainty is None


def test_measurement_qtable_validation():
    table = generate_measurement().to_qtable()
    table.remove_column('source_id')
    with raises(ValueError, match='missing required columns'):
        Measurement.from_qtable(table)

    table = generate_measurement().to_qtable()
    table['x_error'] = [0.1, 0.2] * u.pix
    with raises(ValueError, match='both x_error and y_error'):
        Measurement.from_qtable(table)

    table = QTable({
        'x': [1.0] * u.m,
        'y': [2.0] * u.pix,
        'source_id': [0],
        'exposure_id': [0],
        'detector_id': [0],
    })
    with raises(ValueError, match='pixel units'):
        Measurement.from_qtable(table)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
from pytest import approx, raises

from warpfield import (
    Detector,
    Exposure,
    Measurement,
    Pointing,
    Simulator,
    SourceCatalog,
    Telescope,
)
from warpfield.calibration import IdentityCalibration
from warpfield.projection import GnomonicProjection
from warpfield.simulator import ErrorGenerator, UniformError


class ConstantError(ErrorGenerator):
    def __call__(self, source, seed, shape):
        assert isinstance(source, SourceCatalog)
        assert len(source) == shape[0]
        return np.full(shape, seed / 100)

    def uncertainty(self, source, shape):
        assert isinstance(source, SourceCatalog)
        assert len(source) == shape[0]
        return np.full(shape, 0.25)


class InvalidError(ErrorGenerator):
    def __init__(self, error, uncertainty):
        self.error = error
        self._uncertainty = uncertainty

    def __call__(self, source, seed, shape):
        del source, seed, shape
        return self.error

    def uncertainty(self, source, shape):
        del source, shape
        return self._uncertainty


def generate_simulator(detectors=None, imaging_radius=None):
    if detectors is None:
        detectors = (
            Detector(0.0, [0.0, 0.0], [1.0, 1.0], shape=(2, 4)),
        )
    return Simulator(
        GnomonicProjection(),
        [1.0, 1.0],
        detectors,
        imaging_radius=imaging_radius,
    )


def test_simulator_masks():
    simulator = generate_simulator()

    assert isinstance(simulator, Telescope)
    assert simulator.fov_radius == approx(jnp.sqrt(5.0))
    assert simulator.detector_masks[0](jnp.array([
        [0.0, 0.0],
        [2.0, 4.0],
        [2.1, 0.0],
    ])).tolist() == [True, True, False]
    assert simulator.fov_mask(jnp.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 2.0],
    ])).tolist() == [True, True, False]
    assert len(jax.tree_util.tree_leaves(simulator)) == 4


def test_simulator_encloses_offset_rotated_detectors():
    detectors = (
        Detector(
            90.0, [3.0, 4.0], [0.5, 1.0], shape=(2, 4)),
    )
    simulator = generate_simulator(detectors)
    corners = jnp.array([
        [1.0, 3.5],
        [1.0, 4.5],
        [5.0, 4.5],
        [5.0, 3.5],
    ])

    assert simulator.fov_mask(corners).all()
    assert simulator.fov_radius == approx(jnp.linalg.norm(corners[2]))


def test_simulator_imaging_radius_override():
    simulator = generate_simulator(imaging_radius=1.5)

    assert simulator.fov_radius == approx(1.5)


def test_simulator_observe():
    detectors = (
        Detector(0.0, [0.0, 0.0], [1.0, 1.0], shape=(2, 2)),
    )
    simulator = generate_simulator(detectors)
    source = SourceCatalog(
        ra=[0.0, 0.5, 2.0],
        dec=[0.0, 0.0, 0.0],
    )
    exposure = Exposure(
        Pointing([0.0], [0.0], [0.0]),
        IdentityCalibration(),
    )

    measurement = simulator.observe(source, exposure)

    assert isinstance(measurement, Measurement)
    assert len(measurement) == 2
    assert measurement.xy == approx(
        jnp.array([[1.0, 1.0], [0.4999873, 1.0]]))
    assert measurement.source_index.tolist() == [0, 1]
    assert measurement.exposure_index.tolist() == [0, 0]
    assert measurement.detector_index.tolist() == [0, 0]


def test_simulator_observe_overlapping_detectors():
    detectors = (
        Detector(0.0, [0.0, 0.0], [1.0, 1.0], shape=(2, 2)),
        Detector(0.0, [0.0, 0.0], [1.0, 1.0], shape=(2, 2)),
    )
    simulator = generate_simulator(detectors)
    source = SourceCatalog([0.0], [0.0])
    exposure = Exposure(
        Pointing([0.0], [0.0], [0.0]),
        IdentityCalibration(),
    )

    measurement = simulator.observe(source, exposure)

    assert len(measurement) == 2
    assert measurement.detector_index.tolist() == [0, 1]


def test_simulator_observe_with_gaussian_error():
    simulator = generate_simulator()
    source = SourceCatalog([0.0], [0.0])
    exposure = Exposure(Pointing([0.0], [0.0], [0.0]))
    ideal = simulator.observe(source, exposure)

    default = simulator.observe(source, exposure, error=0.2)
    first = simulator.observe(source, exposure, error=0.2, seed=0)
    second = simulator.observe(source, exposure, error=0.2, seed=0)
    different = simulator.observe(source, exposure, error=0.2, seed=456)

    assert default.xy == approx(first.xy)
    assert first.xy == approx(second.xy)
    assert first.xy != approx(ideal.xy)
    assert first.xy != approx(different.xy)
    assert first.uncertainty == approx(jnp.full((1, 2), 0.2))


def test_uniform_error():
    generator = UniformError(0.2)
    source = SourceCatalog([0.0], [0.0])

    error = generator(source, 123, (1, 2))
    uncertainty = generator.uncertainty(source, (1, 2))

    assert error.shape == (1, 2)
    assert uncertainty == approx(np.full((1, 2), 0.2))


def test_simulator_observe_with_error_generator():
    simulator = generate_simulator()
    source = SourceCatalog([0.0], [0.0])
    exposure = Exposure(Pointing([0.0], [0.0], [0.0]))
    ideal = simulator.observe(source, exposure)

    measurement = simulator.observe(
        source,
        exposure,
        error=ConstantError(),
        seed=10,
    )

    assert measurement.xy == approx(ideal.xy + 0.1)
    assert measurement.uncertainty == approx(jnp.full((1, 2), 0.25))


def test_simulator_validation():
    simulator = generate_simulator()
    source = SourceCatalog([0.0], [0.0])
    exposure = Exposure(Pointing([0.0], [0.0], [0.0]))

    with raises(TypeError, match='SourceCatalog'):
        simulator.observe(object(), object())
    with raises(TypeError, match='Exposure'):
        simulator.observe(source, object())
    with raises(TypeError, match='error'):
        simulator.observe(source, exposure, error=object())
    with raises(ValueError, match='finite and non-negative'):
        simulator.observe(source, exposure, error=-1.0)
    with raises(TypeError, match='standard_deviation'):
        UniformError('invalid')
    with raises(ValueError, match='finite and non-negative'):
        UniformError(np.inf)
    with raises(TypeError, match='seed'):
        simulator.observe(source, exposure, error=1.0, seed=1.5)
    with raises(ValueError, match='non-negative'):
        simulator.observe(source, exposure, error=1.0, seed=-1)
    with raises(ValueError, match='errors should have shape'):
        simulator.observe(
            source,
            exposure,
            error=InvalidError([0.0], [[1.0, 1.0]]),
        )
    with raises(ValueError, match='uncertainties should have shape'):
        simulator.observe(
            source,
            exposure,
            error=InvalidError([[0.0, 0.0]], [1.0]),
        )
    with raises(ValueError, match='errors should be finite'):
        simulator.observe(
            source,
            exposure,
            error=InvalidError([[np.nan, 0.0]], [[1.0, 1.0]]),
        )
    with raises(ValueError, match='finite and non-negative'):
        simulator.observe(
            source,
            exposure,
            error=InvalidError([[0.0, 0.0]], [[-1.0, 1.0]]),
        )

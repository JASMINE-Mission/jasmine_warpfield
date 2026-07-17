#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
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


def test_simulator_validation():
    simulator = generate_simulator()

    with raises(TypeError, match='SourceCatalog'):
        simulator.observe(object(), object())
    with raises(TypeError, match='Exposure'):
        simulator.observe(SourceCatalog([0.0], [0.0]), object())

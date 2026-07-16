#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from astropy.table import QTable
import astropy.units as u
from pytest import approx, raises
import zodiax as zdx

from warpfield import Exposure, Pointing
from warpfield.calibration import (
    IdentityCalibration,
    ScaleCalibration,
)


def generate_pointing():
    return Pointing(
        ra=[10.0, 20.0],
        dec=[-10.0, -20.0],
        position_angle=[1.0, 2.0],
    )


def test_exposure():
    exposure = Exposure(
        generate_pointing(),
        ScaleCalibration(jnp.log(jnp.array([1.0, 1.1]))),
    )
    ra, dec, position_angle, factor = exposure.take(jnp.array([1, 0]))

    assert isinstance(exposure, zdx.Base)
    assert len(exposure) == 2
    assert ra == approx([20.0, 10.0])
    assert dec == approx([-20.0, -10.0])
    assert position_angle == approx([2.0, 1.0])
    assert factor == approx(jnp.array([[1.1], [1.0]]))
    assert len(jax.tree_util.tree_leaves(exposure)) == 4


def test_identity_calibrated_exposure():
    exposure = Exposure(generate_pointing(), IdentityCalibration())

    assert exposure.take(jnp.array([0, 1]))[-1] == approx(
        jnp.ones((2, 1)))


def test_default_identity_calibration():
    exposure = Exposure(generate_pointing())

    assert isinstance(exposure.calibration, IdentityCalibration)
    assert exposure.take(jnp.array([0, 1]))[-1] == approx(
        jnp.ones((2, 1)))


def test_exposure_index_accessor():
    exposure = Exposure(
        generate_pointing(),
        ScaleCalibration([0.0, 0.1]),
    )

    selected = exposure[1]

    assert isinstance(selected, Exposure)
    assert len(selected) == 1
    assert selected.pointing.ra == approx([20.0])
    assert selected.pointing.dec == approx([-20.0])
    assert selected.pointing.position_angle == approx([2.0])
    assert selected.calibration.coefficient == approx([0.1])


def test_identity_exposure_index_accessor():
    exposure = Exposure(generate_pointing(), IdentityCalibration())

    selected = exposure[:1]

    assert len(selected) == 1
    assert isinstance(selected.calibration, IdentityCalibration)


def test_exposure_iteration():
    exposure = Exposure(
        generate_pointing(),
        ScaleCalibration([0.0, 0.1]),
    )

    items = list(exposure)

    assert len(items) == 2
    assert all(isinstance(item, Exposure) for item in items)
    assert all(len(item) == 1 for item in items)
    assert items[0].pointing.ra == approx([10.0])
    assert items[1].calibration.coefficient == approx([0.1])


def test_exposure_zodiax_update():
    exposure = Exposure(generate_pointing(), ScaleCalibration([0.0, 0.1]))
    updated = exposure.set(
        'calibration.coefficient', jnp.array([0.2, 0.3]))

    assert exposure.get('calibration.coefficient') == approx([0.0, 0.1])
    assert updated.get('calibration.coefficient') == approx([0.2, 0.3])


def test_exposure_validation():
    pointing = generate_pointing()

    with raises(TypeError, match='Pointing'):
        Exposure(object(), IdentityCalibration())
    with raises(TypeError, match='Calibration'):
        Exposure(pointing, object())
    with raises(ValueError, match='same length'):
        Exposure(pointing, ScaleCalibration([0.0]))


def test_scale_calibrated_exposure_qtable_roundtrip():
    exposure = Exposure(
        generate_pointing(),
        ScaleCalibration([0.0, 0.1]),
    )

    table = exposure.to_qtable()
    restored = Exposure.from_qtable(table)

    assert table.meta['calibration'] == 'scale'
    assert table['scale_coefficient'].unit == u.dimensionless_unscaled
    assert isinstance(restored.calibration, ScaleCalibration)
    assert restored.pointing.ra == approx(exposure.pointing.ra)
    assert restored.calibration.coefficient == approx([0.0, 0.1])


def test_identity_calibrated_exposure_qtable_roundtrip():
    exposure = Exposure(generate_pointing(), IdentityCalibration())

    table = exposure.to_qtable()
    restored = Exposure.from_qtable(table)

    assert table.meta['calibration'] == 'identity'
    assert isinstance(restored.calibration, IdentityCalibration)


def test_exposure_qtable_validation():
    table = generate_pointing().to_qtable()
    table.meta['calibration'] = 'scale'
    with raises(ValueError, match='scale_coefficient'):
        Exposure.from_qtable(table)

    table = QTable(table)
    table.meta['calibration'] = 'unknown'
    with raises(ValueError, match='Unsupported calibration'):
        Exposure.from_qtable(table)

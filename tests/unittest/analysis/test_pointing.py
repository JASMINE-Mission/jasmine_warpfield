#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.units as u
from pytest import approx, raises
import zodiax as zdx

from warpfield import Pointing


def generate_pointing():
    return Pointing(
        ra=[10.0, 20.0],
        dec=[-10.0, -20.0],
        position_angle=[1.0, 2.0],
    )


def test_pointing():
    pointing = generate_pointing()
    ra, dec, position_angle = pointing.take(jnp.array([1, 0]))

    assert isinstance(pointing, zdx.Base)
    assert len(pointing) == 2
    assert ra == approx([20.0, 10.0])
    assert dec == approx([-20.0, -10.0])
    assert position_angle == approx([2.0, 1.0])
    assert len(jax.tree_util.tree_leaves(pointing)) == 3


def test_pointing_zodiax_update():
    pointing = generate_pointing()
    updated = pointing.add('position_angle', 1.0)

    assert pointing.get('position_angle') == approx([1.0, 2.0])
    assert updated.get('position_angle') == approx([2.0, 3.0])


def test_pointing_iteration():
    items = list(generate_pointing())

    assert len(items) == 2
    assert all(isinstance(item, Pointing) for item in items)
    assert all(len(item) == 1 for item in items)
    assert items[0].ra == approx([10.0])
    assert items[1].position_angle == approx([2.0])


def test_pointing_from_galactic_coord():
    coordinate = SkyCoord(
        l=[0.0, 10.0] * u.deg,
        b=[0.0, 5.0] * u.deg,
        frame='galactic',
    )
    position_angle = [0.0, 30.0] * u.deg

    pointing = Pointing.from_coord(
        'galactic',
        coordinate.l,
        coordinate.b,
        position_angle,
    )

    direction = coordinate.directional_offset_by(
        position_angle,
        1 * u.arcsec,
    ).icrs
    icrs = coordinate.icrs
    expected_angle = icrs.position_angle(direction).degree
    assert pointing.ra == approx(icrs.ra.degree)
    assert pointing.dec == approx(icrs.dec.degree)
    assert pointing.position_angle == approx(expected_angle)


def test_pointing_from_scalar_coord():
    pointing = Pointing.from_coord('icrs', 10.0, 20.0, 30.0 * u.deg)

    assert len(pointing) == 1
    assert pointing.ra == approx([10.0])
    assert pointing.dec == approx([20.0])
    assert pointing.position_angle == approx([30.0], abs=1e-8)


def test_pointing_from_coord_validation():
    with raises(ValueError, match='frame'):
        Pointing.from_coord('ecliptic', 10.0, 20.0, 30.0)
    with raises(ValueError, match='compatible shapes'):
        Pointing.from_coord(
            'icrs',
            [10.0, 20.0],
            [30.0, 40.0, 50.0],
            0.0,
        )


def test_pointing_shape_validation():
    with raises(ValueError, match='same shape'):
        Pointing([1.0], [2.0, 3.0], [4.0])
    with raises(ValueError, match='same shape'):
        Pointing([1.0], [2.0], [3.0, 4.0])


def test_pointing_qtable_roundtrip():
    pointing = generate_pointing()

    table = pointing.to_qtable()
    restored = Pointing.from_qtable(table)

    assert table.colnames == [
        'exposure_id', 'ra', 'dec', 'position_angle']
    assert table['exposure_id'].tolist() == [0, 1]
    assert table['ra'].unit == u.deg
    assert restored.ra == approx(pointing.ra)
    assert restored.dec == approx(pointing.dec)
    assert restored.position_angle == approx(pointing.position_angle)


def test_pointing_qtable_validation():
    with raises(ValueError, match='missing required columns'):
        Pointing.from_qtable(QTable({'ra': [1.0] * u.deg}))
    with raises(ValueError, match='angular units'):
        Pointing.from_qtable(QTable({
            'ra': [1.0] * u.m,
            'dec': [2.0] * u.deg,
            'position_angle': [3.0] * u.deg,
        }))

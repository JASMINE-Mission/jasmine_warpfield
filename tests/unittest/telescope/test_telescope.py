#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
from astropy.coordinates import SkyCoord
import astropy.units as u

from warpfield.telescope import Telescope

from .test_optics import target


@fixture
def telescope():
    ra = 10.0 * u.degree
    dec = 30.0 * u.degree
    pointing = SkyCoord(ra, dec, frame='icrs')
    position_angle = 0.0 * u.degree
    return Telescope(pointing, position_angle)


def test_build_telescope(telescope):
    assert telescope.pointing is not None
    assert telescope.optics is not None
    assert len(telescope.detectors) > 0


def test_telescope_pointing(telescope):
    assert telescope.pointing.icrs.ra.deg == approx(10.0)
    assert telescope.pointing.icrs.dec.deg == approx(30.0)
    assert telescope.position_angle.to_value('degree') == approx(0.0)


def test_telescope_observe(telescope, target):
    result = telescope.observe(target)
    assert len(result[0]) == len(target)

    result = telescope.observe(target, stack=True)
    assert len(result) == len(target)


def test_footprints(telescope):
    fp = telescope.get_footprints('icrs')
    assert len(fp) == 1  # number of detectors
    assert len(fp[0]) == 4 + 1  # vertices for a closed square

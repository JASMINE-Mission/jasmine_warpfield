#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
from astropy.coordinates import SkyCoord
import astropy.units as u

from warpfield.telescope import *


@fixture
def pointing():
    return SkyCoord(
        ra = 83.6333 * u.deg,
        dec = 22.0133 * u.deg)


@fixture
def position_angle():
    return 0.0 * u.deg


def test_jasmine(pointing, position_angle):
    jasmine = get_jasmine(pointing, position_angle, octagonal=False)

    focal_length = jasmine.optics.focal_length.to_value('meter')
    assert focal_length == approx(4.86)

    num_detector = len(jasmine.detectors)
    assert num_detector == 4

    fov_radius = jasmine.optics.field_of_view_radius.to_value('degree')
    assert fov_radius == approx(0.34512, abs=1e-5)


def test_jasmine_masked(pointing, position_angle):
    jasmine = get_jasmine(pointing, position_angle, octagonal=True)

    focal_length = jasmine.optics.focal_length.to_value('meter')
    assert focal_length == approx(4.86)

    num_detector = len(jasmine.detectors)
    assert num_detector == 4

    fov_radius = jasmine.optics.field_of_view_radius.to_value('degree')
    assert fov_radius < 0.34512

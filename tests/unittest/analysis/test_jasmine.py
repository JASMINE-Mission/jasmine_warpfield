#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pytest import approx, raises

from warpfield.distortion import IdentityDistortion, SIPDistortion
from warpfield.instrument.jasmine import get_jasmine
from warpfield.simulator import Simulator
from warpfield.telescope import Telescope
from warpfield.utils import plate_scale
import astropy.units as u


def test_get_jasmine():
    jasmine = get_jasmine()

    assert type(jasmine) is Telescope
    assert len(jasmine.detectors) == 4
    assert jasmine.optics.plate_scale == approx(
        plate_scale(4.86 * u.m)
    )
    assert jasmine.optics.imaging_radius == approx(
        np.sqrt(2) * 20.7
    )

    for detector in jasmine.detectors:
        assert detector.shape == (1920, 1920)
        assert detector.pixel_scale == approx([0.01, 0.01])

    assert [float(detector.rotation) for detector in jasmine.detectors] == [
        0.0,
        90.0,
        180.0,
        270.0,
    ]
    offsets = np.asarray([
        detector.offset for detector in jasmine.detectors
    ])
    assert offsets == approx(np.array([
            [-11.1, -11.1],
            [+11.1, -11.1],
            [+11.1, +11.1],
            [-11.1, +11.1],
        ]))


def test_get_jasmine_simulator():
    jasmine = get_jasmine(simulator=True)

    assert isinstance(jasmine, Simulator)
    assert len(jasmine.detectors) == 4


def test_get_jasmine_distortion():
    distortion = SIPDistortion(
        coeff_x=np.zeros(18),
        coeff_y=np.zeros(18),
    )

    jasmine = get_jasmine(distortion=distortion)

    assert jasmine.optics.distortion is distortion


def test_get_jasmine_validation():
    with raises(TypeError, match='Distortion'):
        get_jasmine(distortion=object())
    with raises(TypeError, match='boolean'):
        get_jasmine(simulator=1)
    assert isinstance(
        get_jasmine().optics.distortion,
        IdentityDistortion,
    )

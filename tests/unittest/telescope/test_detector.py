#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
from astropy.table import QTable
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
import astropy.units as u
import numpy as np

from warpfield.telescope.source import FocalPlanePositionTable
from warpfield.telescope.detector import Detector


@fixture
def detector():
    naxis1 = 1920
    naxis2 = 1920
    pixel_scale = 10 * u.um
    return Detector(naxis1, naxis2, pixel_scale)


@fixture
def fp_position():
    tics = np.arange(-9000, 9001, 1000)
    x, y = np.meshgrid(tics, tics)
    return FocalPlanePositionTable(
        QTable([
            x.ravel() * u.um,
            y.ravel() * u.um,
        ], names=[
            'x', 'y',
        ]))


def test_build_detector(detector):
    assert detector.naxis1 == 1920
    assert detector.naxis2 == 1920
    assert detector.pixel_scale == 10 * u.um
    assert detector.detector_origin == approx([-9600, -9600])


def test_detector_footprint(detector):
    assert isinstance(detector.footprint_as_patch, Rectangle)
    assert isinstance(detector.footprint_as_polygon, Polygon)


def test_detector_capture(detector, fp_position):
    det_position = detector.capture(fp_position)
    assert len(det_position) == len(fp_position)

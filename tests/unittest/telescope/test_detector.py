#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from shapely.geometry import Polygon
import astropy.units as u

from warpfield.telescope.source import FocalPlanePositionTable
from warpfield.telescope.detector import Detector
from .test_source import focalplane


@fixture
def detector():
    naxis1 = 1920
    naxis2 = 1920
    pixel_scale = 10 * u.um
    return Detector(naxis1, naxis2, pixel_scale)


@fixture
def fptable(focalplane):
    return FocalPlanePositionTable(focalplane)


def test_build_detector(detector):
    assert detector.naxis1 == 1920
    assert detector.naxis2 == 1920
    assert detector.pixel_scale == 10 * u.um


def test_detector_geometry(detector):
    assert detector.detector_origin.to_value('um') == approx([-9600, -9600])


def test_detector_footprint(detector):
    assert isinstance(detector.get_footprint_as_patch(), Rectangle)
    assert isinstance(detector.get_footprint_as_polygon(), Polygon)
    assert isinstance(detector.get_first_line_as_patch(), Line2D)


def test_detector_capture(detector, fptable):
    det_position = detector.capture(fptable)
    assert len(det_position) == len(fptable)

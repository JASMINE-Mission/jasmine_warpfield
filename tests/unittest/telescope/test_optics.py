#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from matplotlib.patches import Polygon
import astropy.units as u
import numpy as np

from warpfield.telescope.source import SourceTable
from warpfield.telescope.optics import Optics


@fixture
def optics():
    ra = 10.0 * u.degree
    dec = 30.0 * u.degree
    pointing = SkyCoord(ra, dec, frame='icrs')
    position_angle = 0.0 * u.degree
    focal_length = 4.00 * u.meter
    diameter = 0.3 * u.meter
    return Optics(pointing, position_angle, focal_length, diameter)


@fixture
def target():
    tics = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    na, nd = np.meshgrid(tics, tics)
    a = 10 + na.ravel()
    d = 30 + nd.ravel()
    return SourceTable(
        QTable([
            a * u.degree,
            d * u.degree,
            np.zeros_like(a) * u.mas / u.year,
            np.zeros_like(d) * u.mas / u.year,
            np.ones_like(a) * u.uas,
            2016.0 * np.ones_like(a) * u.year,
        ], names=[
            'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'ref_epoch',
        ]))


def test_build_optics_pointing(optics):
    assert optics.pointing.icrs.ra.deg == approx(10.0)
    assert optics.pointing.icrs.dec.deg == approx(30.0)


def test_build_optics_focal_length(optics):
    assert optics.focal_length.to_value('meter') == approx(4.00)


def test_build_optics_focal_diamter(optics):
    assert optics.diameter.to_value('meter') == approx(0.3)


def test_build_optics_focal_plane_radius(optics):
    assert optics.focal_plane_radius.to_value('cm') == approx(3.0)


def test_build_optics_field_of_view_radius(optics):
    assert optics.field_of_view_radius.to_value('degree') == approx(0.43, 0.1)


def test_optics_projection(optics):
    proj = optics.projection
    assert proj.wcs.cd[1, 1] == approx(optics.scale.to_value('degree/um'))

    orig = [[0, 0], ]
    sky = proj.all_pix2world(orig, 0)
    assert sky[0][0] == approx(optics.pointing.icrs.ra.deg)
    assert sky[0][1] == approx(optics.pointing.icrs.dec.deg)

    center = [[optics.pointing.ra.deg, optics.pointing.dec.deg], ]
    pix = proj.all_world2pix(center, 0)
    assert pix[0] == approx(0.0, abs=1e-8)


def test_optics_patch(optics):
    assert isinstance(optics.get_fov_patch(), Polygon)


def test_optics_imaging(optics, target):
    table = optics.imaging(target)
    assert len(table) == len(target)

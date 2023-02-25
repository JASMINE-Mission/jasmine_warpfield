#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, raises, fixture
from astropy.coordinates import SkyCoord
import astropy.units as u

from warpfield.telescope.util import *


@fixture
def target():
    return SkyCoord(
        ra = 83.6333 * u.deg,
        dec = 22.0133 * u.deg)


def test_eprint():
    eprint('message')


def test_estimate_frame_from_ctype():
    ctypes = ('RA---TAN', 'DEC--TAN')
    frame = estimate_frame_from_ctype(ctypes)
    assert frame == 'icrs'

    ctypes = ('GLON-TAN', 'GLAT-TAN')
    frame = estimate_frame_from_ctype(ctypes)
    assert frame == 'galactic'

    ctypes = ('ELON-TAN', 'ELAT-TAN')
    with raises(ValueError):
        estimate_frame_from_ctype(ctypes)


def test_get_axis_name():
    xl, yl = get_axis_name('galactic')
    assert xl == 'Galactic Longitude'
    assert yl == 'Galactic Latitude'

    xl, yl = get_axis_name('icrs')
    assert xl == 'Right Ascension'
    assert yl == 'Declination'


def test_frame_conversion(target):
    converted = frame_conversion(target, 'icrs')
    assert converted.frame.name == 'icrs'

    converted = frame_conversion(target, 'gcrs')
    assert converted.frame.name == 'gcrs'

    converted = frame_conversion(target, 'gcgrs')
    assert converted.frame.name == 'gcgrs'


def test_get_projection(target):
    proj = get_projection(target)
    assert proj.wcs.crval[0] == approx(target.icrs.ra.deg)
    assert proj.wcs.crval[1] == approx(target.icrs.dec.deg)

    proj = get_projection(target.galactic)
    assert proj.wcs.crval[0] == approx(target.galactic.l.deg)
    assert proj.wcs.crval[1] == approx(target.galactic.b.deg)

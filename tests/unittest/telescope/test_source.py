#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tempfile import NamedTemporaryFile as tempfile
from pytest import approx, fixture
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
import astropy.units as u
import numpy as np

from warpfield.telescope.source import SourceTable
from warpfield.telescope.source import FocalPlanePositionTable
from warpfield.telescope.source import DetectorPositionTable
from warpfield.telescope.source import retrieve_gaia_sources


@fixture
def source():
    return QTable([
        np.array([0, 1, 2]),
        np.array([+0.0, +0.1, +0.2]) * u.degree,
        np.array([-0.1, +0.0, +0.1]) * u.degree,
        np.array([+0.0, +0.1, -1.0]) * u.mas / u.year,
        np.array([+0.0, -0.1, -1.0]) * u.mas / u.year,
        np.array([1.0, 2.0, 3.0]) * u.mas,
        np.array([2016.0, 2016.0, 2016.0]) * u.year,
    ], names=[
        'source_id', 'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'ref_epoch',
    ])


@fixture
def source_simple():
    return QTable([
        np.array([0, 1, 2]),
        np.array([+0.0, +0.1, +0.2]) * u.degree,
        np.array([-0.1, +0.0, +0.1]) * u.degree,
    ], names=[
        'source_id', 'ra', 'dec',
    ])


def generate_grid(a, b, n):
    tics = np.arange(a, b, n)
    x, y = np.meshgrid(tics, tics)
    sid = np.arange(x.size)
    return x, y, sid


@fixture
def focalplane():
    x, y, source_id = generate_grid(-9000, 9001, 1000)
    return QTable([
        source_id,
        x.ravel() * u.um,
        y.ravel() * u.um,
    ], names=[
        'source_id', 'x', 'y',
    ])


@fixture
def detector():
    x, y, source_id = generate_grid(0, 2001, 20)
    return QTable([
        source_id,
        x.ravel(),
        y.ravel(),
    ], names=[
        'source_id', 'nx', 'ny',
    ])


def test_build_sourcetable(source):
    st = SourceTable(source)

    assert st.table['ra'].to_value('degree') == approx([0.0, 0.1, 0.2])
    assert st.table['dec'].to_value('degree') == approx([-0.1, 0.0, 0.1])
    assert st.skycoord.ra.deg == approx([0.0, 0.1, 0.2])
    assert st.skycoord.dec.deg == approx([-0.1, 0.0, 0.1])


def test_build_sourcetable_simple(source_simple):
    st = SourceTable(source_simple)

    assert st.table['ra'].to_value('degree') == approx([0.0, 0.1, 0.2])
    assert st.table['dec'].to_value('degree') == approx([-0.1, 0.0, 0.1])
    assert st.skycoord.ra.deg == approx([0.0, 0.1, 0.2])
    assert st.skycoord.dec.deg == approx([-0.1, 0.0, 0.1])


def test_io_sourcetable(source):
    st = SourceTable(source)
    with tempfile() as fp:
        st.writeto(fp.name, overwrite=True)
        sx = SourceTable.from_fitsfile(fp.name)

    assert isinstance(sx, SourceTable)
    residual = (st.table['ra'] - sx.table['ra']).to_value('degree')
    assert residual == approx(0.0)


def test_build_focalplanepositiontable(focalplane):
    fpt = FocalPlanePositionTable(focalplane)

    assert len(fpt) == 19 * 19


def test_io_focalplanepositiontable(focalplane):
    fpt = FocalPlanePositionTable(focalplane)
    with tempfile() as fp:
        fpt.writeto(fp.name, overwrite=True)
        fpx = FocalPlanePositionTable.from_fitsfile(fp.name)

    assert isinstance(fpx, FocalPlanePositionTable)
    residual = (fpt.table['x'] - fpx.table['x']).to_value('um')
    assert residual == approx(0.0)


def test_build_detectorpositiontable(detector):
    dpt = DetectorPositionTable(detector)

    assert len(dpt) == 101 * 101


def test_apply_space_motion(source):
    st = SourceTable(source)
    epoch = Time(2026.0, format='decimalyear', scale='tcb')
    delta = 10 * u.year
    st.apply_space_motion(epoch)
    ra_est = source['ra'] + delta * source['pmra']
    dec_est = source['dec'] + delta * source['pmdec']

    d_ra = st.skycoord.ra - ra_est
    d_dec = st.skycoord.dec - dec_est
    assert d_ra.to_value('degree') == approx(0.0, abs=1e-8)
    assert d_dec.to_value('degree') == approx(0.0, abs=1e-8)


def test_gaia_source():
    ra = 0.0 * u.degree
    dec = 0.0 * u.degree
    radius = 0.1 * u.degree
    pointing = SkyCoord(ra, dec, frame='icrs')
    st = retrieve_gaia_sources(pointing, radius)

    assert len(st) > 0

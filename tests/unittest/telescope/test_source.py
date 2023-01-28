#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tempfile import NamedTemporaryFile as tempfile
from pytest import approx, fixture
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.units as u
import numpy as np

from warpfield.telescope.source import SourceTable
from warpfield.telescope.source import retrieve_gaia_sources


@fixture
def table():
    return QTable([
        np.array([0.0, 0.1, 0.2]) * u.degree,
        np.array([-0.1, 0.0, 0.1]) * u.degree,
        np.array([0.1, 0.1, 0.1]) * u.mas / u.year,
        np.array([0.1, 0.1, 0.1]) * u.mas / u.year,
        np.array([1.0, 2.0, 3.0]) * u.mas,
        np.array([2016.0, 2016.0, 2016.0]) * u.year,
    ], names=[
        'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'ref_epoch',
    ])


def test_build_sourcetable(table):
    st = SourceTable(table)
    assert st.table['ra'].to_value('degree') == approx([0.0, 0.1, 0.2])
    assert st.table['dec'].to_value('degree') == approx([-0.1, 0.0, 0.1])
    assert st.skycoord.ra.deg == approx([0.0, 0.1, 0.2])
    assert st.skycoord.dec.deg == approx([-0.1, 0.0, 0.1])


def test_io_sourcetable(table):
    st = SourceTable(table)
    with tempfile() as fp:
        st.writeto(fp.name, overwrite=True)
        sx = SourceTable.from_fitsfile(fp.name)

    residual = (st.table['ra'] - sx.table['ra']).to_value('degree')
    assert residual == approx(0.0)


def test_gaia_source():
    ra = 0.0 * u.degree
    dec = 0.0 * u.degree
    radius = 0.1 * u.degree
    pointing = SkyCoord(ra, dec, frame='icrs')
    st = retrieve_gaia_sources(pointing, radius)
    assert len(st) > 0

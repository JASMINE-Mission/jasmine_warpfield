#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
import astropy.units as u
import numpy as np
from pytest import approx, raises

from warpfield import AstrometricCatalog
from warpfield.query import compile_from_gaia, query_gaia
from warpfield.query import gaia


def _gaia_table():
    return QTable({
        'source_id': [10, 20],
        'ra': [10.0, 20.0] * u.deg,
        'ra_error': [0.1, 0.2] * u.mas,
        'dec': [-5.0, 15.0] * u.deg,
        'dec_error': [0.3, 0.4] * u.mas,
        'pmra': [1.0, 2.0] * u.mas / u.yr,
        'pmra_error': [0.5, 0.6] * u.mas / u.yr,
        'pmdec': [3.0, 4.0] * u.mas / u.yr,
        'pmdec_error': [0.7, 0.8] * u.mas / u.yr,
        'parallax': [5.0, 10.0] * u.mas,
        'parallax_error': [0.9, 1.0] * u.mas,
        'phot_g_mean_mag': [15.0, 16.0] * u.mag,
        'phot_g_mean_flux_over_error': [100.0, 50.0],
        'ref_epoch': [2016.0, 2016.0] * u.yr,
    })


def test_compile_from_gaia():
    catalog = compile_from_gaia(_gaia_table())

    assert isinstance(catalog, AstrometricCatalog)
    assert catalog.ra.to_value(u.deg) == approx([10.0, 20.0])
    assert catalog.dec.to_value(u.deg) == approx([-5.0, 15.0])
    assert catalog.pm_ra_cosdec.to_value(u.mas / u.yr) == approx([1.0, 2.0])
    assert catalog.pm_dec.to_value(u.mas / u.yr) == approx([3.0, 4.0])
    assert catalog.parallax.to_value(u.mas) == approx([5.0, 10.0])
    assert catalog.magnitude.to_value(u.mag) == approx([15.0, 16.0])
    assert catalog.magnitude_error.to_value(u.mag) == approx(
        2.5 / np.log(10) / np.array([100.0, 50.0]))
    assert catalog.ra_error.to_value(u.mas) == approx([0.1, 0.2])
    assert catalog.dec_error.to_value(u.mas) == approx([0.3, 0.4])
    assert catalog.pm_ra_cosdec_error.to_value(
        u.mas / u.yr) == approx([0.5, 0.6])
    assert catalog.pm_dec_error.to_value(
        u.mas / u.yr) == approx([0.7, 0.8])
    assert catalog.parallax_error.to_value(u.mas) == approx([0.9, 1.0])
    assert np.all(catalog.epoch == Time(
        [2016.0, 2016.0], format='jyear', scale='tcb'))


def test_compile_from_gaia_accepts_uppercase_columns():
    table = _gaia_table()
    for name in table.colnames:
        table.rename_column(name, name.upper())

    catalog = compile_from_gaia(table)

    assert len(catalog) == 2


def test_compile_from_gaia_validation():
    with raises(TypeError, match='Astropy Table'):
        compile_from_gaia({'ra': [10.0]})

    table = _gaia_table()
    table.remove_column('pmra')
    with raises(ValueError, match='missing required column: pmra'):
        compile_from_gaia(table)


def test_query_gaia(monkeypatch):
    captured = {}

    class Job:
        @staticmethod
        def get_results():
            return _gaia_table()

    def launch_job_async(query):
        captured['query'] = query
        return Job()

    monkeypatch.setattr(gaia.Gaia, 'launch_job_async', launch_job_async)

    catalog = query_gaia(
        SkyCoord(10.0, -5.0, unit='deg'),
        1.5 * u.deg,
        snr_limit=5.0,
        row_limit=25,
    )

    assert isinstance(catalog, AstrometricCatalog)
    assert 'SELECT TOP 25' in captured['query']
    assert 'gaiadr3.gaia_source' in captured['query']
    assert 'parallax_over_error > 5.0' in captured['query']
    assert 'phot_g_mean_mag' in captured['query']


def test_query_gaia_validation():
    center = SkyCoord(10.0, -5.0, unit='deg')

    with raises(ValueError, match='positive scalar'):
        query_gaia(center, 0.0)
    with raises(ValueError, match='positive integer'):
        query_gaia(center, 1.0, row_limit=0)
    with raises(ValueError, match='schema-qualified'):
        query_gaia(center, 1.0, catalog='invalid catalog')

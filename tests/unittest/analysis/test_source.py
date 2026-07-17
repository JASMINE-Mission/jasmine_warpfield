#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from astropy.coordinates import GCRS, ICRS
from astropy.coordinates import get_body_barycentric_posvel
from astropy.table import QTable, Table
from astropy.time import Time
import astropy.units as u
import numpy as np
from pytest import approx, mark, raises
import zodiax as zdx

from warpfield import AstrometricCatalog, SourceCatalog
from warpfield.observer import (
    BaryCentric,
    BCRSObserver,
    GeoCentric,
    GeoCentricN,
    Observatory,
    ObservatoryN,
    SSOObserver,
    SSOObserverN,
)


def test_source_catalog():
    source = SourceCatalog([1.0, 2.0], [3.0, 4.0])

    assert isinstance(source, zdx.Base)
    assert len(source) == 2
    assert source.take(jnp.array([1, 0]))[0] == approx([2.0, 1.0])
    assert source.take(jnp.array([1, 0]))[1] == approx([4.0, 3.0])
    assert len(jax.tree_util.tree_leaves(source)) == 2


def test_source_catalog_optional_attributes():
    source = SourceCatalog(
        [1.0, 2.0],
        [3.0, 4.0],
        magnitude=[15.0, 16.0],
        magnitude_error=[0.01, 0.02],
        ra_error=[1.0e-6, 2.0e-6],
        dec_error=[3.0e-6, 4.0e-6],
    )

    assert source.magnitude == approx([15.0, 16.0])
    assert source.magnitude_error == approx([0.01, 0.02])
    assert source.ra_error == approx([1.0e-6, 2.0e-6])
    assert source.dec_error == approx([3.0e-6, 4.0e-6])
    assert len(jax.tree_util.tree_leaves(source)) == 6
    assert source[1].magnitude == approx([16.0])


def test_source_catalog_zodiax_update():
    source = SourceCatalog([1.0, 2.0], [3.0, 4.0])
    updated = source.set('ra', jnp.array([5.0, 6.0]))

    assert source.get('ra') == approx([1.0, 2.0])
    assert updated.get('ra') == approx([5.0, 6.0])


def test_source_catalog_index_and_iteration():
    catalog = SourceCatalog([1.0, 2.0], [3.0, 4.0])

    selected = catalog[1]
    items = list(catalog)

    assert len(selected) == 1
    assert selected.ra == approx([2.0])
    assert len(items) == 2
    assert items[0].ra == approx([1.0])
    assert items[1].dec == approx([4.0])


def test_source_catalog_shape_validation():
    with raises(ValueError, match='one-dimensional'):
        SourceCatalog([[1.0]], [2.0])
    with raises(ValueError, match='same shape'):
        SourceCatalog([1.0], [2.0, 3.0])
    with raises(ValueError, match='magnitude.*same shape'):
        SourceCatalog([1.0], [2.0], magnitude=[15.0, 16.0])
    with raises(ValueError, match='magnitude_error.*non-negative'):
        SourceCatalog([1.0], [2.0], magnitude_error=[-0.1])


def test_source_catalog_qtable_roundtrip():
    source = SourceCatalog(
        [1.0, 2.0],
        [3.0, 4.0],
        magnitude=[15.0, 16.0],
        magnitude_error=[0.01, 0.02],
        ra_error=[1.0e-6, 2.0e-6],
        dec_error=[3.0e-6, 4.0e-6],
    )

    table = source.to_qtable()
    restored = SourceCatalog.from_qtable(table)

    assert isinstance(table, QTable)
    assert table.colnames[0] == 'source_id'
    assert np.issubdtype(table['source_id'].dtype, np.integer)
    assert table['source_id'] == approx([0, 1])
    assert table['ra'].unit == u.deg
    assert table['dec'].unit == u.deg
    assert table['magnitude'].unit == u.mag
    assert table['magnitude_error'].unit == u.mag
    assert table['ra_error'].unit == u.deg
    assert table['dec_error'].unit == u.deg
    assert restored.ra == approx(source.ra)
    assert restored.dec == approx(source.dec)
    assert restored.magnitude == approx(source.magnitude)
    assert restored.magnitude_error == approx(source.magnitude_error)
    assert restored.ra_error == approx(source.ra_error)
    assert restored.dec_error == approx(source.dec_error)


def test_source_catalog_qtable_validation():
    with raises(TypeError, match='QTable'):
        SourceCatalog.from_qtable(Table({'ra': [1.0], 'dec': [2.0]}))
    with raises(ValueError, match='missing required columns'):
        SourceCatalog.from_qtable(QTable({'ra': [1.0] * u.deg}))
    with raises(ValueError, match='angular units'):
        SourceCatalog.from_qtable(QTable({
            'ra': [1.0] * u.m,
            'dec': [2.0] * u.deg,
        }))


def test_astrometric_catalog():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )

    assert len(catalog) == 2
    assert not isinstance(catalog, zdx.Base)
    assert catalog.skycoord.frame.name == 'icrs'
    assert catalog.skycoord.obstime == Time('2016-01-01')
    assert catalog.magnitude is None
    assert catalog.parallax_error is None


def test_astrometric_catalog_optional_attributes():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
        magnitude=[15.0, 16.0] * u.mag,
        magnitude_error=[0.01, 0.02] * u.mag,
        ra_error=[0.1, 0.2] * u.mas,
        dec_error=[0.3, 0.4] * u.mas,
        pm_ra_cosdec_error=[0.5, 0.6] * u.mas / u.yr,
        pm_dec_error=[0.7, 0.8] * u.mas / u.yr,
        parallax_error=[0.9, 1.0] * u.mas,
    )

    assert catalog.magnitude.to_value(u.mag) == approx([15.0, 16.0])
    assert catalog.magnitude_error.to_value(u.mag) == approx([0.01, 0.02])
    assert catalog.ra_error.to_value(u.mas) == approx([0.1, 0.2])
    assert catalog.dec_error.to_value(u.mas) == approx([0.3, 0.4])
    assert catalog.pm_ra_cosdec_error.to_value(
        u.mas / u.yr) == approx([0.5, 0.6])
    assert catalog.pm_dec_error.to_value(
        u.mas / u.yr) == approx([0.7, 0.8])
    assert catalog.parallax_error.to_value(u.mas) == approx([0.9, 1.0])
    assert catalog[1].magnitude.to_value(u.mag) == approx([16.0])


def test_astrometric_catalog_index_and_iteration():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )

    selected = catalog[1]
    items = list(catalog)

    assert len(selected) == 1
    assert selected.ra.to_value(u.deg) == approx([20.0])
    assert selected.epoch == catalog.epoch
    assert len(items) == 2
    assert items[0].ra.to_value(u.deg) == approx([10.0])
    assert items[1].dec.to_value(u.deg) == approx([15.0])


def test_astrometric_catalog_propagate():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )
    observer = GeoCentric(Time('2025-01-01'))

    source = catalog.propagate(observer)
    expected = catalog.skycoord.apply_space_motion(
        new_obstime=observer.obstime,
    ).transform_to(GCRS(obstime=observer.obstime))

    assert isinstance(source, SourceCatalog)
    assert source.ra == approx(expected.ra.degree)
    assert source.dec == approx(expected.dec.degree)


def test_astrometric_catalog_propagate_optional_attributes():
    catalog = AstrometricCatalog(
        ra=[10.0] * u.deg,
        dec=[-5.0] * u.deg,
        pm_ra_cosdec=[1.0] * u.mas / u.yr,
        pm_dec=[3.0] * u.mas / u.yr,
        parallax=[5.0] * u.mas,
        epoch=Time('2016-01-01'),
        magnitude=[15.0] * u.mag,
        magnitude_error=[0.01] * u.mag,
        ra_error=[0.1] * u.mas,
        dec_error=[0.2] * u.mas,
    )

    source = catalog.propagate(GeoCentric(Time('2025-01-01')))

    assert source.magnitude == approx([15.0])
    assert source.magnitude_error == approx([0.01])
    assert source.ra_error == approx(
        [u.Quantity(0.1, u.mas).to_value(u.deg)])
    assert source.dec_error == approx(
        [u.Quantity(0.2, u.mas).to_value(u.deg)])


def test_astrometric_catalog_propagate_barycentric():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )
    observer = BaryCentric(Time('2025-01-01'))

    source = catalog.propagate(observer)
    expected = catalog.skycoord.apply_space_motion(
        new_obstime=observer.obstime,
    ).transform_to(ICRS())

    assert source.ra == approx(expected.ra.degree)
    assert source.dec == approx(expected.dec.degree)


def test_astrometric_catalog_propagate_geocentric_n():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )
    observer = GeoCentricN(Time('2025-01-01'))
    _, earth_velocity = get_body_barycentric_posvel(
        'earth',
        observer.obstime,
    )

    source = catalog.propagate(observer)
    expected = catalog.skycoord.apply_space_motion(
        new_obstime=observer.obstime,
    ).transform_to(GCRS(
        obstime=observer.obstime,
        obsgeovel=-earth_velocity,
    ))

    assert source.ra == approx(expected.ra.degree)
    assert source.dec == approx(expected.dec.degree)


def test_astrometric_catalog_propagate_bcrs_observer():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )
    observer = BCRSObserver(
        Time('2025-01-01'),
        [1.0, 2.0, 3.0] * u.au,
        [10.0, 20.0, 30.0] * u.km / u.s,
    )

    source = catalog.propagate(observer)
    expected = catalog.skycoord.apply_space_motion(
        new_obstime=observer.obstime,
    ).transform_to(GCRS(
        obstime=observer.obstime,
        obsgeoloc=observer.obsgeoloc,
        obsgeovel=observer.obsgeovel,
    ))

    assert source.ra == approx(expected.ra.degree)
    assert source.dec == approx(expected.dec.degree)


def test_astrometric_catalog_propagate_sso_observer():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )
    observer = SSOObserver(
        Time('2025-01-01'),
        phase=0.25,
        altitude=600 * u.km,
        ltan=6 * u.hourangle,
    )

    source = catalog.propagate(observer)
    expected = catalog.skycoord.apply_space_motion(
        new_obstime=observer.obstime,
    ).transform_to(GCRS(
        obstime=observer.obstime,
        obsgeoloc=observer.obsgeoloc,
        obsgeovel=observer.obsgeovel,
    ))

    assert source.ra == approx(expected.ra.degree)
    assert source.dec == approx(expected.dec.degree)


@mark.parametrize('frame', [Observatory, ObservatoryN])
def test_astrometric_catalog_propagate_observatory(frame):
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )
    observer = frame(
        Time('2025-01-01'),
        139 * u.deg,
        35 * u.deg,
        100 * u.m,
    )

    source = catalog.propagate(observer)
    expected = catalog.skycoord.apply_space_motion(
        new_obstime=observer.obstime,
    ).transform_to(GCRS(
        obstime=observer.obstime,
        obsgeoloc=observer.obsgeoloc,
        obsgeovel=observer.obsgeovel,
    ))

    assert source.ra == approx(expected.ra.degree)
    assert source.dec == approx(expected.dec.degree)


def test_astrometric_catalog_propagate_sso_observer_n():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
    )
    observer = SSOObserverN(
        Time('2025-01-01'),
        phase=0.25,
    )

    source = catalog.propagate(observer)
    expected = catalog.skycoord.apply_space_motion(
        new_obstime=observer.obstime,
    ).transform_to(GCRS(
        obstime=observer.obstime,
        obsgeoloc=observer.obsgeoloc,
        obsgeovel=observer.obsgeovel,
    ))

    assert source.ra == approx(expected.ra.degree)
    assert source.dec == approx(expected.dec.degree)


def test_astrometric_catalog_qtable_roundtrip():
    catalog = AstrometricCatalog(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        pm_ra_cosdec=[1.0, 2.0] * u.mas / u.yr,
        pm_dec=[3.0, 4.0] * u.mas / u.yr,
        parallax=[5.0, 10.0] * u.mas,
        epoch=Time('2016-01-01'),
        magnitude=[15.0, 16.0] * u.mag,
        magnitude_error=[0.01, 0.02] * u.mag,
        ra_error=[0.1, 0.2] * u.mas,
        dec_error=[0.3, 0.4] * u.mas,
        pm_ra_cosdec_error=[0.5, 0.6] * u.mas / u.yr,
        pm_dec_error=[0.7, 0.8] * u.mas / u.yr,
        parallax_error=[0.9, 1.0] * u.mas,
    )

    table = catalog.to_qtable()
    restored = AstrometricCatalog.from_qtable(table)

    assert isinstance(table, QTable)
    assert table.colnames[0] == 'source_id'
    assert np.issubdtype(table['source_id'].dtype, np.integer)
    assert table['source_id'] == approx([0, 1])
    assert isinstance(table['epoch'], Time)
    assert table['epoch'].shape == (2,)
    assert restored.ra.to_value(u.deg) == approx([10.0, 20.0])
    assert restored.dec.to_value(u.deg) == approx([-5.0, 15.0])
    assert restored.pm_ra_cosdec.to_value(u.mas / u.yr) == approx([1.0, 2.0])
    assert restored.pm_dec.to_value(u.mas / u.yr) == approx([3.0, 4.0])
    assert restored.parallax.to_value(u.mas) == approx([5.0, 10.0])
    assert restored.magnitude.to_value(u.mag) == approx([15.0, 16.0])
    assert restored.magnitude_error.to_value(u.mag) == approx([0.01, 0.02])
    assert restored.ra_error.to_value(u.mas) == approx([0.1, 0.2])
    assert restored.dec_error.to_value(u.mas) == approx([0.3, 0.4])
    assert restored.pm_ra_cosdec_error.to_value(
        u.mas / u.yr) == approx([0.5, 0.6])
    assert restored.pm_dec_error.to_value(
        u.mas / u.yr) == approx([0.7, 0.8])
    assert restored.parallax_error.to_value(u.mas) == approx([0.9, 1.0])
    assert np.all(
        restored.epoch == Time(['2016-01-01', '2016-01-01']))


def test_astrometric_catalog_qtable_validation():
    table = QTable({
        'ra': [10.0] * u.deg,
        'dec': [-5.0] * u.deg,
        'pm_ra_cosdec': [1.0] * u.mas / u.yr,
        'pm_dec': [3.0] * u.mas / u.yr,
        'parallax': [5.0] * u.mas,
    })

    with raises(ValueError, match='missing required columns'):
        AstrometricCatalog.from_qtable(table)


def test_astrometric_catalog_validation():
    values = {
        'ra': [10.0] * u.deg,
        'dec': [-5.0] * u.deg,
        'pm_ra_cosdec': [1.0] * u.mas / u.yr,
        'pm_dec': [3.0] * u.mas / u.yr,
        'parallax': [5.0] * u.mas,
        'epoch': Time('2016-01-01'),
    }

    with raises(ValueError, match='same shape'):
        AstrometricCatalog(**(values | {'parallax': [5.0, 6.0] * u.mas}))
    with raises(ValueError, match='non-negative'):
        AstrometricCatalog(**(values | {'parallax': [-1.0] * u.mas}))
    with raises(ValueError, match='ra_error.*same shape'):
        AstrometricCatalog(**(
            values | {'ra_error': [0.1, 0.2] * u.mas}))
    with raises(ValueError, match='parallax_error.*non-negative'):
        AstrometricCatalog(**(
            values | {'parallax_error': [-0.1] * u.mas}))
    with raises(TypeError, match='Observer'):
        AstrometricCatalog(**values).propagate(GCRS(obstime=values['epoch']))

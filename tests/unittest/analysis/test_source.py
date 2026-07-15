#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from astropy.coordinates import GCRS
from astropy.time import Time
import astropy.units as u
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import AstrometricCatalog, SourceCatalog
from warpfield.analysis.observer import GeoCentric


def test_source_catalog():
    source = SourceCatalog([1.0, 2.0], [3.0, 4.0])

    assert isinstance(source, zdx.Base)
    assert len(source) == 2
    assert source.take(jnp.array([1, 0]))[0] == approx([2.0, 1.0])
    assert source.take(jnp.array([1, 0]))[1] == approx([4.0, 3.0])
    assert len(jax.tree_util.tree_leaves(source)) == 2


def test_source_catalog_zodiax_update():
    source = SourceCatalog([1.0, 2.0], [3.0, 4.0])
    updated = source.set('ra', jnp.array([5.0, 6.0]))

    assert source.get('ra') == approx([1.0, 2.0])
    assert updated.get('ra') == approx([5.0, 6.0])


def test_source_catalog_shape_validation():
    with raises(ValueError, match='one-dimensional'):
        SourceCatalog([[1.0]], [2.0])
    with raises(ValueError, match='same shape'):
        SourceCatalog([1.0], [2.0, 3.0])


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
    with raises(TypeError, match='Observer'):
        AstrometricCatalog(**values).propagate(GCRS(obstime=values['epoch']))

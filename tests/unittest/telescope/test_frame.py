#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u

from warpfield.telescope.frame import GCGRS


@fixture
def target():
    ra = 35.0 * u.degree
    dec = 12.0 * u.degree
    pmra = 10.0 * u.mas / u.year
    pmdec = -20.0 * u.mas / u.year
    distance = Distance(parallax=10.0 * u.mas)
    epoch = Time(2016.0, format='decimalyear', scale='tcb')
    return SkyCoord(
        ra=ra, dec=dec, distance=distance,
        pm_ra_cosdec=pmra, pm_dec=pmdec,
        obstime=epoch
    )


def get_skycoord(x, frame):
    return SkyCoord(x.transform_to(frame).spherical)


def test_generate_gcgrs():
    gg = GCGRS()

    # observer is located at the center of the Earth
    obsgeoloc = gg.obsgeoloc.xyz.to_value('meter')
    assert obsgeoloc == approx([0, 0, 0])

    # observer is fixed to the Earth
    obsgeovel = gg.obsgeovel.xyz.to_value('meter/second')
    assert obsgeovel == approx([0, 0, 0])


def test_gcggrs_conversion(target):
    # separation angle between P(icrs) and P(gcrs)
    #   P(icrs): ICRS frame from the Solar System Barycenter
    #   P(gcrs): ICRS frame from the center of the Earth
    coo_icrs = get_skycoord(target, 'icrs')
    coo_gcrs = get_skycoord(target, 'gcrs')
    sep_ig = coo_icrs.separation(coo_gcrs)

    # separation angle between P(galactic) and P(gcgrs)
    #   P(galactic): Galactic frame from the Solar System Barycenter
    #   P(gcgrs): Galactic frame from the center of the Earth
    coo_gal = get_skycoord(target, 'galactic')
    coo_gcg = get_skycoord(target, 'gcgrs')
    sep_gg = coo_gal.separation(coo_gcg)

    # two separation angles should be identical within numerical error
    assert (sep_ig - sep_gg).degree == approx(0.0)

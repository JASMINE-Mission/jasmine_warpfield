#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gaia archive query and catalog adapter"""

import re

from astropy.coordinates import Angle, SkyCoord
from astropy.table import QTable, Table
from astropy.time import Time
import astropy.units as u
from astroquery.gaia import Gaia
import numpy as np

from ..catalog import AstrometricCatalog


__all__ = ['compile_from_gaia', 'query_gaia']


_CATALOG_PATTERN = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$'
)


def _column(table, name):
    """Find a Gaia column without depending on its letter case"""
    columns = {column.casefold(): column for column in table.colnames}
    try:
        return table[columns[name.casefold()]]
    except KeyError as error:
        raise ValueError(
            f'Gaia table is missing required column: {name}'
        ) from error


def _optional_column(table, name):
    """Find an optional Gaia column without depending on letter case"""
    columns = {column.casefold(): column for column in table.colnames}
    column = columns.get(name.casefold())
    return None if column is None else table[column]


def compile_from_gaia(table):
    """Convert a Gaia result table into an AstrometricCatalog

    The input should contain ``ra``, ``dec``, ``pmra``, ``pmdec``,
    ``parallax``, and ``ref_epoch``. Gaia ``pmra`` is interpreted as proper
    motion in right ascension including the cosine declination factor.
    Astrometric errors and G-band photometry are included when their columns
    are available. Magnitude errors are derived from G-band flux
    signal-to-noise ratios.
    """
    if not isinstance(table, Table):
        raise TypeError('`table` should be an Astropy Table instance.')
    table = QTable(table)

    epoch = u.Quantity(_column(table, 'ref_epoch'), unit=u.yr)
    flux_snr = _optional_column(table, 'phot_g_mean_flux_over_error')
    magnitude_error = None
    if flux_snr is not None:
        magnitude_error = (
            2.5 / np.log(10) / u.Quantity(flux_snr, unit=u.one) * u.mag
        )
    return AstrometricCatalog(
        ra=_column(table, 'ra'),
        dec=_column(table, 'dec'),
        pm_ra_cosdec=_column(table, 'pmra'),
        pm_dec=_column(table, 'pmdec'),
        parallax=_column(table, 'parallax'),
        epoch=Time(epoch.to_value(u.yr), format='jyear', scale='tcb'),
        magnitude=_optional_column(table, 'phot_g_mean_mag'),
        magnitude_error=magnitude_error,
        ra_error=_optional_column(table, 'ra_error'),
        dec_error=_optional_column(table, 'dec_error'),
        pm_ra_cosdec_error=_optional_column(table, 'pmra_error'),
        pm_dec_error=_optional_column(table, 'pmdec_error'),
        parallax_error=_optional_column(table, 'parallax_error'),
    )


def _build_query(center, radius, snr_limit, row_limit, catalog):
    """Construct an ADQL query for Gaia astrometric parameters"""
    if not isinstance(center, SkyCoord):
        raise TypeError('`center` should be a SkyCoord instance.')
    center = center.icrs
    if not center.isscalar:
        raise ValueError('`center` should be a scalar coordinate.')

    if not isinstance(radius, Angle):
        radius = Angle(radius, unit=u.deg)
    if not radius.isscalar or radius <= 0 * u.deg:
        raise ValueError('`radius` should be a positive scalar angle.')

    snr_limit = float(snr_limit)
    if not np.isfinite(snr_limit) or snr_limit < 0:
        raise ValueError('`snr_limit` should be finite and non-negative.')
    if (
        not isinstance(row_limit, int)
        or isinstance(row_limit, bool)
        or row_limit == 0
        or row_limit < -1
    ):
        raise ValueError('`row_limit` should be -1 or a positive integer.')
    if not isinstance(catalog, str) or not _CATALOG_PATTERN.fullmatch(catalog):
        raise ValueError('`catalog` should be a schema-qualified name.')

    top = '' if row_limit == -1 else f'TOP {row_limit} '
    return f"""
SELECT {top}
    source_id,
    ra,
    ra_error,
    dec,
    dec_error,
    pmra,
    pmra_error,
    pmdec,
    pmdec_error,
    parallax,
    parallax_error,
    phot_g_mean_mag,
    phot_g_mean_flux_over_error,
    ref_epoch
FROM {catalog}
WHERE
    1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE(
            'ICRS',
            {center.ra.degree},
            {center.dec.degree},
            {radius.to_value(u.deg)}
        )
    )
    AND parallax_over_error > {snr_limit}
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    AND parallax IS NOT NULL
    AND phot_g_mean_mag IS NOT NULL
    AND phot_g_mean_flux_over_error IS NOT NULL
"""


def query_gaia(
    center, radius, snr_limit=10.0, row_limit=-1, catalog='gaiadr3.gaia_source'
):
    """Query Gaia sources and return an AstrometricCatalog

    Arguments:
        center: Center of the search region as a scalar ``SkyCoord``.
        radius: Radius of the search region as an angle or degrees.
        snr_limit: Lower limit on ``parallax_over_error``.
        row_limit: Maximum number of rows, or ``-1`` for no limit.
        catalog: Schema-qualified Gaia archive table name.
    """
    query = _build_query(
        center,
        radius,
        snr_limit,
        row_limit,
        catalog,
    )
    job = Gaia.launch_job_async(query)
    return compile_from_gaia(job.get_results())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" handling astronomical sources """

from astropy.coordinates import SkyCoord, Angle, Distance
from astropy.time import Time
from astroquery.gaia import Gaia
import astropy.units as u
import matplotlib.pyplot as plt

from .util import get_projection

__debug_mode__ = False


def gaia_query_builder(pointing, radius, snr_limit, catalog='gaiaedr3'):
    """ construct a query string """
    return f"""
    SELECT
        source_id,
        ra,
        dec,
        phot_g_mean_mag,
        pmra,
        pmdec,
        parallax,
        ref_epoch
    FROM
        {catalog}.gaia_source
    WHERE
        1=CONTAINS(
          POINT('ICRS', {pointing.icrs.ra.deg}, {pointing.icrs.dec.deg}),
          CIRCLE('ICRS', ra, dec, {radius.deg}))
    AND
        parallax_over_error > {snr_limit}
    """


def retrieve_gaia_sources(pointing, radius, snr_limit=10.0, row_limit=-1):
    """ retrive sources around (lon, lat) from Gaia EDR3 catalog

    Arguments:
      pointing (SkyCoord):
          celestial coordinate of the search center.
      radius (float or Angle):
          search radius in degree.
      snr_limit (float, optional):
          lower limit of parallax over error.
      row_limit (int, optional):
          maximum number of records.

    Return:
      A list of neighbour souces (SkyCoord).
    """

    ## Get an acceess to the Gaia TAP+.
    ##   - Set the target table to Gaia EDR3.
    ##   - Remove the limit of the query number.
    Gaia.MAIN_GAIA_TABLE = 'gaiaedr3.gaia_source'
    Gaia.ROW_LIMIT = row_limit

    if not isinstance(radius, Angle):
        radius = Angle(radius, unit=u.degree)

    pointing = pointing.transform_to('icrs')
    query = gaia_query_builder(pointing, radius, snr_limit)

    res = Gaia.launch_job_async(query)

    if __debug_mode__ is True:
        print(res)

    record = res.get_results()
    epoch = Time(record['ref_epoch'].data, format='decimalyear')

    obj = SkyCoord(ra=record['ra'],
                   dec=record['dec'],
                   pm_ra_cosdec=record['pmra'],
                   pm_dec=record['pmdec'],
                   distance=Distance(parallax=record['parallax'].data * u.mas),
                   obstime=epoch)
    return obj


def display_sources(pointing, sources, **options):
    """ display sources around the specified coordinates

    Arguments:
      pointing (SkyCoord):
          the center of the search point.
      sources (SkyCoord):
          the list of sources.

    Returns:
      a tuble of (figure, axis).
    """

    proj = get_projection(pointing)
    frame = pointing.frame.name

    if frame == 'galactic':
        get_lon = lambda x: getattr(x, 'galactic').l
        get_lat = lambda x: getattr(x, 'galactic').b
        xlabel = 'Galactic Longitude'
        ylabel = 'Galactic Latitude'
    else:
        get_lon = lambda x: getattr(x, 'icrs').ra
        get_lat = lambda x: getattr(x, 'icrs').dec
        xlabel = 'Right Ascension'
        ylabel = 'Declination'

    title = options.pop('title', None)
    marker = options.pop('marker', 'x')
    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(111, projection=proj)
    axis.set_aspect(1.0)
    axis.set_position([0.13, 0.10, 0.85, 0.85])
    axis.scatter(get_lon(sources),
                 get_lat(sources),
                 transform=axis.get_transform(frame),
                 marker=marker,
                 label=title,
                 **options)
    axis.grid()
    if title is not None:
        axis.legend(bbox_to_anchor=[1, 1], loc='lower right', frameon=False)
    axis.set_xlabel(xlabel, fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14)
    return fig, axis


def display_gaia_sources(pointing, radius=0.1):
    """ display Gaia EDR3 sources around the coordinate

    Arguments:
      pointing (SkyCoord):
          celestial coordinate of the search center.
      radius (float or Angle):
          search radius in degree.

    Returns:
      a tuble of (figure, axis).
    """
    src = retrieve_gaia_sources(pointing, radius)
    return display_sources(pointing, src)

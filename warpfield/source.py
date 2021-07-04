#!/usr/bin/env python
# -*- coding: utf-8 -*-
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle, Distance
from astropy.time import Time
import astropy.units as u

from .util import get_projection


__debug_mode__ = False


def retrieve_gaia_sources(
    pointing, radius,
    snr_limit=10.0, row_limit=-1):
  ''' Retrive sources around (lon, lat) from Gaia EDR3 catalog.

  Parameters:
    pointing (SkyCoord)     : the center of the search point.
    radius (float or Angle) : the search radius in degree.
    snr_limit (float)       : the limiting SNR in the parallax.
    row_limit (int)         : the maximum number of records.

  Return:
    A list of neighbour souces (SkyCoord).
  '''
  ## Get an acceess to the Gaia TAP+.
  ##   - Set the target table to Gaia EDR3.
  ##   - Remove the limit of the query number.
  from astroquery.gaia import Gaia
  Gaia.MAIN_GAIA_TABLE = 'gaiaedr3.gaia_source'
  Gaia.ROW_LIMIT = row_limit

  query = '''
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
    gaiaedr3.gaia_source
  WHERE
    1=CONTAINS(
      POINT('ICRS', {ra}, {dec}),
      CIRCLE('ICRS', ra, dec, {radius}))
  AND
    parallax_over_error > {snr_limit}
  '''

  if not isinstance(radius, Angle):
    radius = Angle(radius, unit=u.degree)
  pointing = pointing.transform_to('icrs')

  res = Gaia.launch_job_async(query.format(
    ra=pointing.icrs.ra.deg, dec=pointing.icrs.dec.deg,
    radius=radius.deg, snr_limit=snr_limit))
  if __debug_mode__ is True: print(res)
  record = res.get_results()
  epoch = Time(record['ref_epoch'].data, format='decimalyear')

  obj = SkyCoord(
    ra=record['ra'], dec=record['dec'],
    pm_ra_cosdec=record['pmra'], pm_dec=record['pmdec'],
    distance=Distance(parallax=record['parallax'].data*u.mas),
    obstime=epoch)
  return obj


def display_sources(pointing, sources, title=None):
  ''' Display sources around the specified coordinates.

  Parameters:
    pointing (SkyCoord)     : the center of the search point.
    sources (SkyCoord)      : the list of sources.
  '''
  import matplotlib.pyplot as plt
  import numpy as np

  proj = get_projection(pointing)
  frame = pointing.frame.name

  if frame == 'galactic':
    get_lon = lambda x: getattr(x,'galactic').l
    get_lat = lambda x: getattr(x,'galactic').b
    xlabel  = 'Galactic Longitude'
    ylabel  = 'Galactic Latitude'
  else:
    get_lon = lambda x: getattr(x,'icrs').ra
    get_lat = lambda x: getattr(x,'icrs').dec
    xlabel  = 'Right Ascension'
    ylabel  = 'Declination'

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection=proj)
  ax.set_aspect(1.0)
  ax.set_position([0.13,0.10,0.85,0.85])
  ax.scatter(get_lon(sources), get_lat(sources),
      transform=ax.get_transform(frame), marker='x', label=title)
  ax.grid()
  if title is not None:
    ax.legend(bbox_to_anchor=[1,1], loc='lower right', frameon=False)
  ax.set_xlabel(xlabel, fontsize=14)
  ax.set_ylabel(ylabel, fontsize=14)
  plt.show()


def display_gaia_sources(pointing, radius=0.1):
  ''' Display Gaia EDR3 sources around the coordinate.

  Parameters:
    pointing (SkyCoord)     : the center of the search point.
    radius (float or Angle) : the search radius in degree.
  '''
  src = retrieve_gaia_sources(pointing, radius)
  display_sources(pointing, src)

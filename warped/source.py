#!/usr/bin/env python
# -*- coding: utf-8 -*-
from astropy.coordinates import SkyCoord, Longitude, Latitude
from astropy.time import Time
import astropy.units as u

from .util import Frame, get_projection


def retrieve_gaia_sources(
    lon, lat, radius=0.1, frame=Frame.galactic,
    snr_limit=10.0, row_limit=-1):
  ''' Retrive sources around (lon, lat) from Gaia EDR3 catalog.

  Parameters:
    lon (float or Longitude): the longitude of the frame center.
    lat (float or Latitude) : the latitude of the frame center.
    radius (float)          : the search radius in degree.
    frame (Frame)           : the coordinate frame (icrs or galactic).
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
    ra,dec,pmra,pmdec,parallax,ref_epoch
  FROM
    gaiaedr3.gaia_source
  WHERE
    1=CONTAINS(
      POINT('ICRS', {ra}, {dec}),
      CIRCLE('ICRS', ra, dec, {radius}))
  AND
    parallax_over_error > {snr_limit}
  '''

  if not isinstance(lon, Longitude):
    lon = Longitude(lon, unit=u.degree)
  if not isinstance(lat, Latitude):
    lat = Latitude(lat, unit=u.degree)

  coo = SkyCoord(lon, lat, frame=frame.value)
  res = Gaia.launch_job_async(query.format(
    ra=coo.icrs.ra.deg,dec=coo.icrs.dec.deg,
    radius=radius,snr_limit=snr_limit))
  print(res)
  record = res.get_results()
  epoch = Time(record['ref_epoch'].data, format='decimalyear')

  return SkyCoord(
    ra=record['ra'], dec=record['dec'],
    pm_ra_cosdec=record['pmra'], pm_dec=record['pmdec'],
    distance=1000/record['parallax']*u.pc, obstime=epoch)


def display_sources(lon, lat, sources, frame=Frame.galactic):
  ''' Display sources around the specified coordinates.

  Parameters:
    lon (float or Longitude): the longitude of the frame center.
    lat (float or Latitude) : the latitude of the frame center.
    radius (float)          : the search radius in degree.
    frame (Frame)           : the coordinate frame.
  '''
  import matplotlib.pyplot as plt
  import numpy as np
  current = sources.apply_space_motion(Time.now())

  proj = get_projection(lon, lat, frame)

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(projection=proj)
  ax.scatter(sources.galactic.l, sources.galactic.b,
    transform=ax.get_transform(frame.value), marker='x', label='2016.0')
  ax.scatter(current.galactic.l, current.galactic.b,
    transform=ax.get_transform(frame.value), marker='+', label='current')
  ax.grid()
  ax.legend(bbox_to_anchor=[1,1], loc='lower right', ncol=2, frameon=False)
  ax.set_xlabel('Galactic Longitude (deg)', fontsize=14)
  ax.set_ylabel('Galactic Latitude (deg)', fontsize=14)
  fig.tight_layout()
  plt.show()


def display_gaia_sources(
    lon=2.0, lat=0.0, radius=0.1, frame=Frame.galactic):
  ''' Display Gaia EDR3 sources around the coordinate.

  Parameters:
    lon (float or Longitude): the longitude of the frame center.
    lat (float or Latitude) : the latitude of the frame center.
    radius (float)          : the search radius in degree.
    frame (Frame)           : the coordinate frame.
  '''
  src = retrieve_gaia_sources(lon, lat, radius=radius, frame=frame)
  display_sources(lon, lat, src, frame)

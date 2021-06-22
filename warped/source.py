#!/usr/bin/env python
# -*- coding: utf-8 -*-
from astropy.coordinates import SkyCoord, Longitude, Latitude
from astropy.time import Time
import astropy.units as u



def retrieve_gaia_sources(
    ra, dec, size=0.6, snr_limit=5.0):
  ''' Retrive sources around (ra, dec) from Gaia EDR3 catalog.

  Parameters:
    ra (float or Longitude): the right ascension of the field.
    dec (float or Latitude): the declinaton of the field.
    size (float)           : the size of the search box in degree.
    snr_limit (float)      : the limiting SNR in the parallax.

  Return:
    A list of neighbour souces (SkyCoord).
  '''
  from astroquery.gaia import Gaia
  Gaia.MAIN_GAIA_TABLE = 'gaiaedr3.gaia_source'
  if not isinstance(ra, Longitude):
    ra = Longitude(ra, unit=u.degree)
  if not isinstance(dec, Latitude):
    dec = Latitude(dec, unit=u.degree)

  size = size * u.degree
  coo = SkyCoord(ra=ra, dec=dec, frame='icrs')
  res = Gaia.query_object_async(coordinate=coo, width=size, height=size)
  res = res[res['parallax_over_error']>snr_limit]
  epoch = Time(res['ref_epoch'], format='decimalyear')

  return SkyCoord(
    ra=res['ra'], dec=res['dec'],
    pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'],
    distance=1000/res['parallax']*u.pc, obstime=epoch)
